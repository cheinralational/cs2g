import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from contextlib import contextmanager
from einops import rearrange

from ldm.modules.diffusionmodules.util import conv_nd, linear, avg_pool_nd, zero_module, normalization, timestep_embedding
from ldm.modules.diffusionmodules.openaimodel import AttentionBlock, ResBlock, TimestepEmbedSequential, Downsample, Upsample
from ldm.modules.CVUSA_attention import SpatialTransformer
from utils.util import exists


class SphereDeformableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True):
        super().__init__()
        self.offset_conv = nn.Conv2d(in_channels, 2 * kernel_size * kernel_size,
                                      kernel_size=kernel_size, stride=stride,
                                      padding=padding, dilation=dilation, groups=groups, bias=True)
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.kernel_size = kernel_size

    def circular_pad(self, x):
        _, _, h, w = x.shape
        pad = self.padding
        x = F.pad(x, (pad, pad, 0, 0), mode='circular')
        if pad > 0:
            x = F.pad(x, (0, 0, pad, pad), mode='constant', value=0)
        return x

    def forward(self, x):
        offset = self.offset_conv(x)
        _, _, h, w = x.shape
        kh, kw = self.kernel_size, self.kernel_size
        N = 2 * kh * kw
        offset = offset.permute(0, 2, 3, 1).reshape(-1, h, w, N, 1)
        offset = torch.tanh(offset) * 2

        ys, xs = torch.meshgrid(
            torch.arange(0, h, dtype=torch.float32, device=x.device),
            torch.arange(0, w, dtype=torch.float32, device=x.device),
            indexing='ij'
        )
        grid_y = (ys.unsqueeze(-1) + offset[..., 0::2].squeeze(-1)) * 2.0 / max(h - 1, 1) - 1.0
        grid_x = (xs.unsqueeze(-1) + offset[..., 1::2].squeeze(-1)) * 2.0 / max(w - 1, 1) - 1.0
        grid = torch.stack([grid_x, grid_y], dim=-1)

        output = F.conv2d(x, self.weight, self.bias, self.stride, 0, self.dilation, self.groups)
        return output


class DeformableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True):
        super().__init__()
        self.offset_conv = nn.Conv2d(in_channels, 2 * kernel_size * kernel_size,
                                      kernel_size=kernel_size, stride=stride,
                                      padding=padding, dilation=dilation, groups=groups, bias=True)
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.kernel_size = kernel_size

    def forward(self, x):
        offset = self.offset_conv(x)
        B, _, H, W = offset.shape
        kh, kw = self.kernel_size, self.kernel_size
        offset = offset.reshape(B, 2, kh * kw, H, W)
        offset = torch.tanh(offset) * 2

        from torchvision.ops import deform_conv2d
        return deform_conv2d(x, offset, self.weight, self.bias, self.stride, self.padding, self.dilation)


class ResBlockDeformableBias(ResBlock):
    def __init__(self, channels, emb_channels, out_channels=None, use_conv=False,
                 use_scale_shift_norm=False, dropout=0, use_deformable=False, use_circular=False):
        super().__init__(channels, emb_channels, out_channels, use_conv, use_scale_shift_norm, dropout)
        self.use_deformable = use_deformable
        self.use_circular = use_circular
        if use_deformable:
            if use_circular:
                self.conv1 = SphereDeformableConv2d(channels, channels, 3, 1, 1)
            else:
                self.conv1 = DeformableConv2d(channels, channels, 3, 1, 1)

    def forward(self, x, emb):
        return super().forward(x, emb)


class SphericalControlNet(nn.Module):
    def __init__(self, in_channels=4, model_channels=320, hint_channels=3, num_res_blocks=2,
                 attention_resolutions=[4, 2, 1], dropout=0, channel_mult=[1, 2, 4, 4],
                 conv_resample=True, dims=2, num_heads=8, use_spatial_transformer=True,
                 transformer_depth=1, context_dim=768, use_checkpoint=False, legacy=False):
        super().__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_heads = num_heads
        self.use_spatial_transformer = use_spatial_transformer
        self.transformer_depth = transformer_depth
        self.context_dim = context_dim

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        self.input_hint_block = TimestepEmbedSequential(
            conv_nd(dims, hint_channels, model_channels, 3, padding=1)
        )

        self.input_blocks = nn.ModuleList([
            TimestepEmbedSequential(conv_nd(dims, 4, model_channels, 3, padding=1))
        ])

        self.zero_convs = nn.ModuleList([zero_module(conv_nd(dims, model_channels, model_channels, 1))])

        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1

        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                out_ch = model_channels * mult
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=out_ch,
                        dims=dims,
                        use_scale_shift_norm=False,
                    )
                ]
                ch = out_ch
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(ch, num_heads=num_heads)
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self.zero_convs.append(zero_module(conv_nd(dims, ch, ch, 1)))
                input_block_chans.append(ch)

            if level != len(channel_mult) - 1:
                self.input_blocks.append(
                    TimestepEmbedSequential(Downsample(ch, conv_resample, dims=dims))
                )
                input_block_chans.append(ch)
                ds *= 2
                self.zero_convs.append(zero_module(conv_nd(dims, ch, ch, 1)))

        self.middle_block = TimestepEmbedSequential(
            ResBlock(ch, time_embed_dim, dropout, dims=dims, use_scale_shift_norm=False),
            SpatialTransformer(ch, num_heads, dims // 2, depth=transformer_depth, context_dim=context_dim),
            ResBlock(ch, time_embed_dim, dropout, dims=dims, use_scale_shift_norm=False),
        )
        self.middle_block_out = zero_module(conv_nd(dims, ch, ch, 1))

    def forward(self, x, hint, timesteps, context):
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)

        guided_hint = self.input_hint_block(hint, emb, context)

        h = x
        h = self.input_blocks[0](h, emb, context)
        h = h + guided_hint

        for module in self.input_blocks[1:]:
            h = module(h, emb, context)

        h = self.middle_block(h, emb, context)
        return [self.middle_block_out(h)]
