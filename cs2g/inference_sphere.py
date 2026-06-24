import os
import sys
import argparse
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from collections import OrderedDict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.crossgeo_geo_ldm_diffusion.crossgeo_ddim import CrossGEODDIMSamplerSphere
from models.CVUSA_geo_ldm_diffusion.ddim_CVUSA import CVUSA_DDIMSamplerSphere
from utils.util import instantiate_from_config
import yaml


def disabled_train(self, mode=True):
    return self


def load_model_from_config(config_path, ckpt_path, device="cuda"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    model = instantiate_from_config(config["model"])
    
    if ckpt_path and os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        state_dict = checkpoint.get("state_dict", checkpoint)
        
        model_state = OrderedDict()
        for k, v in state_dict.items():
            model_state[k] = v
        
        missing, unexpected = model.load_state_dict(model_state, strict=False)
        print(f"Loaded checkpoint: {ckpt_path}")
        if missing:
            print(f"  Missing keys: {len(missing)}")
        if unexpected:
            print(f"  Unexpected keys: {len(unexpected)}")
    
    model = model.to(device)
    model.eval()
    model.train = disabled_train
    
    for param in model.parameters():
        param.requires_grad = False
    
    return model


def inference(config_path, ckpt_path, sat_img_path, output_path, device="cuda", steps=50, scale=7.5):
    model = load_model_from_config(config_path, ckpt_path, device)
    
    sat_img = Image.open(sat_img_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    sat_tensor = transform(sat_img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        cond_label = model.condition_model_sat(sat_tensor)
        cond_label = cond_label[:, 1:, :]
        
        dummy = torch.zeros(1, 3, 256, 256).to(device)
        dummy = dummy * 2 - 1
        with torch.no_grad():
            enc = model.pre_AE_model.encode(dummy).sample()
        _, c, h, w = enc.shape
        
        shape = [1, c, h, w]
        x_T = torch.randn(shape, device=device)
        
        shape_batch = [c, h, w]
        
        sampler = CrossGEODDIMSamplerSphere(model.DDPM, model.pre_AE_model, model.scale_factor)
        sampler._sat_img = sat_tensor
        
        samples_ddim, _ = sampler.sample(
            S=steps,
            conditioning=cond_label,
            batch_size=1,
            shape=shape_batch,
            verbose=True,
            unconditional_guidance_scale=scale,
            unconditional_conditioning=None,
            eta=1,
            x_T=x_T,
            temperature=1,
        )
        
        samples_ddim = samples_ddim * (1 / model.scale_factor)
        generated = model.pre_AE_model.decode(samples_ddim)
        generated = torch.clamp((generated + 1.0) / 2.0, min=0.0, max=1.0)
        
        result = generated[0].cpu()
        to_pil = transforms.ToPILImage()
        result_img = to_pil(result)
        result_img.save(output_path)
        print(f"Saved result to: {output_path}")
    
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--sat", type=str, required=True, help="Path to satellite image")
    parser.add_argument("--output", type=str, default="outputs/output_result.png")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--scale", type=float, default=7.5)
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    
    inference(
        config_path=args.config,
        ckpt_path=args.ckpt,
        sat_img_path=args.sat,
        output_path=args.output,
        device=args.device,
        steps=args.steps,
        scale=args.scale,
    )
