<h1 align="center">ControlS2S: 可控卫星到街景图像生成</h1>

<p align="center">
  <a href="https://arxiv.org/pdf/2502.03498">
    <img src="https://img.shields.io/badge/Paper-ICLR%202025-olive" alt="ICLR 2025">
  </a>
  <a href="https://github.com/zexianghui/CS2S_pose_environment">
    <img src="https://img.shields.io/github/stars/zexianghui/CS2S_pose_environment?color=yellow" alt="GitHub Stars">
  </a>
  <a href="https://github.com/zexianghui/CS2S_pose_environment/blob/main/LICENSE">
    <img src="https://img.shields.io/github/license/zexianghui/CS2S_pose_environment?color=blue" alt="License">
  </a>
</p>

<p align="center">
  <img src="assets/Framework_2.png" alt="ControlS2S Framework" width="85%">
</p>

---

## 项目简介

本项目是 ICLR 2025 论文 **"Controllable Satellite-to-Street-View Synthesis with Precise Pose Alignment and Zero-Shot Environmental Control"** 的代码实现，专注于从卫星图像生成街景图像的跨视角合成任务。该工作基于 Stable Diffusion 潜在扩散模型，通过**迭代单应性调整 (IHA)** 实现精确的姿态对齐，并结合**文本引导采样**支持光照、天气等环境条件的零样本控制。

本仓库基于 [**CS2S（ControlS2S）**](https://github.com/zexianghui/CS2S_pose_environment) 官方实现进行重构与增强，其扩散模型骨干借鉴了 [**CrossGeo**](https://github.com/YujiaoShi/HighlyAccurate) 的跨地理域泛化思想与数据集处理范式，球面几何投影部分则参考了 [**SPND**](https://github.com/chronos123/SpND) 相关工作，修复了原版中多项影响生成效果的逻辑缺陷，并引入了统一的训练/推理入口和动态潜在空间适配。

> **核心论文**: [Controllable Satellite-to-Street-View Synthesis with Precise Pose Alignment and Zero-Shot Environmental Control](https://arxiv.org/pdf/2502.03498)  
> **项目仓库**: [https://github.com/zexianghui/CS2S_pose_environment](https://github.com/zexianghui/CS2S_pose_environment)

---

## 核心特性

| 特性 | 说明 |
|------|------|
| **迭代单应性调整 (IHA)** | 在去噪过程的每一步动态校正姿态，确保卫星-街景空间一致性 |
| **球面 ControlNet** | 专为全景图像设计的 388M 参数 ControlNet，采用可变形卷积适配球面几何 |
| **零样本环境控制** | 通过 CLIP 文本嵌入实现光照、季节、天气（如 "in autumn"、"at night"）的条件控制 |
| **CrossGeo 跨域泛化** | 支持 CVUSA（美国）、KITTI（德国）、VIGOR（四城市）三个数据集的联合零样本迁移训练 |
| **动态潜在空间适配** | 运行时从 VAE 编码器自动推导潜在形状，避免硬编码 `(4,16,64)` 带来的跨数据集崩溃 |
| **Classifier-Free Guidance** | 支持无分类器引导采样，可调节 `unconditional_guidance_scale` 控制生成质量与多样性平衡 |

---

## 框架总览

```
┌──────────────────────────────────────────────────┐
│                  卫星图像 (256×256)                 │
│                      │▲                           │
│          ┌───────────▼┴───────────┐               │
│          │   VIT_224 卫星编码器    │               │
│          │   (condition_model_sat) │               │
│          └───────────┬───────────┘               │
│                      │ [B,257,768]               │
│          ┌───────────▼───────────┐               │
│          │    Stable Diffusion   │               │
│          │  UNet + CrossAttention │◄── CLIP Text  │
│          │  + SphericalControlNet │    Embedding  │
│          └───────────┬───────────┘               │
│                      │ 潜在空间去噪                │
│                      │ 每步执行 IHA 姿态校正       │
│          ┌───────────▼───────────┐               │
│          │    VAE Decoder        │               │
│          │    (pre_AE_model)     │               │
│          └───────────┬───────────┘               │
│                      │▲                           │
│             街景全景图 (128×512 / 256×256)         │
└──────────────────────────────────────────────────┘
```

### 两阶段训练范式

- **Stage 1（基础训练）**: 仅训练 UNet 去噪器 + 卫星条件编码器，不启用 SphericalControlNet。建立卫星特征到街景潜在表示的基础映射。
- **Stage 2（球面微调）**: 加载 Stage 1 权重，冻结 UNet 和卫星编码器，仅训练新增的 SphericalControlNet。通过可变形卷积捕获全景图像的球面几何特征，进一步提升生成质量。

---

## 快速开始

### 1. 环境配置

```bash
conda create -n ControlS2S python=3.8
conda activate ControlS2S
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install pytorch-lightning==1.9.5
pip install omegaconf einops matplotlib opencv-python==4.9.0.80 scikit-image==0.21.0 kornia prefetch_generator lpips pytorch-msssim
pip install -e git+https://github.com/CompVis/taming-transformers.git@master#egg=taming-transformers
pip install -e git+https://github.com/openai/CLIP.git@main#egg=clip

# 下载 Stable Diffusion v1.4 权重
mkdir ckpt && cd ckpt
curl -L https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4.ckpt -o sd-v1-4.ckpt
```

### 2. 数据集准备

```bash
dataset/
├── CVUSA/          # 美国街景 (512×128 全景图 + 256×256 卫星图)
├── VIGOR/          # 四城市街景 (含 GPS 与相机参数)
├── KITTI_location/ # 德国街景 (含相机内外参)
└── CrossGeo/       # 跨域泛化数据集 (含 quad_info.json 3D 参数)
```

| 数据集 | 来源 |
|--------|------|
| CVUSA | [Sat2Density](https://github.com/qianmingduowan/Sat2Density) |
| VIGOR | [VIGOR](https://github.com/Jeff-Zilence/VIGOR) & [SliceMatch](https://github.com/tudelft-iv/SliceMatch) |
| KITTI | [HighlyAccurate](https://github.com/YujiaoShi/HighlyAccurate) |

### 3. 训练

使用统一入口 `train_all.py`，支持所有数据集和两阶段训练：

```bash
# CrossGeo Stage1（默认数据集）
python train_all.py --dataset crossgeo --stage 1 --max-epochs 50 --batch-size 2

# CrossGeo Stage2（启用 SphericalControlNet）
python train_all.py --dataset crossgeo --stage 2 --max-epochs 50 --batch-size 2 --stage1-ckpt outputs/stage1/<run>/checkpoints/last.ckpt

# CVUSA 训练
python train_all.py --dataset CVUSA --stage 1 --max-epochs 50 --batch-size 2

# KITTI 训练
python train_all.py --dataset KITTI --stage 1 --max-epochs 50 --batch-size 2

# VIGOR 训练
python train_all.py --dataset VIGOR --stage 1 --max-epochs 50 --batch-size 2
```

**主要训练参数**:

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--dataset` | 数据集: CVUSA / KITTI / VIGOR / crossgeo | `crossgeo` |
| `--stage` | 训练阶段: 1=基础 / 2=球面微调 | `1` |
| `--devices` | GPU 设备号 | `"0"` |
| `--max-epochs` | 最大训练轮数 | `50` |
| `--batch-size` | 批次大小 | `2` |
| `--accumulate-grad-batches` | 梯度累积步数 | `4` |
| `--lr` | 学习率（覆盖配置文件） | 配置文件中的值 |
| `--stage1-ckpt` | Stage2 训练时加载的 Stage1 checkpoint | `None` |
| `--seed` | 随机种子 | `24` |

### 4. 推理

#### CrossGeo 推理（支持 SphericalControlNet）

```bash
python inference_sphere.py \
  --config configs/Boost_Sat2Den/crossgeo_geo_ldm_sphere.yaml \
  --ckpt outputs/stage2/<run>/checkpoints/last.ckpt \
  --sat path/to/satellite_image.png \
  --output result/output.png \
  --steps 50 --scale 7.5
```

#### CVUSA / KITTI / VIGOR 推理与可视化

```bash
# CVUSA
python visualization.py \
  --base configs/Boost_Sat2Den/CVUSA_geo_ldm.yaml \
  --devices 0, --strategy ddp \
  --test result/CVUSA_ckpt/checkpoints/CVUSA.ckpt \
  --function 1

# KITTI
python visualization.py \
  --base configs/Boost_Sat2Den/KITTI_geo_ldm.yaml \
  --devices 0, --strategy ddp \
  --test result/KITTI_ckpt/checkpoints/KITTI.ckpt \
  --function 1

# VIGOR
python visualization.py \
  --base configs/Boost_Sat2Den/VIGOR_geo_ldm.yaml \
  --devices 0, --strategy ddp \
  --test result/VIGOR_ckpt/checkpoints/VIGOR.ckpt \
  --function 1
```

**可视化模式 (`--function`)**：

| 值 | 模式 | 说明 |
|----|------|------|
| `1` | 基础生成 | 卫星图 → 街景图，含 IHA 姿态校正 |
| `2` | 姿态增强 | 额外叠加地面特征引导的姿态校正 |
| `3+` | 环境控制 | 启用文本引导（如 "in autumn", "at night"） |

---

## 项目结构

```
cs2g/
├── configs/                        # OmegaConf YAML 配置文件
│   ├── autoencoder/                # VAE 自编码器配置
│   └── Boost_Sat2Den/              # 主模型配置 (CVUSA/KITTI/VIGOR/CrossGeo)
│       └── train/                  # 训练专用配置
├── models/                         # 模型实现
│   ├── CVUSA_geo_ldm/              # CVUSA 条件模型 (训练逻辑)
│   ├── CVUSA_geo_ldm_diffusion/    # CVUSA 扩散模型 (DDPM + DDIM 采样器 + UNet)
│   ├── KITTI_geo_ldm/              # KITTI 条件模型
│   ├── KITTI_geo_ldm_diffusion/    # KITTI 扩散模型
│   ├── VIGOR_geo_ldm/              # VIGOR 条件模型
│   ├── VIGOR_geo_ldm_diffusion/    # VIGOR 扩散模型
│   ├── crossgeo_geo_ldm/           # CrossGeo 条件模型
│   │   ├── crossgeo_txt_control.py # CrossGEO 主类 (含两阶段训练逻辑)
│   │   └── grd_condition_model.py  # 地面条件编码器 (UNet)
│   ├── crossgeo_geo_ldm_diffusion/ # CrossGeo 扩散模型
│   │   ├── crossgeo_ddim.py        # DDIM 采样器 (含 SphericalControlNet 分支)
│   │   ├── crossgeo_latent_diffusion.py  # DDPM 扩散过程
│   │   └── openaimodel.py          # UNet 去噪器
│   ├── spherical_controlnet.py     # 球面 ControlNet (SphereDeformableConv2d)
│   ├── geometry/                   # 几何变换模块 (sat2grd, grd2sat 等)
│   ├── loss_fun/                   # 损失函数 (triplet_loss 等)
│   ├── eval/                       # 评估指标
│   ├── autoencoder/                # 自编码器实现
│   └── swin_transformer_v2/        # Swin Transformer V2 实现
├── ldm/                            # 潜在扩散模型基础库
│   ├── modules/                    # 注意力机制 (CrossAttention, SpatialTransformer)
│   └── models/diffusion/           # DDIM/DDPM/PLMS 采样器基类
├── dataloader/                     # 数据加载器
│   ├── crossgeo_txt.py             # CrossGeo 数据集 (含 quad_info.json 3D 参数)
│   ├── CVUSA_txt.py                # CVUSA 数据集
│   ├── KITTI_wo_loc.py             # KITTI 数据集
│   └── VIGOR_corr.py               # VIGOR 数据集
├── utils/                          # 工具函数 (instanciate_from_config 等)
├── train_all.py                    # ★ 统一训练入口 (Stage1 + Stage2)
├── visualization.py                # ★ 可视化/批量推理入口
├── inference_sphere.py             # CrossGeo 单图推理
└── main.py                         # 原始训练入口
```

---

## 参考项目

本项目作为 ControlS2S（ICLR 2025）的重构增强版本，在实现过程中参考并借鉴了以下优秀工作的关键组件与思想：

| 项目 | 说明 | 链接 |
|------|------|------|
| **CS2S (ControlS2S)** | ICLR 2025 官方实现，提出了 IHA 姿态对齐与文本引导环境控制 | [GitHub](https://github.com/zexianghui/CS2S_pose_environment) |
| **CrossGeo** | 跨地理域卫星-街景生成数据集与泛化方法，提供了 CVUSA/KITTI/VIGOR 的联合训练范式 | 参见 [HighlyAccurate](https://github.com/YujiaoShi/HighlyAccurate)、[VIGOR](https://github.com/Jeff-Zilence/VIGOR) |
| **SPND** | 球面全景图像的可变形卷积投影模型，启发了 SphericalControlNet 中的 `SphereDeformableConv2d` 设计 | [GitHub](https://github.com/chronos123/SpND) |
| **Stable Diffusion** | CompVis 的潜在扩散模型，提供 VAE 编码器/解码器与 DDPM 基础架构 | [GitHub](https://github.com/CompVis/stable-diffusion) |
| **Sat2Density** | 卫星到街景密度图生成，为 CVUSA/VIGOR 数据集提供预处理流水线 | [GitHub](https://github.com/qianmingduowan/Sat2Density) |
| **Sat2Str** | 卫星到全景街景合成，提供了跨视角注意力机制基础 | [GitHub](https://github.com/YujiaoShi/Sat2StrPanoramaSynthesis) |

---

## 模型检查点

官方预训练权重下载（适用于 CVUSA / KITTI / VIGOR）：

- [校内网盘](http://pan.njust.edu.cn/#/link/zgGzHgpgIuoBBFGIv22v)
- [OneDrive](https://1drv.ms/f/c/86d953bfc66eb903/IgBZnCNAt101TJhiB-uc49pSAbAkpVMQoeo3frjvjTkueXA?e=c9baJ8)

---

## 引用

如果本项目对您的研究有帮助，请引用：

```bibtex
@article{ze2025controllable,
  title={Controllable Satellite-to-Street-View Synthesis with Precise Pose Alignment and Zero-Shot Environmental Control},
  author={Ze, Xianghui and Song, Zhenbo and Wang, Qiwei and Lu, Jianfeng and Shi, Yujiao},
  journal={arXiv preprint arXiv:2502.03498},
  year={2025}
}
```

---

## 联系方式

如有问题，欢迎联系 [`zexh@njust.edu.cn`](mailto:zexh@njust.edu.cn)。
