<h1 align="center">CS2G — 可控卫星到街景图像生成</h1>

<p align="center">
  <img src="assets/Framework_2.png" alt="Framework Overview" width="85%">
</p>

---

## 项目简介

本项目实现了一种**从卫星图像生成街景图像**的跨视角合成方法，基于 Stable Diffusion 潜在扩散模型（Latent Diffusion Model, LDM），核心包括两个训练阶段与两种推理模式。

相关工作参考：

- [**CS2S (ControlS2S)**](https://github.com/zexianghui/CS2S_pose_environment) — 迭代单应性调整（IHA）姿态对齐与文本引导环境控制
- [**CrossGeo**](https://github.com/YujiaoShi/HighlyAccurate) — 跨地理域泛化与 CVUSA/KITTI/VIGOR 联合训练范式
- [**SPND**](https://github.com/chronos123/SpND) — 球面全景图像的可变形卷积投影模型

---

## 核心原理

### 输入输出

| 项 | 说明 |
|----|------|
| **输入** | RGB 卫星图像（任意分辨率，自动 resize 到 256×256），可选文本提示（如 "in autumn"、"at night"） |
| **输出** | 对应位置的 RGB 街景图像（CVUSA 为 128×512 全景，CrossGeo 为 256×256 地面视角） |
| **基础工具** | Stable Diffusion v1.4（VAE 编码/解码 + UNet 去噪骨架）、OpenAI CLIP（文本嵌入）、VIT_224（卫星条件编码器） |

### 训练范式

两阶段训练，Stage 1 训练 UNet + 卫星条件编码器，Stage 2 冻结基础模型仅微调 SphericalControlNet：

| 阶段 | 训练模块 | 冻结模块 |
|------|---------|---------|
| **Stage 1** | UNet 去噪器 + VIT_224 卫星编码器 + 地面条件模型 | VAE（SD v1.4 权重） |
| **Stage 2** | SphericalControlNet（约 388M 参数） | VAE + UNet + VIT_224 |

### 创新点

**1. 球面 ControlNet — 全景几何约束**

普通卷积对全景图左右边界的拼缝不敏感。本项目的 `SphericalControlNet` 引入 `SphereDeformableConv2d`：水平方向使用 `F.pad(mode='circular')` 实现循环填充，同时用可学习偏移量（offset）自适应调整采样位置，使 ControlNet 能够捕获全景图像的球面几何特性。

**2. 动态潜在空间适配**

不同编码器产生的潜在表示尺寸不同（CVUSA 为 `[4,16,64]`，CrossGeo 为 `[4,32,32]`），硬编码会导致跨数据集崩溃。本项目在运行时从 VAE encoder 输出中动态推导潜在形状（`enc.shape`），所有采样器、几何变换中的形状参数均由实际编码推导，消除了对特定数据集的依赖。

**3. 统一训练入口与 CG 分支修复**

原版各数据集使用不同训练脚本，且 Classifier-Free Guidance 分支存在未定义变量的运行时崩溃。本项目以 `train_all.py` 统一四数据集的两阶段训练，并修复了所有 DDIM 采样器中 CG 分支的引用错误、噪声生成缓存溢出等底层缺陷。

---

## 依赖环境

- Python 3.8+
- PyTorch 1.13.1 + CUDA 11.7（>= 12GB 显存）
- PyTorch Lightning 1.9.5
- Stable Diffusion v1.4 checkpoint

### 安装

```bash
conda create -n cs2g python=3.8
conda activate cs2g
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install pytorch-lightning==1.9.5
pip install omegaconf einops matplotlib opencv-python==4.9.0.80 scikit-image==0.21.0 kornia prefetch_generator lpips pytorch-msssim
pip install -e git+https://github.com/CompVis/taming-transformers.git@master#egg=taming-transformers
pip install -e git+https://github.com/openai/CLIP.git@main#egg=clip

mkdir ckpt && cd ckpt
curl -L https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4.ckpt -o sd-v1-4.ckpt
```

---

## 数据集

| 数据集 | 数据格式 | 来源 |
|--------|---------|------|
| **CVUSA** | 卫星 256×256 → 全景 128×512，美国 | [Sat2Density](https://github.com/qianmingduowan/Sat2Density) |
| **KITTI** | 卫星 → 前视相机 + 相机内外参，德国 | [HighlyAccurate](https://github.com/YujiaoShi/HighlyAccurate) |
| **VIGOR** | 卫星 → 全景 + GPS/相机参数，4 个美国城市 | [VIGOR](https://github.com/Jeff-Zilence/VIGOR) |
| **CrossGeo** | 卫星 256×256 → 地面 256×256 + UAV，含 3D quad_info.json | 来自 [CrossGeo](https://github.com/YujiaoShi/HighlyAccurate) 项目 |

数据集目录结构：

```
dataset/
├── CVUSA/
├── KITTI_location/
├── VIGOR/
└── crossgeo/
    └── data/
        ├── pair_0/
        │   ├── ground_1_rgb.jpg
        │   ├── ground_1_satellite.png
        │   ├── ground_2_rgb.jpg
        │   ├── ground_2_satellite.png
        │   └── quad_info.json
        ├── pair_1/
        └── ...
```

---

## 运行方式

### 验证环境

```bash
python _verify_crossgeo.py      # Stage-1 验证
python _verify_stage2.py        # Stage-2 验证
```

### Stage 1 训练（基础扩散模型）

```bash
python cs2g/train_all.py \
    --dataset crossgeo \
    --stage 1 \
    --devices 0 \
    --max-epochs 50 \
    --batch-size 2 \
    --accumulate-grad-batches 4 \
    --lr 1e-5
```

训练产物保存在 `outputs/stage1/<时间戳>/checkpoints/`，包含 `last.ckpt` 和按 L1_loss 最优的 epoch checkpoint。

### Stage 2 训练（SphericalControlNet）

```bash
python cs2g/train_all.py \
    --dataset crossgeo \
    --stage 2 \
    --stage1-ckpt outputs/stage1/<run_name>/checkpoints/last.ckpt \
    --devices 0 \
    --max-epochs 50 \
    --batch-size 2 \
    --accumulate-grad-batches 4 \
    --lr 1e-5
```

训练产物保存在 `outputs/stage2/<时间戳>/checkpoints/`。

### 推理（单张卫星图 → 街景图）

```bash
python cs2g/inference_sphere.py \
    --config cs2g/configs/Boost_Sat2Den/crossgeo_geo_ldm_sphere.yaml \
    --ckpt outputs/stage2/<run_name>/checkpoints/last.ckpt \
    --sat dataset/crossgeo/data/pair_0/ground_1_satellite.png \
    --output output.png \
    --steps 50 \
    --scale 7.5
```

**推理参数说明**：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--config` | YAML 配置文件路径 | 必填 |
| `--ckpt` | Stage-2 训练得到的 checkpoint | 必填 |
| `--sat` | 输入卫星图像路径（任意分辨率，自动 resize 到 256×256） | 必填 |
| `--output` | 输出街景图像路径 | `outputs/output_result.png` |
| `--steps` | DDIM 采样步数 | `50` |
| `--scale` | CFG 引导强度 | `7.5` |
| `--device` | 运行设备 | `cuda` |

### 一键脚本（Windows）

项目根目录提供 `.bat` 脚本用于 Windows 环境：

| 脚本 | 功能 |
|------|------|
| `00_verify.bat` | 验证数据、模型、checkpoint 是否就绪 |
| `01_train_stage1.bat` | Stage-1 训练（支持命令行传参 `01_train_stage1.bat <epochs> <batch>`） |
| `02_train_stage2.bat` | Stage-2 训练（自动查找 Stage-1 checkpoint） |
| `03_inference.bat` | 单图推理（自动查找 Stage-2 checkpoint，自动映射 config YAML） |

---

## 项目结构

```
cs2g/
├── configs/
│   └── Boost_Sat2Den/
│       ├── crossgeo_geo_ldm_stage1.yaml    # CrossGeo Stage-1 配置
│       ├── crossgeo_geo_ldm_sphere.yaml    # CrossGeo Stage-2 配置（含 SphericalControlNet）
│       ├── CVUSA_geo_ldm.yaml              # CVUSA 配置
│       ├── KITTI_geo_ldm.yaml              # KITTI 配置
│       └── VIGOR_geo_ldm.yaml              # VIGOR 配置
├── models/
│   ├── crossgeo_geo_ldm/
│   │   └── crossgeo_txt_control.py         # ★ CrossGEO_Sat2Den_ddpm 主模型（LightningModule）
│   ├── crossgeo_geo_ldm_diffusion/
│   │   ├── crossgeo_latent_diffusion.py    # CrossGeo DDPM（扩散过程 + p_losses）
│   │   ├── crossgeo_ddim.py               # DDIM 采样器（含 SphericalControlNet 分支）
│   │   └── openaimodel.py                 # UNet 去噪器
│   ├── CVUSA_geo_ldm/                      # CVUSA 条件模型
│   ├── CVUSA_geo_ldm_diffusion/            # CVUSA 扩散模型 + DDIM 采样器
│   ├── KITTI_geo_ldm/                      # KITTI 条件模型
│   ├── KITTI_geo_ldm_diffusion/            # KITTI 扩散模型 + DDIM 采样器
│   ├── VIGOR_geo_ldm/                      # VIGOR 条件模型
│   ├── VIGOR_geo_ldm_diffusion/            # VIGOR 扩散模型 + DDIM 采样器
│   ├── spherical_controlnet.py             # ★ SphericalControlNet (SphereDeformableConv2d)
│   ├── geometry/                           # 几何变换 (sat2grd, grd2sat, 投影映射)
│   ├── loss_fun/                           # 损失函数
│   ├── eval/                               # 评估指标
│   └── autoencoder/                        # VAE 自编码器
├── ldm/                                    # 潜在扩散模型基础库
│   ├── modules/                            # CrossAttention, SpatialTransformer, 编码器
│   └── models/diffusion/                   # DDIM/DDPM 基类
├── dataloader/
│   ├── crossgeo_txt.py                     # CrossGeo 数据集
│   ├── CVUSA_txt.py                        # CVUSA 数据集
│   ├── KITTI_wo_loc.py                     # KITTI 数据集
│   └── VIGOR_corr.py                       # VIGOR 数据集
├── utils/                                  # 工具函数 (instantiate_from_config, callback 等)
├── train_all.py                            # ★ 统一训练入口
├── visualization.py                        # 批量推理/可视化入口
├── inference_sphere.py                     # CrossGeo 单图推理入口
└── main.py                                 # 原始训练入口
```

---

## 训练参数速查

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--dataset` | 数据集: `crossgeo` / `CVUSA` / `KITTI` / `VIGOR` | `crossgeo` |
| `--stage` | 训练阶段: `1`（基础）/ `2`（SphericalControlNet） | `1` |
| `--devices` | GPU 设备号 | `"0"` |
| `--max-epochs` | 最大训练轮数 | `50` |
| `--batch-size` | 批次大小 | `2` |
| `--accumulate-grad-batches` | 梯度累积步数 | `4` |
| `--lr` | 学习率（覆盖 YAML 配置中的值） | `None`（使用配置文件值） |
| `--stage1-ckpt` | Stage-2 训练时加载的 Stage-1 checkpoint | `None` |
| `--resume` | 恢复训练的 checkpoint 路径（仅 Stage-1） | `None` |
| `--seed` | 随机种子 | `24` |

---

## 联络

如有问题，欢迎联系：[`couplechein@163.com`](mailto:couplechein@163.com)
