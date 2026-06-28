#!/bin/bash
# ================================================================
#  01 - Stage-1 训练: 基础 CS2S (卫星→街景 扩散模型)
#  训练内容: UNet (去噪网络) + VIT_224 (卫星编码器)
#  VAE 冻结 (使用 SD 预训练权重)
# ================================================================
#  支持数据集: crossgeo / CVUSA / KITTI / VIGOR
#  使用方法: DATASET=crossgeo bash 01_train_stage1.sh
#  ----------------------------------------------------------------
#  CVUSA:  卫星(256x256) → 全景(128x512), 美国全国
#  KITTI:  卫星 → 前视相机, KITTI (德国)
#  VIGOR:  卫星 → 全景, 4个美国城市
#  crossgeo:卫星(256x256) → 地面(256x256) + UAV, 含3D参数
# ================================================================
# 可修改参数:
#   DATASET       数据集名 (默认 crossgeo)
#   EPOCHS        训练轮数 (默认 50)
#   BATCH         每GPU batch size (默认 2)
#   ACCUMULATE    梯度累积步数 (默认 4)
#   LR            学习率 (默认 1e-5)
#   DEVICES       GPU编号 (默认 0)
# ================================================================

set -e
cd "$(dirname "$0")"

DATASET=${DATASET:-crossgeo}
EPOCHS=${EPOCHS:-50}
BATCH=${BATCH:-2}
ACCUMULATE=${ACCUMULATE:-4}
LR=${LR:-1e-5}
DEVICES=${DEVICES:-0}

echo "============================================================"
echo "  01 - Stage-1 训练"
echo "  Dataset: $DATASET"
echo "  卫星 → 街景  基础 CS2S 扩散模型"
echo "============================================================"
echo ""
echo "  参数:"
echo "    Epochs:    $EPOCHS"
echo "    Batch:     $BATCH"
echo "    Accumulate: $ACCUMULATE"
echo "    LR:        $LR"
echo "    Devices:   $DEVICES"
echo ""

python cs2g/train_all.py \$GUIDANCE$DDIM_STEPS$OUT$LR$ACCUMULATE
    --dataset "$DATASET" \$GUIDANCE$DDIM_STEPS$OUT$LR$ACCUMULATE
    --stage 1 \$GUIDANCE$DDIM_STEPS$OUT$LR$ACCUMULATE
    --max-epochs $EPOCHS \$GUIDANCE$DDIM_STEPS$OUT$LR$ACCUMULATE
    --batch-size $BATCH \$GUIDANCE$DDIM_STEPS$OUT$LR$ACCUMULATE
    --accumulate-grad-batches $ACCUMULATE \$GUIDANCE$DDIM_STEPS$OUT$LR$ACCUMULATE
    --lr $LR \$GUIDANCE$DDIM_STEPS$OUT$LR$ACCUMULATE
    --devices $DEVICES

echo ""
echo "============================================================"
echo "  Stage-1 训练完成！"
echo "  Checkpoint 保存在 outputs/stage1/ 下"
echo ""
echo "  下一步: bash 02_train_stage2.sh"
echo "         (确保 DATASET= 和当前一致)"
echo "============================================================"
