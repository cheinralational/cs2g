#!/bin/bash
# ================================================================
#  02 - Stage-2 训练: SphericalControlNet
#  训练内容: 仅 SphericalControlNet (3.88亿参数)
#  基础模型 (VAE+UNet+VIT_224) 全部冻结
# ================================================================
#  支持数据集: crossgeo / CVUSA / KITTI / VIGOR
#  使用方法: DATASET=crossgeo bash 02_train_stage2.sh outputs/stage1/xxx/checkpoints/last.ckpt
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

# 自动查找 Stage-1 checkpoint
if [ -z "$1" ]; then
    STAGE1_CKPT=$(find outputs/stage1 -name "last.ckpt" -type f -printf '%T@ %p\$GUIDANCE$DDIM_STEPS$OUT$LR$ACCUMULATEn' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)
    if [ -z "$STAGE1_CKPT" ]; then
        echo "[ERROR] 找不到 Stage-1 checkpoint"
        echo "  请先运行 01_train_stage1.sh 完成 Stage-1 训练"
        echo "  或手动指定: bash 02_train_stage2.sh <stage1_ckpt路径>"
        exit 1
    fi
else
    STAGE1_CKPT="$1"
fi

if [ ! -f "$STAGE1_CKPT" ]; then
    echo "[ERROR] Stage-1 checkpoint 不存在: $STAGE1_CKPT"
    exit 1
fi

echo "============================================================"
echo "  02 - Stage-2 训练"
echo "  Dataset: $DATASET"
echo "  SphericalControlNet (球面几何约束控制网络)"
echo "============================================================"
echo ""
echo "  Stage-1 ckpt: $STAGE1_CKPT"
echo "  Epochs:       $EPOCHS"
echo "  Batch:        $BATCH"
echo "  Accumulate:   $ACCUMULATE"
echo "  LR:           $LR"
echo "  Devices:      $DEVICES"
echo ""

python cs2g/train_all.py \$GUIDANCE$DDIM_STEPS$OUT$LR$ACCUMULATE
    --dataset "$DATASET" \$GUIDANCE$DDIM_STEPS$OUT$LR$ACCUMULATE
    --stage 2 \$GUIDANCE$DDIM_STEPS$OUT$LR$ACCUMULATE
    --stage1-ckpt "$STAGE1_CKPT" \$GUIDANCE$DDIM_STEPS$OUT$LR$ACCUMULATE
    --max-epochs $EPOCHS \$GUIDANCE$DDIM_STEPS$OUT$LR$ACCUMULATE
    --batch-size $BATCH \$GUIDANCE$DDIM_STEPS$OUT$LR$ACCUMULATE
    --accumulate-grad-batches $ACCUMULATE \$GUIDANCE$DDIM_STEPS$OUT$LR$ACCUMULATE
    --lr $LR \$GUIDANCE$DDIM_STEPS$OUT$LR$ACCUMULATE
    --devices $DEVICES

echo ""
echo "============================================================"
echo "  Stage-2 训练完成！"
echo "  Checkpoint 保存在 result/stage2/ 下"
echo ""
echo "  下一步: bash 03_inference.sh"
echo "============================================================"
