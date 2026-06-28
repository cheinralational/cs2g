#!/bin/bash
# ================================================================
#  03 - 推理: 卫星图 → 街景图 生成
#  使用训练好的模型进行单张图片的推理
# ================================================================
#  支持数据集: crossgeo / CVUSA / KITTI / VIGOR
#  使用方法:
#    DATASET=crossgeo bash 03_inference.sh result/stage2/xxx/last.ckpt test.jpg
#  ----------------------------------------------------------------
#  各数据集测试图示例:
#    crossgeo: dataset/crossgeo/data/pair_0/ground_1_satellite.jpg
#    CVUSA:    dataset/CVUSA/bingmap/19/img_0000.jpg
# ================================================================
# 可修改参数:
#   DATASET       数据集名 (默认 crossgeo)
#   DDIM_STEPS    DDIM 采样步数 (默认 50)
#   GUIDANCE      classifier-free guidance 强度 (默认 7.5)
# ================================================================

set -e
cd "$(dirname "$0")"

DATASET=${DATASET:-crossgeo}
DDIM_STEPS=${DDIM_STEPS:-50}
GUIDANCE=${GUIDANCE:-7.5}

# 自动映射 config YAML
case "$DATASET" in
    CVUSA)    CONFIG="cs2g/configs/Boost_Sat2Den/CVUSA_geo_ldm.yaml" ;;
    KITTI)    CONFIG="cs2g/configs/Boost_Sat2Den/KITTI_geo_ldm.yaml" ;;
    VIGOR)    CONFIG="cs2g/configs/Boost_Sat2Den/VIGOR_geo_ldm.yaml" ;;
    crossgeo) CONFIG="cs2g/configs/Boost_Sat2Den/crossgeo_geo_ldm_sphere.yaml" ;;
    *)        echo "[ERROR] 未知数据集: $DATASET"; exit 1 ;;
esac

# 自动查找 Stage-2 checkpoint
if [ -z "$1" ]; then
    CKPT=$(find outputs/stage2 -name "last.ckpt" -type f -printf '%T@ %p\$GUIDANCE$DDIM_STEPS$OUT$LR$ACCUMULATEn' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)
else
    CKPT="$1"
fi

if [ -z "$CKPT" ]; then
    echo "[ERROR] 找不到 Stage-2 checkpoint"
    echo "  请先运行 02_train_stage2.sh 完成 Stage-2 训练"
    echo "  或手动指定: bash 03_inference.sh <ckpt路径> <卫星图路径>"
    exit 1
fi

if [ -z "$2" ]; then
    if [ "$DATASET" = "crossgeo" ]; then
        SAT="dataset/crossgeo/data/pair_0/ground_1_satellite.jpg"
    elif [ "$DATASET" = "CVUSA" ]; then
        SAT="dataset/CVUSA/bingmap/19/img_0000.jpg"
    else
        echo "[ERROR] 请指定卫星图路径: bash 03_inference.sh <ckpt路径> <卫星图路径>"
        exit 1
    fi
    echo "[INFO] 未指定卫星图，使用默认: $SAT"
else
    SAT="$2"
fi

OUT=${3:-outputs/output_${DATASET}.png}

if [ ! -f "$CKPT" ]; then
    echo "[ERROR] Checkpoint 不存在: $CKPT"
    exit 1
fi

if [ ! -f "$SAT" ]; then
    echo "[ERROR] 卫星图不存在: $SAT"
    exit 1
fi

echo "============================================================"
echo "  03 - 推理: 卫星图 → 街景图"
echo "  Dataset: $DATASET"
echo "============================================================"
echo ""
echo "  Config:         $CONFIG"
echo "  Checkpoint:     $CKPT"
echo "  输入卫星图:     $SAT"
echo "  输出图像:       $OUT"
echo "  DDIM 步数:      $DDIM_STEPS"
echo "  Guidance scale: $GUIDANCE"
echo ""

python cs2g/inference_sphere.py \$GUIDANCE$DDIM_STEPS$OUT$LR$ACCUMULATE
    --config "$CONFIG" \$GUIDANCE$DDIM_STEPS$OUT$LR$ACCUMULATE
    --ckpt "$CKPT" \$GUIDANCE$DDIM_STEPS$OUT$LR$ACCUMULATE
    --sat "$SAT" \$GUIDANCE$DDIM_STEPS$OUT$LR$ACCUMULATE
    --output "$OUT" \$GUIDANCE$DDIM_STEPS$OUT$LR$ACCUMULATE
    --steps $DDIM_STEPS \$GUIDANCE$DDIM_STEPS$OUT$LR$ACCUMULATE
    --scale $GUIDANCE

echo ""
echo "============================================================"
echo "  推理完成！输出图像: $OUT"
echo "============================================================"
