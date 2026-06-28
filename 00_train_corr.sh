#!/bin/bash
# ================================================================
#  00 - IHA 姿态对齐网络训练 (grd_solver + sat_solver)
#
#  训练内容: 训练用于 IHA 姿态对齐的两个小型比对网络
#  前置条件: 需要先完成 Stage-1 训练 (01_train_stage1.sh)
#
#  输入: Stage-1 checkpoint + 对应数据集
#  输出: result/localization_corr/<数据集>/grd_solver.pth
#                               /sat_solver.pth
# ================================================================
#  使用方法:
#    DATASET=CVUSA bash 00_train_corr.sh
#    DATASET=CVUSA bash 00_train_corr.sh <stage1_ckpt路径>
# ================================================================
# 可修改参数:
#   DATASET       数据集名 (默认 CVUSA)
#   EPOCHS        训练轮数 (默认 500)
#   LR            学习率 (默认 1e-5)
#   DEVICES       GPU编号 (默认 0)
# ================================================================

set -e
cd "$(dirname "$0")"

DATASET=${DATASET:-CVUSA}
EPOCHS=${EPOCHS:-500}
LR=${LR:-1e-5}
DEVICES=${DEVICES:-0}

case "$DATASET" in
    CVUSA) BASE_CONFIG="cs2g/configs/Boost_Sat2Den/train/CVUSA_geo_ldm.yaml" ;;
    KITTI) BASE_CONFIG="cs2g/configs/Boost_Sat2Den/train/KITTI_geo_ldm.yaml" ;;
    VIGOR) BASE_CONFIG="cs2g/configs/Boost_Sat2Den/train/VIGOR_geo_ldm.yaml" ;;
    *)     echo "[ERROR] 不支持的数据集: $DATASET"; echo "  支持: CVUSA / KITTI / VIGOR"; exit 1 ;;
esac

if [ -z "$1" ]; then
    STAGE1_CKPT=$(find outputs/stage1 -name "last.ckpt" -type f -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)
else
    STAGE1_CKPT="$1"
fi

if [ -z "$STAGE1_CKPT" ]; then
    echo "[ERROR] 找不到 Stage-1 checkpoint"
    echo "  请先运行 01_train_stage1.sh 完成 Stage-1 训练"
    exit 1
fi

if [ ! -f "$STAGE1_CKPT" ]; then
    echo "[ERROR] Stage-1 checkpoint 不存在: $STAGE1_CKPT"
    exit 1
fi

OUTPUT_DIR="result/localization_corr/${DATASET}"
GRD_OUT="${OUTPUT_DIR}/grd_solver.pth"
SAT_OUT="${OUTPUT_DIR}/sat_solver.pth"

if [ -f "$GRD_OUT" ] && [ -f "$SAT_OUT" ]; then
    echo "[INFO] 已有训练好的比对网络权重:"
    echo "  grd_solver: $GRD_OUT"
    echo "  sat_solver: $SAT_OUT"
    echo ""
    read -p "重新训练吗? (y/n): " REDO
    if [ "$REDO" != "y" ]; then
        echo "  跳过训练，使用已有权重"
        echo ""
        echo "============================================================"
        echo "  权重就绪！"
        echo "============================================================"
        exit 0
    fi
fi

echo "============================================================"
echo "  00 - IHA 姿态对齐网络训练"
echo "  Dataset: $DATASET"
echo "  grd_solver + sat_solver"
echo "============================================================"
echo ""
echo "  Stage-1 ckpt: $STAGE1_CKPT"
echo "  Config:       $BASE_CONFIG"
echo "  Epochs:       $EPOCHS"
echo "  LR:           $LR"
echo "  Output dir:   $OUTPUT_DIR"
echo ""

export CUDA_VISIBLE_DEVICES=$DEVICES
mkdir -p "$OUTPUT_DIR"

python cs2g/train_corr_sig.py \
    --base "$BASE_CONFIG" \
    --test "$STAGE1_CKPT" \
    --logdir "$OUTPUT_DIR"

# Copy the latest checkpoint to a fixed name
echo ""
echo "[INFO] 正在整理最终权重..."
LATEST_GRD=$(find "${DATASET}" -name "grd_solver_epoch*.pth" -type f -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)
LATEST_SAT=$(find "${DATASET}" -name "sat_solver_epoch*.pth" -type f -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)

if [ -n "$LATEST_GRD" ]; then
    cp "$LATEST_GRD" "$GRD_OUT"
    echo "  grd_solver: $GRD_OUT"
else
    echo "  [WARN] grd_solver 权重未生成，请检查训练日志"
fi

if [ -n "$LATEST_SAT" ]; then
    cp "$LATEST_SAT" "$SAT_OUT"
    echo "  sat_solver: $SAT_OUT"
else
    echo "  [WARN] sat_solver 权重未生成，请检查训练日志"
fi

echo ""
echo "============================================================"
echo "  比对网络训练完成！"
echo ""
echo "  下一步: bash 03_inference.sh --iha"
echo "============================================================"
