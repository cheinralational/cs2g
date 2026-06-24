#!/bin/bash
# 使用方法: bash 00_verify.sh
#          (确保已激活 conda 环境)

cd "$(dirname "$0")"

echo "============================================================"
echo "  00 - 程序可行性验证"
echo "  检查数据、模型、checkpoint 是否就绪"
echo "============================================================"
echo ""

echo "[1/2] 运行 Stage-1 验证..."
python _verify_crossgeo.py
if [ $? -ne 0 ]; then
    echo ""
    echo "[FAIL] Stage-1 验证失败，请检查错误信息"
    exit 1
fi

echo ""
echo "[2/2] 运行 Stage-2 验证..."
python _verify_stage2.py
if [ $? -ne 0 ]; then
    echo ""
    echo "[FAIL] Stage-2 验证失败，请检查错误信息"
    exit 1
fi

echo ""
echo "============================================================"
echo "  全部验证通过！可以开始训练。"
echo "============================================================"
