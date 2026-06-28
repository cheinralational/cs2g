@echo off
chcp 65001 >nul
set PYTHON="C:\Users\和顾\cs2s_sphere\Scripts\python.exe"

:: ================================================================
::  01 - Stage-1 训练: 基础 CS2S (卫星→街景 扩散模型)
::  训练内容: UNet (去噪网络) + VIT_224 (卫星编码器)
::  VAE 冻结 (使用 SD 预训练权重)
:: ================================================================
::  支持数据集: crossgeo / CVUSA / KITTI / VIGOR
::  使用方法: 修改下方 DATASET= 变量，然后双击运行
::  ----------------------------------------------------------------
::  CVUSA:  卫星(256x256) → 全景(128x512), 美国全国
::  KITTI:  卫星 → 前视相机, KITTI (德国)
::  VIGOR:  卫星 → 全景, 4个美国城市
::  crossgeo:卫星(256x256) → 地面(256x256) + UAV, 含3D参数
:: ================================================================

:: ★★★ 选择数据集: crossgeo / CVUSA / KITTI / VIGOR ★★★
set DATASET=crossgeo

set EPOCHS=50
set BATCH=2
set ACCUMULATE=4
set LR=1e-5
set DEVICES=0

:: 支持命令行参数: 01_train_stage1.bat <epochs> <batch>
if not "%1"=="" set EPOCHS=%1
if not "%2"=="" set BATCH=%2

echo ============================================================
echo   01 - Stage-1 训练
echo   Dataset: %DATASET%
echo   卫星 → 街景  基础 CS2S 扩散模型
echo ============================================================
echo.
echo   参数:
echo     Epochs:     %EPOCHS%
echo     Batch:      %BATCH%
echo     Accumulate: %ACCUMULATE%
echo     LR:         %LR%
echo     Devices:    %DEVICES%
echo.

%PYTHON% cs2g/train_all.py ^
    --dataset %DATASET% ^
    --stage 1 ^
    --max-epochs %EPOCHS% ^
    --batch-size %BATCH% ^
    --accumulate-grad-batches %ACCUMULATE% ^
    --lr %LR% ^
    --devices %DEVICES%

echo.
echo ============================================================
echo   Stage-1 训练完成！
echo   Checkpoint 保存在 outputs/stage1/ 下
echo.
echo   下一步: 双击 02_train_stage2.bat 训练 Stage-2
echo          (确保脚本中 DATASET= 和当前一致)
echo ============================================================
pause
