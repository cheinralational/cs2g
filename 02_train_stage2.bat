@echo off
chcp 65001 >nul
set PYTHON="C:\Users\和顾\cs2s_sphere\Scripts\python.exe"

:: ================================================================
::  02 - Stage-2 训练: SphericalControlNet
::  训练内容: 仅 SphericalControlNet (3.88亿参数)
::  基础模型 (VAE+UNet+VIT_224) 全部冻结
:: ================================================================
::  支持数据集: crossgeo / CVUSA / KITTI / VIGOR
::  使用方法: 修改下方 DATASET= 变量和 STAGE1_CKPT= 路径，双击运行
:: ================================================================

:: ★★★ 选择数据集 (必须和 Stage-1 一致) ★★★
set DATASET=crossgeo

set EPOCHS=50
set BATCH=2
set ACCUMULATE=4
set LR=1e-5
set DEVICES=0

:: ★★★ 请修改为 Stage-1 的 checkpoint 实际路径 ★★★
:: (不填则自动查找最新)
set STAGE1_CKPT=

:: 自动查找
if "%STAGE1_CKPT%"=="" (
    echo [INFO] 正在自动查找 Stage-1 checkpoint...
    for /f "delims=" %%i in ('dir /s /b /o-d outputs\stage1\last.ckpt 2^>nul') do (
        set STAGE1_CKPT=%%i
        goto :found
    )
    echo [ERROR] 找不到 Stage-1 checkpoint！
    echo   请先运行 01_train_stage1.bat 完成 Stage-1 训练
    echo   或修改本文件中的 STAGE1_CKPT 变量为实际路径
    pause
    exit /b 1
)
:found

if not exist "%STAGE1_CKPT%" (
    echo [ERROR] Stage-1 checkpoint 不存在: %STAGE1_CKPT%
    pause
    exit /b 1
)

echo ============================================================
echo   02 - Stage-2 训练
echo   Dataset: %DATASET%
echo   SphericalControlNet (球面几何约束控制网络)
echo ============================================================
echo.
echo   Stage-1 ckpt: %STAGE1_CKPT%
echo   Epochs:       %EPOCHS%
echo   Batch:        %BATCH%
echo   Accumulate:   %ACCUMULATE%
echo   LR:           %LR%
echo   Devices:      %DEVICES%
echo.

%PYTHON% cs2g/train_all.py ^
    --dataset %DATASET% ^
    --stage 2 ^
    --stage1-ckpt "%STAGE1_CKPT%" ^
    --max-epochs %EPOCHS% ^
    --batch-size %BATCH% ^
    --accumulate-grad-batches %ACCUMULATE% ^
    --lr %LR% ^
    --devices %DEVICES%

echo.
echo ============================================================
echo   Stage-2 训练完成！
echo   Checkpoint 保存在 outputs/stage2/ 下
echo.
echo   下一步: 双击 03_inference.bat 推理出图
echo ============================================================
pause
