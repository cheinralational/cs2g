@echo off
chcp 65001 >nul
set PYTHON="C:\Users\和顾\cs2s_sphere\Scripts\python.exe"

:: ================================================================
::  03 - 推理: 卫星图 → 街景图 生成
::  使用训练好的模型进行单张图片的推理
:: ================================================================
::  支持数据集: crossgeo / CVUSA / KITTI / VIGOR
::  使用方法: 修改下方 DATASET=、CKPT=、SAT=，然后双击运行
:: ================================================================

:: ★★★ 选择数据集 ★★★
set DATASET=crossgeo

:: ★★★ 请修改为 Stage-2 checkpoint 的实际路径 ★★★
:: (不填则自动查找最新)
set CKPT=

:: ★★★ 请修改为输入的卫星图路径 ★★★
:: 各数据集测试图示例:
::   crossgeo: dataset/crossgeo/data/pair_0/ground_1_satellite.jpg
::   CVUSA:    dataset/CVUSA/bingmap/19/img_0000.jpg
set SAT=

set DDIM_STEPS=50
set GUIDANCE=7.5
set OUT=outputs/output_result.png

:: 自动查找 checkpoint
if "%CKPT%"=="" (
    echo [INFO] 正在自动查找 Stage-2 checkpoint...
    for /f "delims=" %%i in ('dir /s /b /o-d outputs\stage2\last.ckpt 2^>nul') do (
        set CKPT=%%i
        goto :ckpt_found
    )
)
:ckpt_found

:: 自动映射 config YAML
if "%DATASET%"=="CVUSA"    set CONFIG=cs2g/configs/Boost_Sat2Den/CVUSA_geo_ldm.yaml
if "%DATASET%"=="KITTI"    set CONFIG=cs2g/configs/Boost_Sat2Den/KITTI_geo_ldm.yaml
if "%DATASET%"=="VIGOR"    set CONFIG=cs2g/configs/Boost_Sat2Den/VIGOR_geo_ldm.yaml
if "%DATASET%"=="crossgeo" set CONFIG=cs2g/configs/Boost_Sat2Den/crossgeo_geo_ldm_sphere.yaml

:: 如果没设置卫星图，尝试默认
if "%SAT%"=="" (
    if "%DATASET%"=="crossgeo" set SAT=dataset/crossgeo/data/pair_0/ground_1_satellite.jpg
    if "%DATASET%"=="CVUSA"    set SAT=dataset/CVUSA/bingmap/19/img_0000.jpg
)

if "%CKPT%"=="" (
    echo [ERROR] 找不到 Stage-2 checkpoint！
    echo   请先运行 02_train_stage2.bat 完成 Stage-2 训练
    pause
    exit /b 1
)

if not exist "%CKPT%" (
    echo [ERROR] Checkpoint 不存在: %CKPT%
    pause
    exit /b 1
)

if not exist "%SAT%" (
    echo [ERROR] 卫星图不存在: %SAT%
    echo   请修改脚本中的 SAT= 变量为实际图片路径
    pause
    exit /b 1
)

echo ============================================================
echo   03 - 推理: 卫星图 → 街景图
echo   Dataset: %DATASET%
echo ============================================================
echo.
echo   Config:   %CONFIG%
echo   Checkpoint:   %CKPT%
echo   输入卫星图:    %SAT%
echo   输出图像:      %OUT%
echo   DDIM 步数:     %DDIM_STEPS%
echo   Guidance scale: %GUIDANCE%
echo.

%PYTHON% cs2g/inference_sphere.py ^
    --config %CONFIG% ^
    --ckpt "%CKPT%" ^
    --sat "%SAT%" ^
    --output "%OUT%" ^
    --steps %DDIM_STEPS% ^
    --scale %GUIDANCE%

echo.
echo ============================================================
echo   推理完成！输出图像: %OUT%
echo ============================================================
pause
