@echo off
chcp 65001 >nul
set PYTHON="C:\Users\和顾\cs2s_sphere\Scripts\python.exe"

:: ================================================================
::  00 - IHA 姿态对齐网络训练 (grd_solver + sat_solver)
::
::  训练内容: 训练用于 IHA 姿态对齐的两个小型比对网络
::  前置条件: 需要先完成 Stage-1 训练 (01_train_stage1.bat)
::
::  输入: Stage-1 checkpoint + 对应数据集
::  输出: result/localization_corr/<数据集>/grd_solver.pth
::                            /sat_solver.pth
:: ================================================================
::  使用方法: 修改下方 DATASET= 和 STAGE1_CKPT=，双击运行
:: ================================================================

:: ★★★ 选择数据集 ★★★
set DATASET=CVUSA

:: ★★★ 请修改为 Stage-1 checkpoint 的实际路径 ★★★
:: (不填则自动查找)
set STAGE1_CKPT=

set EPOCHS=500
set LR=1e-5
set DEVICES=0

:: 数据集配置映射
if "%DATASET%"=="CVUSA" set BASE_CONFIG=cs2g/configs/Boost_Sat2Den/train/CVUSA_geo_ldm.yaml
if "%DATASET%"=="KITTI" set BASE_CONFIG=cs2g/configs/Boost_Sat2Den/train/KITTI_geo_ldm.yaml
if "%DATASET%"=="VIGOR" set BASE_CONFIG=cs2g/configs/Boost_Sat2Den/train/VIGOR_geo_ldm.yaml

if "%BASE_CONFIG%"=="" (
    echo [ERROR] 不支持的数据集: %DATASET%
    echo   支持: CVUSA / KITTI / VIGOR
    echo   (crossgeo 暂无 IHA 支持)
    pause
    exit /b 1
)

:: 自动查找 Stage-1 checkpoint
if "%STAGE1_CKPT%"=="" (
    echo [INFO] 正在自动查找 Stage-1 checkpoint...
    for /f "delims=" %%i in ('dir /s /b /o-d outputs\stage1\last.ckpt 2^>nul') do (
        set STAGE1_CKPT=%%i
        goto :ckpt_found
    )
    echo [ERROR] 找不到 Stage-1 checkpoint！
    echo   请先运行 01_train_stage1.bat 完成 Stage-1 训练
    pause
    exit /b 1
)
:ckpt_found

if not exist "%STAGE1_CKPT%" (
    echo [ERROR] Stage-1 checkpoint 不存在: %STAGE1_CKPT%
    pause
    exit /b 1
)

set OUTPUT_DIR=result\localization_corr\%DATASET%
set GRD_OUT=%OUTPUT_DIR%\grd_solver.pth
set SAT_OUT=%OUTPUT_DIR%\sat_solver.pth

:: 检查是否已经有训练好的权重
if exist "%GRD_OUT%" if exist "%SAT_OUT%" (
    echo [INFO] 已有训练好的比对网络权重:
    echo   grd_solver: %GRD_OUT%
    echo   sat_solver: %SAT_OUT%
    echo.
    set /p REDO="重新训练吗? (y/n): "
    if /i not "%REDO%"=="y" (
        echo   跳过训练，使用已有权重
        goto :done
    )
)

echo ============================================================
echo   00 - IHA 姿态对齐网络训练
echo   Dataset: %DATASET%
echo   grd_solver + sat_solver
echo ============================================================
echo.
echo   Stage-1 ckpt: %STAGE1_CKPT%
echo   Config:       %BASE_CONFIG%
echo   Epochs:       %EPOCHS%
echo   LR:           %LR%
echo   Output dir:   %OUTPUT_DIR%
echo.

mkdir "%OUTPUT_DIR%" 2>nul

%PYTHON% cs2g/train_corr_sig.py ^
    --base %BASE_CONFIG% ^
    --test "%STAGE1_CKPT%" ^
    --logdir "%OUTPUT_DIR%"

:: 将最终的 grd_solver 和 sat_solver 重命名为固定名称
:: train_corr_sig.py 每5个epoch保存一次，命名为 *_epoch{N}.pth
:: 取最后一个 epoch 的作为最终权重
echo.
echo [INFO] 正在整理最终权重...
for /f "delims=" %%i in ('dir /s /b /o-n "%DATASET%\grd_solver_epoch*.pth" 2^>nul') do (
    copy /y "%%i" "%GRD_OUT%" >nul
    goto :grd_done
)
:grd_done
for /f "delims=" %%i in ('dir /s /b /o-n "%DATASET%\sat_solver_epoch*.pth" 2^>nul') do (
    copy /y "%%i" "%SAT_OUT%" >nul
    goto :sat_done
)
:sat_done

if not exist "%GRD_OUT%" (
    echo [WARN] grd_solver 权重未生成，请检查训练日志
) else (
    echo    grd_solver: %GRD_OUT%
)
if not exist "%SAT_OUT%" (
    echo [WARN] sat_solver 权重未生成，请检查训练日志
) else (
    echo    sat_solver: %SAT_OUT%
)

:done
echo.
echo ============================================================
echo   比对网络训练完成！
echo.
echo   下一步: 使用 03_inference.bat 并开启 --iha
echo          (脚本中设置 IHA=1)
echo ============================================================
pause
