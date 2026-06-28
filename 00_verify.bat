@echo off
chcp 65001 >nul
set PYTHON="C:\Users\和顾\cs2s_sphere\Scripts\python.exe"

echo ============================================================
echo   00 - 程序可行性验证
echo   检查数据、模型、checkpoint 是否就绪
echo ============================================================
echo.

echo [1/2] 运行 Stage-1 验证...
%PYTHON% _verify_crossgeo.py
if %errorlevel% neq 0 (
    echo.
    echo [FAIL] Stage-1 验证失败，请检查错误信息
    pause
    exit /b 1
)

echo.
echo [2/2] 运行 Stage-2 验证...
%PYTHON% _verify_stage2.py
if %errorlevel% neq 0 (
    echo.
    echo [FAIL] Stage-2 验证失败，请检查错误信息
    pause
    exit /b 1
)

echo.
echo ============================================================
echo   全部验证通过！可以开始训练。
echo ============================================================
pause
