@echo off
setlocal
echo == Deep-Live-Cam live fork: Windows setup ==
where python >nul 2>nul || (echo Python 3.12 not found in PATH. Install it from python.org and tick "Add to PATH". & exit /b 1)
python -c "import sys; assert sys.version_info[:2]==(3,12), sys.version" 2>nul || (echo Python 3.12 is required ^(found another version^). & exit /b 1)
if not exist venv (python -m venv venv || exit /b 1)
call venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install .wheels\insightface-0.7.3-cp312-cp312-win_amd64.whl || exit /b 1
pip install -r requirements.txt || exit /b 1
where ffmpeg >nul 2>nul || echo NOTE: ffmpeg not found in PATH - needed only for video output. See README.
echo.
rem TensorRT 10 needs compute capability 7.5+ (GTX 16xx / RTX 20xx and newer).
set TRT_OK=
for /f "usebackq delims=" %%c in (`nvidia-smi --query-gpu=compute_cap --format=csv^,noheader 2^>nul`) do (
    if not defined TRT_OK for /f "tokens=1 delims=." %%m in ("%%c") do if %%m GEQ 8 set TRT_OK=1
    if not defined TRT_OK if "%%c"=="7.5" set TRT_OK=1
)
if defined TRT_OK (
    echo Your GPU supports TensorRT: models run 2-3x faster ^(~3.4 GB download^).
    choice /C YN /T 30 /D Y /M "Install the TensorRT backend now (auto-yes in 30 s)"
    if not errorlevel 2 call install-tensorrt.bat
)
echo.
echo Done. Start with run-cuda.bat
