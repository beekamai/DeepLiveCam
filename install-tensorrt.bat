@echo off
setlocal
echo == Optional: TensorRT backend (2-3x faster models, ~3.4 GB download) ==
if not exist venv\Scripts\python.exe (echo Run install-windows.bat first. & exit /b 1)
call venv\Scripts\activate.bat
pip install -r requirements-tensorrt.txt || exit /b 1
echo.
echo Done. The first start builds an engine per model (16-60 s each, cached in models\trt_cache).
