@echo off
echo =========================================
echo DotsOCR CPU Configuration Test
echo =========================================
echo.

echo 1. Testing CPU compatibility...
python quick_cpu_test.py
if errorlevel 1 (
    echo.
    echo ❌ CPU test failed. Please check your installation.
    pause
    exit /b 1
)

echo.
echo 2. Verifying CPU configuration...
python verify_cpu_config.py
if errorlevel 1 (
    echo.
    echo ❌ Configuration verification failed.
    pause
    exit /b 1
)

echo.
echo ✅ SUCCESS: DotsOCR is ready for CPU-only operation!
echo.
echo Next steps:
echo 1. Place your image in the current directory
echo 2. Run: python dots_ocr/parser.py "your_image.jpg" --use_hf --output "./output"
echo.
pause