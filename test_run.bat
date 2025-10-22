@echo off
echo Testing DotsOCR with HuggingFace model...
echo.

REM Change to the correct directory
cd /d "c:\wamp64\www\dotsocr\dots.ocr"

REM Run the test script first
echo Running model test...
python test_hf_model.py
echo.

REM If test passes, you can run with a real image like this:
REM python dots_ocr/parser.py "path/to/your/image.jpg" --use_hf --output "./output"

echo.
echo Test completed. If successful, you can now use:
echo python dots_ocr/parser.py "your_image.jpg" --use_hf --output "./output"
pause