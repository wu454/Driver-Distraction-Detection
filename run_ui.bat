@echo off
cd /d "%~dp0"
pip install -q gradio 2>nul
python driver_distraction_ui.py --device cuda --port 7860
pause
