@echo off
cd /d "%~dp0"
python slideshow_driver_preview.py --subjects p021,p024 --max-per-class 15 --hold-seconds 1 --title-card --section-title
pause
