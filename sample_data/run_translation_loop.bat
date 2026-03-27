@echo off
echo Starting translation loop...
:loop
python "translate_long_texts_local.py"
echo Batch completed, restarting...
timeout /t 3 /nobreak >nul
goto loop
