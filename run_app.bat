@echo off
REM Start Ollama server
start "" "C:\Users\nipun\AppData\Local\Programs\Ollama\ollama.exe" start

REM Activate your existing torch-gpu virtual environment
call ..\torch-gpu\Scripts\activate

REM Run Streamlit app
streamlit run app.py
