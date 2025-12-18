@echo off
echo ========================================
echo   Multiple Disease Prediction App
echo ========================================
echo.

echo Starting application...
echo.

cd Multiple-Disease-Prediction-Webapp\Frontend

echo Activating virtual environment...
call ..\..\..\.venv\Scripts\activate.bat

echo.
echo Starting Streamlit...
echo.
echo The app will open in your browser at:
echo http://localhost:8501
echo.
echo Press Ctrl+C to stop the server
echo.

streamlit run app.py

pause
