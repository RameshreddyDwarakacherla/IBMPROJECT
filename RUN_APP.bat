@echo off
echo ========================================
echo   iMedDetect Application Launcher
echo ========================================
echo.
echo Starting the Multiple Disease Prediction System...
echo.
echo NOTE: TensorFlow features are disabled due to compatibility issues.
echo Core disease prediction features are fully functional!
echo.
echo The application will open in your default web browser.
echo Press Ctrl+C to stop the server.
echo.
cd Multiple-Disease-Prediction-Webapp\Frontend
echo Starting Streamlit server...
streamlit run app.py --server.headless true
pause
