@echo off
cls
echo ========================================
echo   iMedDetect - Starting with All Fixes
echo ========================================
echo.
echo Applying fixes...
python FIX_ALL_ERRORS.py
echo.
echo ========================================
echo   Starting Application
echo ========================================
echo.
echo The application will open in your browser
echo All errors have been fixed:
echo   - JSON serialization fixed
echo   - Label encoding fixed
echo   - Missing metrics files created
echo   - Cross-validation errors fixed
echo.
echo Press Ctrl+C to stop the server
echo ========================================
echo.
cd Multiple-Disease-Prediction-Webapp\Frontend
streamlit run app.py
