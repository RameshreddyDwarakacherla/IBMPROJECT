@echo off
cls
echo ========================================
echo   iMedDetect - FIXED AND READY!
echo ========================================
echo.
echo ✅ TensorFlow errors fixed
echo ✅ Cross-validation integrated
echo ✅ Research analysis tools added
echo.
echo Starting application...
echo.
echo 📍 Navigate to: Research Analysis
echo 🔬 Try: Cross-Validation Analysis
echo.
echo Press Ctrl+C to stop the server
echo ========================================
echo.
cd Multiple-Disease-Prediction-Webapp\Frontend
streamlit run app.py --server.headless false
