# 🚀 Start the Application

## Quick Start

### Option 1: Double-click the batch file
```
RUN_APP.bat
```

### Option 2: Run commands manually

```powershell
# 1. Go to Frontend directory
cd Multiple-Disease-Prediction-Webapp\Frontend

# 2. Activate virtual environment
..\..\..\.venv\Scripts\activate

# 3. Start Streamlit
streamlit run app.py
```

### Option 3: One command
```powershell
cd Multiple-Disease-Prediction-Webapp\Frontend && streamlit run app.py
```

---

## 🌐 Access the App

Once started, the app will open automatically at:
```
http://localhost:8501
```

Or manually open your browser and go to that URL.

---

## 🎯 Features Available

### Disease Prediction
- 🩺 Diabetes Prediction
- ❤️ Heart Disease Prediction
- 🧠 Parkinson's Disease Prediction
- 🫘 Liver Disease Prediction
- 🦠 Hepatitis Prediction
- 🫁 Chronic Kidney Disease Prediction

### Research Analysis Tools
- 📊 Cross-Validation Analysis
- 🔬 SHAP Explainable AI (Diabetes, Heart, Liver)
- ⚡ Hyperparameter Tuning Documentation
- 📈 Model Comparison

### Advanced ML Models
- XGBoost
- Gradient Boosting
- Extra Trees
- Random Forest

---

## 🛑 Stop the App

Press `Ctrl + C` in the terminal/command prompt

---

## 🐛 Troubleshooting

### "streamlit: command not found"
```powershell
pip install streamlit
```

### "No module named 'sklearn'"
```powershell
pip install -r requirements.txt
```

### Port already in use
```powershell
streamlit run app.py --server.port 8502
```

### Virtual environment not activated
```powershell
.venv\Scripts\activate
```

---

## 📱 App Navigation

1. **Sidebar** - Select disease or analysis tool
2. **Main Panel** - Enter parameters and get predictions
3. **Research Analysis** - Access advanced analysis tools
4. **Model Comparison** - Compare all models

---

**Ready to start? Run `RUN_APP.bat` or use the commands above!** 🚀
