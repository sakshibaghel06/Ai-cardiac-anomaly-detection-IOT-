# 🫀 Heart Disease Prediction AI Service

A production-style Machine Learning service for Heart Disease Prediction using multiple classification algorithms.

The project includes:

- Dataset preprocessing
- Exploratory Data Analysis (EDA)
- Model training & comparison
- Hyperparameter tuning
- Model serialization
- FastAPI prediction API

---

# 📁 Project Structure

```
ai_service/
│
├── api/                    # FastAPI application
├── dataset/                # Heart disease datasets
├── models/                 # Generated model artifacts (created after training)
├── notebooks/              # EDA notebooks
├── reports/                # Generated graphs and reports
├── src/                    # Training and preprocessing code
│
├── inspect_datasets.py
├── verify_api.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

# 🐍 Python Version

Use **Python 3.10.x**

Verify:

```bash
python --version
```

Expected:

```
Python 3.10.x
```

---

# ⚙️ Create Virtual Environment

Windows:

```bash
python -m venv .venv
```

Activate:

PowerShell

```powershell
.venv\Scripts\Activate.ps1
```

CMD

```cmd
.venv\Scripts\activate.bat
```

---

# 📦 Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 📊 Verify Dataset

Datasets should be inside:

```
dataset/

heart.csv

heart_cleveland_upload.csv
```

(Optional)

```bash
python inspect_datasets.py
```

---

# 🚀 Train the Model

Run:

```bash
python -m src.train_pipeline
```

After successful training, the following artifacts will be generated:

```
models/

best_model.joblib

preprocessing_pipeline.joblib

model_comparison.csv
```

---

# 📈 EDA

EDA reports and graphs will be generated inside:

```
reports/
```

Optional notebook:

```
notebooks/eda_notebook.ipynb
```

---

# 🌐 Start the FastAPI Server

```bash
uvicorn api.main:app --reload --port 8000
```

API Documentation:

```
http://127.0.0.1:8000/docs
```

Swagger UI will be available automatically.

---

# ✅ Verify the API

Run:

```bash
python verify_api.py
```

Expected output:

```
Prediction: Heart Disease
Risk Level: Moderate
Probability: 0.xx
```

---

# 📌 Features

- Dataset preprocessing
- Feature engineering
- Multiple ML models
- Hyperparameter tuning
- Cross-validation
- Best model selection
- Model serialization
- FastAPI inference service
- Modular project structure

---

# 📚 Technologies Used

- Python 3.10
- Pandas
- NumPy
- Scikit-learn
- CatBoost
- XGBoost
- LightGBM
- FastAPI
- Joblib
- Matplotlib
- Seaborn

---

# 👨‍💻 Workflow

```
Dataset
    │
    ▼
EDA
    │
    ▼
Preprocessing
    │
    ▼
Train Multiple Models
    │
    ▼
Select Best Model
    │
    ▼
Save Model
    │
    ▼
FastAPI
    │
    ▼
Prediction
```

---

# 📄 Notes

- Ensure Python 3.10 is installed.
- Do **not** commit the `.venv/`, `models/`, `reports/`, or `training.log` files.
- If model artifacts are missing, run the training pipeline before starting the API.