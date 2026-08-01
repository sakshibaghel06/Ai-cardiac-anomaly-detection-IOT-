import logging
from pathlib import Path
from typing import Dict

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title='Heart Disease Prediction API', version='1.0.0')


class PredictionRequest(BaseModel):
    age: int
    sex: str
    cp: str
    trestbps: int
    chol: int
    fbs: int
    restecg: str
    thalach: int
    exang: str
    oldpeak: float
    slope: str


class PredictionResponse(BaseModel):
    prediction: str
    risk_level: str
    probability: float
    recommendation: str


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / 'models' / 'best_model.joblib'
PREPROCESSOR_PATH = BASE_DIR / 'models' / 'preprocessing_pipeline.joblib'


@app.get('/health')
def health() -> Dict[str, str]:
    return {'status': 'ok'}


@app.post('/predict', response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    try:
        model = joblib.load(MODEL_PATH)
        preprocessor = joblib.load(PREPROCESSOR_PATH)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail='Model artifacts not found. Train the pipeline first.') from exc

    input_df = pd.DataFrame([request.dict()])
    input_df['sex'] = input_df['sex'].str.lower().replace({'male': 'male', 'female': 'female'})
    input_df['cp'] = input_df['cp'].str.lower().replace({'typical angina': 'typical_angina', 'atypical angina': 'atypical_angina', 'non-anginal': 'non_anginal', 'asymptomatic': 'asymptomatic'})
    input_df['restecg'] = input_df['restecg'].str.lower()
    input_df['exang'] = input_df['exang'].str.lower().replace({'yes': 'yes', 'no': 'no'})
    input_df['slope'] = input_df['slope'].str.lower().replace({'upsloping': 'upsloping', 'flat': 'flat', 'downsloping': 'downsloping'})

    transformed = preprocessor.transform(input_df)
    probability = float(model.predict_proba(transformed)[0][1])
    prediction = int(model.predict(transformed)[0])

    if prediction == 1:
        label = 'Heart Disease'
        risk_level = 'High' if probability >= 0.7 else 'Moderate'
        recommendation = 'Immediate consultation recommended.' if probability >= 0.7 else 'Consult a clinician soon.'
    else:
        label = 'No Heart Disease'
        risk_level = 'Low'
        recommendation = 'Continue routine preventive care.'

    return PredictionResponse(
        prediction=label,
        risk_level=risk_level,
        probability=round(probability, 3),
        recommendation=recommendation,
    )
