import sys
sys.path.append('api')
from main import app
from fastapi.testclient import TestClient

client = TestClient(app)
resp = client.post('/predict', json={
    'age': 56,
    'sex': 'male',
    'cp': 'typical_angina',
    'trestbps': 140,
    'chol': 260,
    'fbs': 0,
    'restecg': 'normal',
    'thalach': 150,
    'exang': 'no',
    'oldpeak': 1.2,
    'slope': 'flat'
})
print(resp.status_code)
print(resp.json())
