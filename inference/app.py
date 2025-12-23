import pickle
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Student Pass Predictor")

with open("model/model.pkl", "rb") as f:
    model = pickle.load(f)

class Student(BaseModel):
    study_hours: float
    attendance: float        # 0..1
    previous_score: float    # e.g. 0..20

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(student: Student):
    X = np.array([[student.study_hours, student.attendance, student.previous_score]])
    prob_pass = float(model.predict_proba(X)[0][1])
    pred = "PASS" if prob_pass >= 0.5 else "FAIL"
    return {"prediction": pred, "probability": round(prob_pass, 4)}
