from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

dt = joblib.load("decision_tree_model.pkl")
rf = joblib.load("random_forest_model.pkl")
scaler = joblib.load("scaler.pkl")

@app.get("/")
def home():
    return {"message": "Backend is running"}

class InputData(BaseModel):
    Age: int
    Daily_Usage_Hours: float
    Phone_Checks_Per_Day: int
    Time_on_Social_Media: float
    Time_on_Gaming: float
    Sleep_Hours: float
    Exercise_Hours: float

features = [
    "Age",
    "Daily_Usage_Hours",
    "Phone_Checks_Per_Day",
    "Time_on_Social_Media",
    "Time_on_Gaming",
    "Sleep_Hours",
    "Exercise_Hours",
]

@app.post("/predict")
def predict(data: InputData):
    arr = np.array([[getattr(data, f) for f in features]])
    arr_scaled = scaler.transform(arr)

    prediction = rf.predict(arr_scaled)[0]   # <-- FIX
    proba = rf.predict_proba(arr_scaled)[0]

    # jika label model adalah string, mapping by index juga aman
    classes = rf.classes_

    result = {
        "prediction": prediction,
        "probabilities": {
            str(classes[i]): float(proba[i]) for i in range(len(classes))
        }
    }
    return result
