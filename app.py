from fastapi import FastAPI
import joblib
import os

app = FastAPI()

version = os.getenv("MODEL_VERSION","v1")
model = joblib.load(f"model_{version}.joblib")

@app.get("/")
def home():
    return {"message":"API successful"}

@app.post("/predict")
def predict(f1:float, f2: float, f3:float, f4:float):
    pred = model.predict([[f1, f2, f3, f4]])
    return {"prediction":int(pred[0])}
