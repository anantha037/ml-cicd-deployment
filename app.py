from fastapi import FastAPI
import joblib

app = FastAPI()

model = joblib.load("model.joblib")

@app.get("/")
def home():
    return {"message":"API successful"}

@app.post("/predict")
def predict(f1:float, f2: float, f3:float, f4:float):
    pred = model.predict([[f1, f2, f3, f4]])
    return {"prediction":int(pred[0])}
