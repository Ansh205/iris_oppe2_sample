'''from fastapi import FastAPI, Body
from model import load_model
import logging

app = FastAPI()
model = load_model()

logging.basicConfig(level=logging.INFO)

@app.get("/")
def home():
    return {"message": "API running"}

@app.post("/predict")
def predict(data: list = Body(...)):
    preds = model.predict(data)
    logging.info(f"Input: {data}, Prediction: {preds}")
    return {"prediction": preds.tolist()}

'''

from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()

model = joblib.load("model.pkl")

@app.get("/")
def home():
    return {"message": "API running"}

@app.post("/predict")
def predict(data: dict):
    features = np.array([
        data["sepal_length"],
        data["sepal_width"],
        data["petal_length"],
        data["petal_width"]
    ]).reshape(1, -1)

    pred = model.predict(features)[0]

    return {"prediction": int(pred)}