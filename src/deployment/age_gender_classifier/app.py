from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI()

# Load the model and preprocessor
model = joblib.load("classifier.joblib")
preprocessor = joblib.load("preprocessor.joblib")


class InputData(BaseModel):
    data: list  # List of dicts (records)

@app.post("/predict")
def predict(input_data: InputData):
    df = pd.DataFrame(input_data.data)
    X = df.drop(columns=["label"], errors="ignore")
    preds = model.predict(preprocessor.transform(X))
    return {"predictions": preds.tolist()}
