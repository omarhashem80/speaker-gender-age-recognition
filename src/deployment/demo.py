# app.py
import gradio as gr
import joblib
import numpy as np

model = joblib.load("classifier.joblib")
scaler = joblib.load("preprocessor.joblib")

def predict(features):
    X = scaler.transform([features])
    pred = model.predict(X)[0]
    return pred

inputs = [gr.inputs.Textbox(label=f"Feature {i+1}") for i in range(10)]  # Update number accordingly
gr.Interface(fn=predict, inputs=inputs, outputs="label").launch()
