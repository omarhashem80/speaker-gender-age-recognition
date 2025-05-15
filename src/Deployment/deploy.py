import pandas as pd
import requests

df = pd.read_csv("../data.csv").head()
data = df.to_dict(orient="records")


url = "https://omarhashem80-age-gender-classifier.hf.space/predict"


response = requests.post(url, json={"data": data})

if response.status_code == 200:
    predictions = response.json()["predictions"]
    print("Predictions:", predictions)
    print("True Labels:", df["label"].tolist())
else:
    print("Error:", response.status_code)
    print("Response text:", response.text)
