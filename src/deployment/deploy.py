from huggingface_hub import hf_hub_download
import joblib
import pandas as pd

model_path = hf_hub_download(repo_id="OmarHashem80/age_gender_classifier", filename="classifier.joblib")
model = joblib.load(model_path)
preprocessor_path = hf_hub_download(repo_id="OmarHashem80/age_gender_classifier", filename="preprocessor.joblib")
preprocessor = joblib.load(preprocessor_path)

df = pd.read_csv("../data.csv").head()
X = df.drop(columns=["label"])
print(model.predict(preprocessor.transform(X)))
print(df.label)