# infer.py
from src.DataCleaning.dataCleaning import DataCleaning
from src.preprocess.preprocess import process_all_files
from src.FeatureExtraction.features import process_csv

from huggingface_hub import hf_hub_download
import requests

url = "https://omarhashem80-age-gender-classifier.hf.space/predict"


def infer(data_dir, dataset_csv_path):

    # TODO: MODIFIY THIS

    # Load The Dataset from the data_dir
    pathes_df = DataCleaning(data_dir, dataset_csv_path)
    # Preprocess the data
    edited_pathes_df = process_all_files(
        data_dir, f"{os.path.dirname(data_dir)}\\preprocessed", pathes_df
    )
    #  Load features from file
    features_df = process_csv(edited_pathes_df, "output_features.csv")
    # features_path = os.path.join(data_dir, "features.csv")  # assuming features.csv

    # X = pd.read_csv(features_path)
    # Load preprocessing pipeline

    # model_path = hf_hub_download(
    #     repo_id="OmarHashem80/age_gender_classifier", filename="classifier.joblib"
    # )
    # model = joblib.load(model_path)
    # preprocessor_path = hf_hub_download(
    #     repo_id="OmarHashem80/age_gender_classifier", filename="preprocessor.joblib"
    # )
    # preprocessor = joblib.load(preprocessor_path)
    # X_processed = preprocessor.transform(features_df)
    # Predict
    response = requests.post(url, json={"data": features_df.to_dict(orient="records")})
    if response.status_code == 200:
        predictions = response.json()["predictions"]
        # Save predictions
        results_path = os.path.join(data_dir, "results.txt")
        with open(results_path, "w") as f:
            for pred in predictions:
                f.write(f"{pred}\n")

        print(f"Inference completed. Predictions saved to {results_path}")
    else:
        print(f"Error: {response.status_code} - {response.text}")


if __name__ == "__main__":
    import sys
    import os
    import joblib
    import pandas as pd
    import numpy as np

    data_dir = sys.argv[1]
    dataset_dir = sys.argv[2]
    infer(data_dir, dataset_dir)
