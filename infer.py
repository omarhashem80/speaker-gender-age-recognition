from src.DataCleaning.dataCleaning import DataCleaning
from src.Preprocessing.preprocess import process_all_files
from src.FeatureExtraction.features import process_csv

from huggingface_hub import hf_hub_download
import re

url = "https://omarhashem80-age-gender-classifier.hf.space/predict"


def natural_sort_key(s):
    parts = re.split(r"(\d+)", s)
    return [int(part) if part.isdigit() else part.lower() for part in parts]


def infer(data_dir):


    # # Load The Dataset from the data_dir
    # pathes_df = DataCleaning(data_dir, dataset_csv_path)
    # Preprocess the data
    pathes_df = process_all_files(
        data_dir,
        f"{os.path.abspath((data_dir))}\\preprocessed",
    )
    #  Load features from file
    pathes_df = pathes_df.sort_values(
        by=["path"], key=lambda x: x.map(natural_sort_key)
    )
    print(pathes_df.head())
    features_df = process_csv(pathes_df, "output_features.csv")
    # features_path = os.path.join(data_dir, "features.csv")  # assuming features.csv

    # X = pd.read_csv(features_path)
    # Load preprocessing pipeline

    model_path = hf_hub_download(
        repo_id="OmarHashem80/age_gender_classifier", filename="classifier.joblib"
    )
    model = joblib.load(model_path)
    preprocessor_path = hf_hub_download(
        repo_id="OmarHashem80/age_gender_classifier", filename="preprocessor.joblib"
    )
    preprocessor = joblib.load(preprocessor_path)
    X_processed = preprocessor.transform(features_df)
    # Predict
    predictions = model.predict(X_processed)
    # Save predictions
    os.makedirs("output", exist_ok=True)

    results_path = os.path.join(
        "output", "results.txt"
    )  # results_path = os.path.join(data_dir, "results.txt")
    with open(results_path, "w") as f:
        for pred in predictions:
            f.write(f"{pred}\n")
    print(f"Inference completed. Predictions saved to {results_path}")


if __name__ == "__main__":
    import sys
    import os
    import joblib
    import pandas as pd
    import numpy as np

    data_dir = sys.argv[1]
    infer(data_dir)
