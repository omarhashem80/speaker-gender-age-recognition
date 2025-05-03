# infer.py
from src.DataCleaning.dataCleaning import DataCleaning
from src.preprocess.preprocess import process_all_files
from src.FeatureExtraction.features import process_csv


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

    preprocessor = joblib.load("src/Modeling/preprocessor.joblib")
    #  Transform features
    X_processed = preprocessor.transform(features_df)
    # Load trained model
    # TODO: MODIFY THE NAMES OF FILE
    model = joblib.load("src/Modeling/classifier.joblib")
    # Predict
    predictions = model.predict(X_processed)
    # Save predictions
    results_path = os.path.join(data_dir, "results.txt")
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
    dataset_dir = sys.argv[2]
    infer(data_dir, dataset_dir)
