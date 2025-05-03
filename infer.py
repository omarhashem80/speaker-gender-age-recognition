# infer.py

def infer(data_dir):

    #TODO: MODIFIY THIS
    #  Load features from file
    features_path = os.path.join(data_dir, 'features.csv')  # assuming features.csv
    X = pd.read_csv(features_path)
    # Load preprocessing pipeline
    preprocessor = joblib.load('src/Modeling/preprocessor.joblib')
    #  Transform features
    X_processed = preprocessor.transform(X)
    # Load trained model
    #TODO: MODIFY THE NAMES OF FILE
    model = joblib.load('src/Modeling/.classifier.joblib')
    # Predict
    predictions = model.predict(X_processed)
    # Save predictions
    results_path = os.path.join(data_dir, 'results.txt')
    with open(results_path, 'w') as f:
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
