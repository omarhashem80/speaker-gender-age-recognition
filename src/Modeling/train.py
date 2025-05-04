import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
import os


def training(X_train, y_train):
    """
    Trains an XGBoost classifier using grid search with cross-validation,
    tunes hyperparameters, and saves the best model to a file.

    Parameters:
    ----------
    X_train : np.ndarray or pd.DataFrame
        The training feature data.

    y_train : np.ndarray or pd.Series
        The training target labels.

    Returns:
    -------
    best_model
    """

    print("Training model with cross-validation and tuning...")

    # Define the base XGBoost classifier
    base_model = XGBClassifier(
        objective="multi:softmax", eval_metric="mlogloss", random_state=42, n_jobs=-1
    )

    # Hyperparameter grid for tuning
    param_grid = {
        "n_estimators": [200],  # Number of trees
        "learning_rate": [0.05, 0.1, 0.3],  # Learning rates to try
    }

    # Perform grid search with 5-fold cross-validation
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring="accuracy",
        cv=5,
        n_jobs=-1,
        verbose=1,
    )

    # Fit model to training data
    grid_search.fit(X_train, y_train)

    # Get the best model found during grid search
    best_model = grid_search.best_estimator_
    print("Best parameters found:", grid_search.best_params_)

    # Get the directory of the script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Define model save path relative to the script's directory
    model_path = os.path.join(script_dir, "classifier.joblib")

    # Convert to absolute path
    model_path = os.path.abspath(model_path)

    # Save the best model to a file
    joblib.dump(best_model, model_path)
    print(f"Best model saved as '{model_path}'")

    return best_model


if __name__ == "__main__":
    from preprocessor import preprocessing
    import pandas as pd

    df = pd.read_csv("merged_output_features.csv")
    X_train_resampled, y_train_resampled, X_test_transformed, y_test, pipeline = (
        preprocessing(df, "label")
    )
    model = training(X_train_resampled, y_train_resampled)
