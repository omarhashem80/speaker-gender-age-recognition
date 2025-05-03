import joblib
import optuna
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os


def training(X_train, y_train):
    """
    Trains an XGBoost classifier using Optuna hyperparameter optimization,
    and saves the best model to a file.

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

    print("Training model with Optuna hyperparameter tuning...")

    # Split a validation set from training data
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # Define objective function for Optuna
    def objective(trial):
        params = {
            'objective': 'multi:softmax',
            'num_class': len(set(y_train)),
            'eval_metric': 'mlogloss',
            'learning_rate': trial.suggest_float("learning_rate", 0.01, 0.3),
            'n_estimators': trial.suggest_int("n_estimators", 100, 300),
            'max_depth': trial.suggest_int("max_depth", 3, 10),
            'subsample': trial.suggest_float("subsample", 0.5, 1.0),
            'colsample_bytree': trial.suggest_float("colsample_bytree", 0.5, 1.0),
            'lambda': trial.suggest_float("lambda", 1e-8, 1.0, log=True),
            'alpha': trial.suggest_float("alpha", 1e-8, 1.0, log=True),
            'random_state': 42,
            'n_jobs': -1,
        }

        model = XGBClassifier(**params)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], early_stopping_rounds=10, verbose=False)
        preds = model.predict(X_val)
        return 1.0 - accuracy_score(y_val, preds)  # Minimize error

    # Run Optuna optimization
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=30)
    best_params = study.best_params
    print("Best parameters found:", best_params)

    # Train best model on full training data
    best_params.update({
        'objective': 'multi:softmax',
        'num_class': len(set(y_train)),
        'eval_metric': 'mlogloss',
        'random_state': 42,
        'n_jobs': -1,
    })

    best_model = XGBClassifier(**best_params)
    best_model.fit(X_train, y_train)

    # Save model
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "classifier.joblib")
    model_path = os.path.abspath(model_path)
    joblib.dump(best_model, model_path)
    print(f"✅ Best model saved as '{model_path}'")

    return best_model
