from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
import joblib



def train_and_save_model(X_train, y_train, filename="classifier.joblib"):
    print("Training model with cross-validation and tuning...")

    # Define base model
    base_model = XGBClassifier(
        objective='multi:softmax',
        eval_metric='mlogloss',
        random_state=42,
        n_jobs=-1,
    )

    # hyperparameter tuning
    param_grid = {
        'n_estimators': [200],
        'learning_rate': [0.05, 0.1, 0.3]
    }

    #Grid Search with 5-fold CV
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring='accuracy',
        cv=5,
        n_jobs=-1,
        verbose=1
    )
  

    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    best_model = grid_search
    print("Best parameters found:", grid_search.best_params_)
    joblib.dump(best_model, filename)
    print(f"Best model saved as '{filename}'")

    return best_model



if __name__ == "__main__":
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from imblearn.over_sampling import SMOTE
    import os
    # Load the dataframe
    def load_and_concatenate(folder_path):
        dataframes = []
        for filename in os.listdir(folder_path):
            if filename.endswith('.csv'):
                file_path = os.path.join(folder_path, filename)
                df = pd.read_csv(file_path)
                dataframes.append(df)
        return pd.concat(dataframes, ignore_index=True)
    script_directory = os.path.dirname(os.path.abspath(__file__))

    features_folder_path = os.path.join(script_directory, "../../features_200")

    folder_path = os.path.abspath(features_folder_path)
    target = 'label'
    df = load_and_concatenate(folder_path)
    X = df.drop(columns=["label"])
    y = df["label"]

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    X_train = StandardScaler().fit_transform(X_train)
    X_train, y_train = SMOTE(random_state=42).fit_resample(X_train, y_train)

    # Train and save the model
    train_and_save_model(X_train, y_train)