import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, f1_score, accuracy_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE


# 1. Load and concatenate all CSV files in a folder
def load_and_concatenate(folder_path):
    dataframes = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            df = pd.read_csv(file_path)
            dataframes.append(df)
    return pd.concat(dataframes, ignore_index=True)


# Step 2.5: Apply SMOTE to training data
def apply_smote(X_train, y_train):
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    return X_resampled, y_resampled


# Step 3: Train model
def train_model_xg(X_train, y_train):
    model = XGBClassifier(
        n_estimators=200,
        eval_metric='mlogloss',
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


# Step 4: Evaluate model
def evaluate_model_xg(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))


if __name__ == '__main__':
    folder_path = 'D:/College/Third Year/Second Term/Pattern/Project/speaker-gender-age-recognition/features_200'
    target = 'label'
    data = load_and_concatenate(folder_path)

    print('loaded')

    data.dropna(inplace=True)
    X = data.drop(columns=[target])
    y = data[target]


    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)


    X_train_resampled, y_train_resampled = apply_smote(X_train_scaled, y_train)
    print('smoted')

    model = train_model_xg(X_train_resampled, y_train_resampled)
    evaluate_model_xg(model, X_train_resampled, y_train_resampled)
    evaluate_model_xg(model, X_test_scaled, y_test)
