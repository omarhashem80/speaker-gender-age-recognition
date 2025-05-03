import os
import joblib
import pandas as pd
import warnings
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from imblearn.over_sampling import SMOTE

warnings.filterwarnings('ignore')


# Function to load and preprocess data and save the preprocessor
def preprocessing(df, target_name, preprocessor_path):
    # Ensure no NaN in the final dataset (just in case)
    df.dropna(inplace=True)

    X = df.drop(columns=[target_name])
    y = df[target_name]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Create a pipeline with KNN Imputer, scaler, and SMOTE
    pipeline = Pipeline([
        ('imputer', KNNImputer(n_neighbors=5)),  # KNN Imputer to fill NaNs
        ('scaler', StandardScaler())  # Scaler to standardize data
    ])

    # Fit the pipeline on training data and transform it
    X_train_transformed = pipeline.fit_transform(X_train)
    X_test_transformed = pipeline.transform(X_test)

    # Save the pipeline (imputer + scaler) to the provided path
    joblib.dump(pipeline, preprocessor_path)
    print(f"✅ Preprocessor pipeline saved to '{preprocessor_path}'")

    # Apply SMOTE to training data
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_transformed, y_train)
    print("✅ SMOTE applied to training data")

    return X_train_resampled, y_train_resampled, X_test_transformed, y_test, pipeline


# Example usage
if __name__ == '__main__':
    # Function to load and concatenate all CSV files in a folder
    def load_and_concatenate(folder_path):
        dataframes = []
        for filename in os.listdir(folder_path):
            if filename.endswith('.csv'):
                file_path = os.path.join(folder_path, filename)
                df = pd.read_csv(file_path)
                dataframes.append(df)
        return pd.concat(dataframes, ignore_index=True)

    # Get the directory of the script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Define preprocessor save path relative to the script's directory
    preprocessor_path = os.path.join(script_dir, "preprocessor.joblib")

    # Convert to absolute path
    preprocessor_path = os.path.abspath(preprocessor_path)

    # Specify the path for feature folder
    features_folder_path = os.path.join(script_dir, "../../features_200")
    folder_path = os.path.abspath(features_folder_path)

    # Load data
    target_col = 'label'
    df = load_and_concatenate(folder_path)

    # Call function to preprocess and save the pipeline
    X_train_resampled, y_train_resampled, X_test_transformed, y_test, pipeline = preprocessing(df, target_name=target_col, preprocessor_path=preprocessor_path)
