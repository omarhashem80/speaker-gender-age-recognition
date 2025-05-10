import os
import joblib
import pandas as pd
import warnings
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")


def preprocessing(df, target_name, oversample=False):
    """
    Preprocesses the input DataFrame by handling missing values,
    scaling features, applying optional SMOTE oversampling, and
    saving the preprocessing pipeline to a file.

    Parameters
    ----------
    df : pandas.DataFrame
        The input dataset containing features and the target column.
    target_name : str
        The name of the target column in the dataset.
    oversample : bool, optional (default=False)
        Whether to apply SMOTE to balance class distribution in the training set.

    Returns
    -------
    X_train_resampled : np.ndarray
        Resampled and preprocessed training feature data.
    y_train_resampled : np.ndarray
        Corresponding labels for the resampled training data.
    X_test_transformed : np.ndarray
        Preprocessed test feature data.
    y_test : pandas.Series
        True labels for the test set.
    pipeline : sklearn.pipeline.Pipeline
        The fitted preprocessing pipeline (imputer + scaler).
    """
    df.dropna(inplace=True)

    # Separate features and target
    X = df.drop(columns=[target_name])
    y = df[target_name]

    # Train-test split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Define pipeline: imputation + scaling
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # Fit-transform training set, transform test set
    X_train_transformed = pipeline.fit_transform(X_train)
    X_test_transformed = pipeline.transform(X_test)

    # Save the preprocessing pipeline
    script_dir = os.path.dirname(os.path.abspath(__file__))
    preprocessor_path = os.path.abspath(os.path.join(script_dir, "preprocessor.joblib"))
    joblib.dump(pipeline, preprocessor_path)
    print(f"Preprocessor pipeline saved to '{preprocessor_path}'")

    # Apply SMOTE if oversample flag is True
    X_train_resampled, y_train_resampled = X_train_transformed, y_train
    if oversample:
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(
            X_train_resampled, y_train_resampled
        )
        print("SMOTE applied to training data")

    return X_train_resampled, y_train_resampled, X_test_transformed, y_test, pipeline


if __name__ == "__main__":
    def load_and_concatenate(folder_path):
        """
        Load and concatenate all CSV files in a given folder.

        Parameters
        ----------
        folder_path : str
            Path to the folder containing CSV files.

        Returns
        -------
        pandas.DataFrame
            Concatenated DataFrame of all CSVs.
        """
        dataframes = []
        for filename in os.listdir(folder_path):
            if filename.endswith(".csv"):
                file_path = os.path.join(folder_path, filename)
                df = pd.read_csv(file_path)
                dataframes.append(df)
        return pd.concat(dataframes, ignore_index=True)

    # Locate features folder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    folder_path = os.path.abspath(os.path.join(script_dir, "../../features_200"))

    # Load and save all features
    df = load_and_concatenate(folder_path)
    df.to_csv("all_features.csv", index=False)

    # Uncomment and set the target column to use preprocessing
    # target_col = "label"
    # X_train_resampled, y_train_resampled, X_test_transformed, y_test, pipeline = preprocessing(
    #     df, target_name=target_col, oversample=True
    # )
