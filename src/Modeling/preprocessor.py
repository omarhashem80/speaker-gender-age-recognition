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


def preprocessing(df, target_name):
    """
    Loads and preprocesses the input DataFrame by handling missing values,
    scaling features, applying SMOTE, and saving the preprocessing pipeline.

    Parameters:
    ----------
    df : pandas.DataFrame
        The input dataset containing features and the target column.

    target_name : str
        The name of the target column in the dataset.


    Returns:
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

    # Ensure the dataset has no missing values
    df.dropna(inplace=True)

    # Separate features and target
    X = df.drop(columns=[target_name])
    y = df[target_name]

    # Split the data into train and test sets with stratified sampling
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Define preprocessing pipeline: impute missing values and scale features
    pipeline = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )

    # Fit and transform training data, transform test data
    X_train_transformed = pipeline.fit_transform(X_train)
    X_test_transformed = pipeline.transform(X_test)

    # Get the directory of the script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Define preprocessor save path relative to the script's directory
    preprocessor_path = os.path.join(script_dir, "preprocessor.joblib")

    # Convert to absolute path
    preprocessor_path = os.path.abspath(preprocessor_path)

    # Save the fitted pipeline to the specified file
    joblib.dump(pipeline, preprocessor_path)
    print(f"Preprocessor pipeline saved to '{preprocessor_path}'")

    # Apply SMOTE to handle class imbalance in training data
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(
        X_train_transformed, y_train
    )
    print("SMOTE applied to training data")

    return X_train_resampled, y_train_resampled, X_test_transformed, y_test, pipeline


if __name__ == "__main__":
    # Function to load and concatenate all CSV files in a folder
    def load_and_concatenate(folder_path):
        dataframes = []
        for filename in os.listdir(folder_path):
            if filename.endswith(".csv"):
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

    # # Specify the path for feature folder
    # features_folder_path = os.path.join(script_dir, "../../features_200")
    # folder_path = os.path.abspath(features_folder_path)

    # Load data
    target_col = "label"
    # df = load_and_concatenate(folder_path)
    df = pd.read_csv("merged_output_features.csv")
    # Call function to preprocess and save the pipeline
    X_train_resampled, y_train_resampled, X_test_transformed, y_test, pipeline = (
        preprocessing(df, target_name=target_col)
    )
