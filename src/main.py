from Modeling.preprocessor import preprocessing
from Modeling.train import training
import os
import pandas as pd

def load_and_concatenate(folder_path):
    dataframes = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            df = pd.read_csv(file_path)
            dataframes.append(df)
    return pd.concat(dataframes, ignore_index=True)

script_dir = os.path.dirname(os.path.abspath(__file__))
features_folder_path = os.path.join(script_dir, "../features_200")
folder_path = os.path.abspath(features_folder_path)

# Load data
target_col = 'label'
df = load_and_concatenate(folder_path)
df.to_csv('data.csv', index=False)
