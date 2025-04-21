import pandas as pd
import os

FOLDER_PATH = "./data/raw"


def clean_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path, sep="\t")
    # Remove Rows Containing Audio Files That Don't Exist
    df = df.sort_values(by=["path"])
    df = df[df["path"].apply(lambda x: os.path.exists(os.path.join(FOLDER_PATH, x)))]
    # Remove unnecessary columns
    removed_columns = ["sentence", "up_votes", "down_votes", "accent"]
    df.drop(columns=removed_columns, inplace=True)
    df.to_csv("./filtered_data.csv", index=False, sep=",")
    return df


if __name__ == "__main__":
    if os.path.exists("./filtered_data.csv"):
        print("Filtered data already exists. Skipping filtering step.")
    else:
        clean_df = clean_data("./filtered_data_labeled.tsv")
        print(clean_df.isnull().any(axis=1).sum())
