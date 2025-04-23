import pandas as pd
import os

FOLDER_PATH = "./data/raw"


def clean_audio_folder(folder_path: str = FOLDER_PATH) -> None:
    for file in os.listdir(folder_path):
        if os.path.getsize(os.path.join(folder_path, file)) == 0:
            print(f"Deleting empty file: {file}")
            os.remove(os.path.join(folder_path, file))
    return


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


def move_files_dont_exist_in_csv_file_after_cleaning():
    df = pd.read_csv("./filtered_data.csv")
    existing_files = set(df["path"].values)  # O(1) lookup

    not_found_dir = os.path.join(FOLDER_PATH, "NotFound")
    os.makedirs(not_found_dir, exist_ok=True)

    for file in os.listdir(FOLDER_PATH):
        full_path = os.path.join(FOLDER_PATH, file)
        if os.path.isfile(full_path) and file not in existing_files:
            print(f"Moving: {file}")
            os.rename(full_path, os.path.join(not_found_dir, file))


if __name__ == "__main__":
    if os.path.exists("./filtered_data.csv"):
        print("Filtered data already exists. Skipping filtering step.")
        move_files_dont_exist_in_csv_file_after_cleaning()
    else:
        clean_audio_folder()
        clean_df = clean_data("./filtered_data_labeled.tsv")
        print(clean_df.isnull().any(axis=1).sum())
