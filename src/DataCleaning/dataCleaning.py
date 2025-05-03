import pandas as pd
import os


FOLDER_PATH = "./data/raw"


def clean_audio_folder(folder_path: str = FOLDER_PATH):
    for file in os.listdir(folder_path):
        if os.path.getsize(os.path.join(folder_path, file)) == 0 and os.path.isfile(
            os.path.join(folder_path, file)
        ):
            print(f"Deleting empty file: {file}")
            os.remove(os.path.join(folder_path, file))


def clean_data(data_dir: str, file_path: str) -> pd.DataFrame:
    if file_path.endswith(".tsv"):
        df = pd.read_csv(file_path, sep="\t")
    else:
        df = pd.read_csv(file_path)
    # Remove Rows Containing Audio Files That Don't Exist
    df = df.sort_values(by=["path"])
    df = df[df["path"].apply(lambda x: os.path.exists(os.path.join(data_dir, x)))]
    df["path"] = df["path"].apply(lambda x: os.path.join(data_dir, x))
    # Remove unnecessary columns
    removed_columns = ["sentence", "up_votes", "down_votes", "accent"]
    df.drop(columns=removed_columns, inplace=True)

    df.to_csv(
        f"{os.path.basename(file_path).split('.')[0]}_cleaned.csv", index=False, sep=","
    )
    return df


def move_files_not_in_csv_file_after_cleaning(data_dir: str, cleaned_file_path: str):
    df = pd.read_csv(cleaned_file_path)
    existing_files = set(df["path"].values)  # O(1) lookup

    not_found_dir = os.path.join(data_dir, "NotFound")
    os.makedirs(not_found_dir, exist_ok=True)

    for file in os.listdir(data_dir):
        full_path = os.path.join(data_dir, file)
        if os.path.isfile(full_path) and file not in existing_files:
            print(f"Moving: {file}")
            os.rename(full_path, os.path.join(not_found_dir, file))


def DataCleaning(data_dir: str, dataset_csv_path: str):
    pathes_df = None
    if os.path.exists(
        f"{os.path.basename(dataset_csv_path).split('.')[0]}_cleaned.csv"
    ):
        pathes_df = pd.read_csv(
            f"{os.path.basename(dataset_csv_path).split('.')[0]}_cleaned.csv"
        )
        print("Filtered data already exists. Skipping filtering step.")
    else:
        clean_audio_folder(data_dir)
        pathes_df = clean_data(data_dir, dataset_csv_path)

    return pathes_df


if __name__ == "__main__":
    DataCleaning("data\\raw", "filtered_data_labeled.tsv")
