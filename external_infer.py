# external_infer.py
import subprocess
import time


def external_infer(infer_script_path, data_dir, dataset_path):
    start_time = time.time()

    # Run infer.py as a subprocess
    subprocess.run(["python", infer_script_path, data_dir, dataset_path], check=True)
    end_time = time.time()
    elapsed_time = end_time - start_time

    # Save the time to time.txt
    with open("time.txt", "w") as f:
        f.write(f"{elapsed_time:.4f}\n")

    print(f"Inference completed in {elapsed_time:.4f} seconds.")


if __name__ == "__main__":
    external_infer("infer.py", "data/", "merged_dataset.csv")
