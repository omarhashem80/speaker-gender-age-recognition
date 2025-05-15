import subprocess
import time
import os


def external_infer(infer_script_path, data_dir):
    start_time = time.time()

    # Run infer.py as a subprocess
    subprocess.run(["python3", infer_script_path, data_dir], check=True)
    end_time = time.time()
    elapsed_time = end_time - start_time

    # Save the time to time.txt
    os.makedirs("output", exist_ok=True)
    # Write to the file in the 'output' directory
    with open(os.path.join("output", "time.txt"), "w") as f:
        f.write(f"{elapsed_time:.4f}\n")

    print(f"Inference completed in {elapsed_time:.4f} seconds.")


if __name__ == "__main__":
    external_infer("infer.py", "data/")
