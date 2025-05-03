from huggingface_hub import HfApi, HfFolder, Repository
import os
import os
import stat
import shutil

# def remove_readonly(func, path, excinfo):
#     os.chmod(path, stat.S_IWRITE)
#     func(path)


# Variables
model_name = "speaker-age-gender-xgboost"
local_dir = "D:/College/Third Year/Second Term/Pattern/Project/speaker-gender-age-recognition/src/Modeling"
user = "OmarHashem80" 
repo_id = f"{user}/{model_name}"

# # Clean up if needed
# if os.path.exists(local_dir):
#     shutil.rmtree(local_dir, onerror=remove_readonly)

# Create repo
api = HfApi()
api.create_repo(repo_id=repo_id, exist_ok=True)

# Clone the repo locally
repo = Repository(local_dir=local_dir, clone_from=repo_id)

# Copy your files
shutil.copy("classifier.joblib", os.path.join(local_dir, ".classifier.joblib"))
shutil.copy("preprocessor.joblib", os.path.join(local_dir, "preprocessor.joblib"))
shutil.copy("README.md", os.path.join(local_dir, "README.md"))

# Push to HF
repo.push_to_hub(commit_message="Initial model upload")
