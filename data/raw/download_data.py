import os
import shutil
import kagglehub

print("Downloading credit-risk-dataset directly from Kaggle using kagglehub...")
try:
    # 1. Download the latest version of the dataset using kagglehub
    downloaded_path = kagglehub.dataset_download("laotse/credit-risk-dataset")
    print("Path to dataset files in Kagglehub cache:", downloaded_path)
    
    # 2. Locate the CSV file inside the downloaded path
    source_file = os.path.join(downloaded_path, "credit_risk_dataset.csv")
    
    # 3. Define our target raw data directory
    raw_dir = "data/raw"
    os.makedirs(raw_dir, exist_ok=True)
    target_file = os.path.join(raw_dir, "credit_risk_dataset.csv")
    
    # 4. Copy the file into our standard repository folder
    shutil.copy(source_file, target_file)
    print("Successfully copied raw dataset into: %s" % target_file)
except Exception as e:
    print("Error downloading or copying dataset: %s" % e)
