"""
SafeFall AI - Le2i Fall Dataset Downloader
Downloads the Le2i Fall Dataset (tuyenldvn/falldataset-imvia) from Kaggle
and extracts it into data/le2i_dataset.
"""

import os
import sys
import zipfile
import time
from kaggle.api.kaggle_api_extended import KaggleApi

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_SLUG = "tuyenldvn/falldataset-imvia"
TARGET_DIR = os.path.join(PROJECT_ROOT, "data", "le2i_dataset")

def download_and_extract():
    os.makedirs(TARGET_DIR, exist_ok=True)
    print(f"Target directory: {TARGET_DIR}")
    
    api = KaggleApi()
    api.authenticate()
    print("Kaggle API authenticated successfully.")
    
    print(f"Downloading dataset '{DATASET_SLUG}'...")
    start_time = time.time()
    
    # Download dataset zip file
    api.dataset_download_files(DATASET_SLUG, path=TARGET_DIR, unzip=True, quiet=False)
    
    elapsed = time.time() - start_time
    print(f"\nDownload and extraction completed in {elapsed:.1f} seconds ({elapsed/60:.2f} minutes).")
    
    # Verify contents
    extracted_items = os.listdir(TARGET_DIR)
    print(f"Items in {TARGET_DIR}: {extracted_items}")

if __name__ == "__main__":
    download_and_extract()
