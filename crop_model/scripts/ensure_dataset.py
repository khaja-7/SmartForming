"""
Script to download Crop Recommendation dataset if missing and train the ensemble model.
"""
import os
import sys
import urllib.request
import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(BASE_DIR, "crop_model", "data", "combined")
DATA_FILE = os.path.join(DATA_DIR, "final_crop_dataset.csv")

os.makedirs(DATA_DIR, exist_ok=True)

if not os.path.exists(DATA_FILE):
    print("Downloading Crop Recommendation dataset...")
    url = "https://raw.githubusercontent.com/Gladiator07/Harvestify/master/Data-processed/crop_recommendation.csv"
    urllib.request.urlretrieve(url, DATA_FILE)
    
    # Verify and normalize column names if needed
    df = pd.read_csv(DATA_FILE)
    rename_map = {
        "N": "Nitrogen",
        "P": "Phosphorus",
        "K": "Potassium",
        "temperature": "Temperature",
        "humidity": "Humidity",
        "ph": "pH",
        "rainfall": "Rainfall",
        "label": "Crop"
    }
    df.rename(columns=rename_map, inplace=True)
    df.to_csv(DATA_FILE, index=False)
    print(f"Dataset saved to {DATA_FILE} with {len(df)} rows and columns: {list(df.columns)}")
else:
    print(f"Dataset already exists at {DATA_FILE}")
