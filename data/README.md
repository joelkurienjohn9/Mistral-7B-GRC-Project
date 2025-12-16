# Data Directory

This directory structure is for organizing your datasets and models. **Data files are NOT committed to git** due to their large size.

## Directory Structure

```
data/
├── raw/          # Original, immutable datasets
├── processed/    # Cleaned and preprocessed data
└── models/       # Trained model files and checkpoints
```

## ⚠️ Important: Large Data Files (100GB+)

**DO NOT commit data files to git!** The `.gitignore` is configured to exclude all data files except `.gitkeep` placeholders.

## Best Practices for Large Datasets

### Option 1: Cloud Storage (Recommended)
Store data in cloud services and download as needed:

```bash
# AWS S3
aws s3 sync s3://your-bucket/datasets/ data/raw/

# Google Cloud Storage
gsutil -m rsync -r gs://your-bucket/datasets/ data/raw/

# Azure Blob Storage
az storage blob download-batch --source your-container --destination data/raw/
```

### Option 2: Data Version Control (DVC)
Use DVC to track large files:

```bash
pip install dvc dvc-s3  # or dvc-gs, dvc-azure

# Track data files
dvc add data/raw/large_dataset.csv
git add data/raw/large_dataset.csv.dvc .gitignore
git commit -m "Add dataset tracking"

# Push to remote storage
dvc remote add -d myremote s3://mybucket/dvc-storage
dvc push
```

Team members can pull data:
```bash
dvc pull
```

### Option 3: Shared Network Drive
Use a shared location and symlink:

```bash
# Linux/Mac
ln -s /path/to/shared/data data/raw

# Windows (PowerShell as Admin)
New-Item -ItemType SymbolicLink -Path "data\raw" -Target "\\network\share\data"
```

### Option 4: Download Script
Create a script to download data:

```python
# scripts/download_data.py
import requests
import os

def download_dataset():
    url = "https://your-data-source.com/dataset.zip"
    output_path = "data/raw/dataset.zip"
    
    print("Downloading dataset...")
    # Download logic here
    
if __name__ == "__main__":
    download_dataset()
```

Then document in README:
```bash
python scripts/download_data.py
```

## What Gets Committed?

✅ **Committed to Git:**
- Directory structure (`.gitkeep` files)
- Data processing scripts
- Data documentation
- Small sample/test datasets (< 10MB)

❌ **NOT Committed:**
- Large datasets (100GB+)
- Model checkpoints
- Training logs
- Temporary/cache files

## Sample Data

For testing and CI/CD, keep small sample datasets:

```
data/
├── samples/          # Small test datasets (< 10MB)
│   └── test.csv
```

Add this to git:
```bash
git add data/samples/test.csv
```

## Environment Variables

Configure data paths in `.env`:

```bash
# .env
RAW_DATA_PATH=/mnt/data/ml/raw
PROCESSED_DATA_PATH=/mnt/data/ml/processed
MODELS_PATH=/mnt/models
```

## Recommended Tools

- **DVC** - Data Version Control (like git for data)
- **MinIO** - Self-hosted S3-compatible storage
- **Hugging Face Datasets** - For public ML datasets
- **Kaggle API** - For Kaggle datasets
- **AWS S3 / Google Cloud Storage** - Cloud storage

## Example: Setup with DVC

```bash
# Install DVC
pip install dvc dvc-s3

# Initialize DVC
dvc init

# Add remote storage
dvc remote add -d storage s3://my-ml-bucket/data

# Track large files
dvc add data/raw/large_dataset.parquet
dvc add data/models/trained_model.pth

# Commit tracking files
git add data/raw/.gitignore data/raw/large_dataset.parquet.dvc
git commit -m "Track dataset with DVC"

# Push data to remote storage
dvc push

# Team members can pull data
git pull
dvc pull
```

## Need Help?

If you're unsure about data management for your specific use case, consider:
1. Dataset size and growth rate
2. Team size and collaboration needs
3. Budget for cloud storage
4. Compliance and security requirements

