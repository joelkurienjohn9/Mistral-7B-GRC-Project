# MISTRAL-B-GRC

This repository contains the code and resources for the **MISTRAL-B-GRC Machine Learning project**.

## Project Structure

- **configs/** - Configuration files for experiments and training.  
- **data/** - Dataset storage and preprocessing scripts.  
- **docs/** - Documentation and project-related notes.  
- **logs/** - Log files generated during training or evaluation.  
- **models/** - Trained models and checkpoints.  
- **notebooks/** - Jupyter notebooks for experiments, model quantization, and layer pruning.  
- **scripts/** - Helper scripts for environment setup and dependency management.  
- **src/** - Source code for training, evaluation, and utility functions.  
- **venv/** - Python virtual environment (if created).  

## Environment Setup

Before running the project, you need to set up the environment and install all required dependencies.

### 1. Setup Environment

**Script:** `setup_windows.ps1`  

**Description:**  
This script will:  
- Create a Python virtual environment (if not already created).  
- Install all required Python packages and dependencies for the project.  
- Check for required system tools (like Git or CUDA) and provide guidance if missing.  

### 2. Verify Environment

**Script:** `verify_setup.ps1`  

**Description:**  
This script will:  
- Confirm that Python, packages, and GPU support (if required) are correctly installed.  
- Run basic checks to ensure the environment is ready for training or evaluation.  
- Report any missing components or configuration issues.  

### 3. Clean Up Environment

**Script:** `cleanup_environment.ps1`  

**Description:**  
This script will:  
- Remove the Python virtual environment and all installed packages.  
- Delete temporary files or cache created during setup.  
- Reset the environment so you can perform a fresh setup.  

## Usage

Once the environment is set up and verified, you can:  
- Run training or evaluation using scripts in the `src/` folder.  
- Experiment with Jupyter notebooks in the `notebooks/` folder.  
- Check logs in the `logs/` folder and manage models in the `models/` folder.  

## Notes

- Always run the **verify script** after setup to ensure everything is working correctly.  
- Use the **cleanup script** if you want to reset the environment before a fresh setup.  
- Keep your virtual environment isolated to avoid dependency conflicts.  

---

