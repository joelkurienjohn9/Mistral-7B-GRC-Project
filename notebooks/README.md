# Notebooks Directory

This directory contains Jupyter notebooks for the AI fine-tuning project.

## Organization

- **exploratory/**: Initial data exploration and analysis notebooks
- **training/**: Model training and fine-tuning notebooks  
- **evaluation/**: Model evaluation and testing notebooks
- **experiments/**: Experimental notebooks for testing new ideas

## Naming Convention

Use the following naming convention for notebooks:
- `01_data_exploration.ipynb`
- `02_model_training.ipynb`
- `03_model_evaluation.ipynb`
- `exp_new_architecture.ipynb` (for experimental notebooks)

## Best Practices

1. **Clear Documentation**: Each notebook should have markdown cells explaining the purpose and methodology
2. **Reproducibility**: Set random seeds and document package versions
3. **Modular Code**: Move reusable functions to the `src/` directory
4. **Version Control**: Use `nbstripout` to clean notebooks before committing
5. **Data Paths**: Use relative paths and environment variables for data locations

## Getting Started

1. Activate your virtual environment
2. Start Jupyter Lab: `jupyter lab`
3. Navigate to this directory to create new notebooks
