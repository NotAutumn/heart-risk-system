"""One-click runner for the full project workflow."""

from data_processing import run_data_processing
from model_training import train_models
from shap_analysis import run_shap_analysis

if __name__ == "__main__":
    print("[1/3] Running data processing...")
    run_data_processing()
    print("[2/3] Training models...")
    train_models()
    print("[3/3] Running SHAP analysis...")
    run_shap_analysis()
    print("Workflow completed successfully.")

