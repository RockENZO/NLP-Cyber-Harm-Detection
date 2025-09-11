"""
Setup and Installation Script for NLP Fraud Detection Baseline
============================================================

This script helps set up the environment and run the baseline models.
"""

import subprocess
import sys
import os

def install_requirements():
    """
    Install required packages from requirements.txt
    """
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
        print("✓ All packages installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error installing packages: {e}")
        return False

def check_gpu_availability():
    """
    Check if GPU is available for PyTorch
    """
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA version: {torch.version.cuda}")
            return True
        else:
            print("⚠ GPU not available. Using CPU (training will be slower)")
            return False
    except ImportError:
        print("⚠ PyTorch not installed yet")
        return False

def download_nltk_data():
    """
    Download required NLTK data
    """
    try:
        import nltk
        print("Downloading NLTK data...")
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        print("✓ NLTK data downloaded successfully!")
        return True
    except ImportError:
        print("⚠ NLTK not installed yet")
        return False

def run_basic_baseline():
    """
    Run the basic baseline models (traditional ML)
    """
    print("\n" + "="*50)
    print("Running Traditional ML Baseline Models")
    print("="*50)
    
    try:
        exec(open('baseline_fraud_detection.py').read())
        return True
    except Exception as e:
        print(f"✗ Error running baseline models: {e}")
        return False

def run_bert_baseline():
    """
    Run the BERT baseline model
    """
    print("\n" + "="*50)
    print("Running BERT Baseline Model")
    print("="*50)
    print("Note: This requires significant computational resources")
    
    response = input("Do you want to run BERT training? (y/n): ").lower()
    if response == 'y':
        try:
            exec(open('bert_fraud_detection.py').read())
            return True
        except Exception as e:
            print(f"✗ Error running BERT model: {e}")
            return False
    else:
        print("Skipping BERT training")
        return True

def main():
    """
    Main setup and execution function
    """
    print("NLP Fraud Detection Baseline Setup")
    print("="*40)
    
    # Check current directory
    current_dir = os.getcwd()
    print(f"Current directory: {current_dir}")
    
    # Install requirements
    print("\n1. Installing Dependencies...")
    if not install_requirements():
        print("Setup failed. Please install dependencies manually.")
        return
    
    # Check GPU
    print("\n2. Checking System Configuration...")
    gpu_available = check_gpu_availability()
    
    # Download NLTK data
    print("\n3. Setting up NLTK...")
    download_nltk_data()
    
    # Run models
    print("\n4. Running Baseline Models...")
    
    # Traditional ML models
    if run_basic_baseline():
        print("✓ Traditional ML baseline completed successfully!")
    
    # BERT model (optional)
    if run_bert_baseline():
        print("✓ BERT baseline completed successfully!")
    
    print("\n" + "="*50)
    print("Setup Complete!")
    print("="*50)
    print("\nNext Steps:")
    print("1. Review the model results and visualizations")
    print("2. Replace sample data with real fraud datasets")
    print("3. Experiment with hyperparameters")
    print("4. Consider deploying as a web application")
    
    if not gpu_available:
        print("\nNote: For faster BERT training, consider using:")
        print("- Google Colab (free GPU access)")
        print("- AWS/GCP cloud instances")
        print("- Local GPU setup")

if __name__ == "__main__":
    main()