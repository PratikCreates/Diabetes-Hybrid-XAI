#!/usr/bin/env python3
"""
Diabetes Prediction Ensemble - Usage Example

This script demonstrates how to use the diabetes prediction ensemble model
with custom data or load and analyze the existing results.
"""

import sys
import os
import json
import pandas as pd

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from diabetes_prediction_simplified import DiabetesPredictionEnsemble

def main():
    """Main example function"""
    
    print("🏥 Diabetes Hybrid XAI - Ensemble Prediction System")
    print("=" * 50)
    
    # Example 1: Load and analyze existing results
    print("\n📊 Loading existing results...")
    
    try:
        with open('../outputs/results/detailed_results.json', 'r') as f:
            results = json.load(f)
        
        print("\n🎯 Model Performance Summary:")
        for model_name, metrics in results['individual_models'].items():
            print(f"  {model_name}:")
            print(f"    - Accuracy: {metrics['accuracy']:.4f}")
            print(f"    - AUC-ROC: {metrics['auc_roc']:.4f}")
            print(f"    - F1-Score: {metrics['f1_score']:.4f}")
        
        print(f"\n🥇 Best Model: {results['summary']['best_model']['model']}")
        print(f"   Best Accuracy: {results['summary']['best_model']['accuracy']:.4f}")
        
    except FileNotFoundError:
        print("⚠️  Results not found. Run the main script first:")
        print("   python diabetes_prediction_simplified.py")
    
    # Example 2: Run the complete pipeline (commented out to avoid re-running)
    print("\n🔄 To run the complete pipeline:")
    print("   python src/diabetes_prediction_simplified.py")
    
    # Uncomment the following lines to run the pipeline:
    # predictor = DiabetesPredictionEnsemble()
    # predictor.run_complete_pipeline()
    
    print("\n✅ Analysis complete!")

if __name__ == "__main__":
    main()