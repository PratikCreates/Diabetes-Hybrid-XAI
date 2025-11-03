#!/usr/bin/env python3
"""
Simplified Diabetes Prediction Using Hybrid Ensemble Framework
Implementation focuses on core functionality without SHAP for now
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, 
                           roc_curve, precision_recall_curve, f1_score, accuracy_score,
                           brier_score_loss)
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
import warnings
import joblib
import json
from datetime import datetime
import os

warnings.filterwarnings('ignore')

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

class DiabetesPredictionEnsemble:
    """Simplified Hybrid Ensemble Framework for Diabetes Prediction"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.imputer = KNNImputer(n_neighbors=5)
        self.smote = SMOTE(random_state=random_state, k_neighbors=5)
        
        # Base learners
        self.rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=random_state,
            class_weight='balanced'
        )
        
        self.xgb_model = XGBClassifier(
            learning_rate=0.1,
            max_depth=6,
            n_estimators=100,
            random_state=random_state,
            eval_metric='logloss'
        )
        
        self.lgb_model = LGBMClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_leaves=31,
            feature_fraction=0.8,
            random_state=random_state,
            verbose=-1
        )
        
        # Meta-learner
        self.meta_learner = LogisticRegression(random_state=random_state, C=1.0)
        
        # Results storage
        self.results = {}
        self.feature_importance = {}
        
        print("Diabetes Prediction Ensemble Framework Initialized")
    
    def load_and_preprocess_data(self, file_path):
        """Load and preprocess the diabetes dataset"""
        print("\n=== DATA LOADING AND PREPROCESSING ===")
        
        # Load dataset
        df = pd.read_csv(file_path, header=None)
        
        # Add column names for Pima Indians Diabetes Dataset
        df.columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                     'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
        
        print(f"Dataset shape: {df.shape}")
        print(f"Class distribution: {df['Outcome'].value_counts()}")
        
        # Define feature columns
        feature_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                       'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
        
        # Handle zero values that represent missing data
        zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
        for col in zero_cols:
            zero_count = (df[col] == 0).sum()
            print(f"Zero values in {col}: {zero_count}")
        
        # Separate features and target
        X = df[feature_cols].copy()
        y = df['Outcome'].copy()
        
        # Handle missing values (zeros) with domain knowledge
        X.loc[X['Glucose'] == 0, 'Glucose'] = np.nan
        X.loc[X['BloodPressure'] == 0, 'BloodPressure'] = np.nan
        X.loc[X['SkinThickness'] == 0, 'SkinThickness'] = np.nan
        X.loc[X['Insulin'] == 0, 'Insulin'] = np.nan
        X.loc[X['BMI'] == 0, 'BMI'] = np.nan
        
        print(f"Missing values before imputation:\n{X.isnull().sum()}")
        
        # Impute missing values using KNN
        X_imputed = self.imputer.fit_transform(X)
        X_imputed = pd.DataFrame(X_imputed, columns=feature_cols)
        
        print(f"Missing values after imputation: {X_imputed.isnull().sum().sum()}")
        
        # Feature engineering
        X_imputed['BMI_Category'] = pd.cut(X_imputed['BMI'], 
                                          bins=[0, 18.5, 25, 30, 100], 
                                          labels=[0, 1, 2, 3])
        
        X_imputed['Age_Group'] = pd.cut(X_imputed['Age'], 
                                       bins=[0, 30, 50, 100], 
                                       labels=[0, 1, 2])
        
        # Interaction features
        X_imputed['Glucose_BMI_Interaction'] = X_imputed['Glucose'] * X_imputed['BMI']
        
        self.feature_names = X_imputed.columns.tolist()
        print(f"Total features after engineering: {len(self.feature_names)}")
        
        return X_imputed, y
    
    def handle_class_imbalance(self, X, y):
        """Handle class imbalance using SMOTE"""
        print("\n=== CLASS IMBALANCE HANDLING ===")
        
        print(f"Original class distribution:\n{y.value_counts()}")
        print(f"Original class proportions:\n{y.value_counts(normalize=True)}")
        
        X_resampled, y_resampled = self.smote.fit_resample(X, y)
        
        print(f"Resampled class distribution:\n{pd.Series(y_resampled).value_counts()}")
        print(f"Resampled class proportions:\n{pd.Series(y_resampled).value_counts(normalize=True)}")
        
        return X_resampled, y_resampled
    
    def train_base_learners(self, X_train, y_train):
        """Train individual base learners"""
        print("\n=== TRAINING BASE LEARNERS ===")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train Random Forest
        print("Training Random Forest...")
        self.rf_model.fit(X_train_scaled, y_train)
        
        # Train XGBoost
        print("Training XGBoost...")
        self.xgb_model.fit(X_train_scaled, y_train)
        
        # Train LightGBM
        print("Training LightGBM...")
        self.lgb_model.fit(X_train_scaled, y_train)
        
        print("Base learners trained successfully!")
    
    def create_stacking_predictions(self, X_train, y_train, X_val):
        """Create stacking predictions using out-of-fold approach"""
        print("\n=== CREATING STACKING PREDICTIONS ===")
        
        n_samples = len(X_train)
        n_val = len(X_val)
        
        # Initialize prediction arrays
        train_preds = np.zeros((n_samples, 3))  # 3 base learners
        val_preds = np.zeros((n_val, 3))
        
        # Stratified K-Fold for out-of-fold predictions
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
            print(f"Processing fold {fold + 1}/5...")
            
            X_fold_train = X_train.iloc[train_idx]
            X_fold_val = X_train.iloc[val_idx]
            y_fold_train = y_train.iloc[train_idx]
            
            # Scale features for this fold
            scaler_fold = StandardScaler()
            X_fold_train_scaled = scaler_fold.fit_transform(X_fold_train)
            X_fold_val_scaled = scaler_fold.transform(X_fold_val)
            X_val_scaled_fold = self.scaler.transform(X_val)
            
            # Train base learners on fold
            rf_fold = RandomForestClassifier(
                n_estimators=100, max_depth=10, min_samples_split=5,
                random_state=self.random_state, class_weight='balanced'
            )
            rf_fold.fit(X_fold_train_scaled, y_fold_train)
            
            xgb_fold = XGBClassifier(
                learning_rate=0.1, max_depth=6, n_estimators=100,
                random_state=self.random_state, eval_metric='logloss'
            )
            xgb_fold.fit(X_fold_train_scaled, y_fold_train)
            
            lgb_fold = LGBMClassifier(
                n_estimators=100, learning_rate=0.1, max_leaves=31,
                feature_fraction=0.8, random_state=self.random_state, verbose=-1
            )
            lgb_fold.fit(X_fold_train_scaled, y_fold_train)
            
            # Make predictions
            train_preds[val_idx, 0] = rf_fold.predict_proba(X_fold_val_scaled)[:, 1]
            train_preds[val_idx, 1] = xgb_fold.predict_proba(X_fold_val_scaled)[:, 1]
            train_preds[val_idx, 2] = lgb_fold.predict_proba(X_fold_val_scaled)[:, 1]
            
            val_preds[:, 0] += rf_fold.predict_proba(X_val_scaled_fold)[:, 1] / 5
            val_preds[:, 1] += xgb_fold.predict_proba(X_val_scaled_fold)[:, 1] / 5
            val_preds[:, 2] += lgb_fold.predict_proba(X_val_scaled_fold)[:, 1] / 5
        
        return train_preds, val_preds
    
    def train_meta_learner(self, train_preds, y_train):
        """Train the meta-learner for stacking"""
        print("\n=== TRAINING META-LEARNER ===")
        
        # Train meta-learner
        self.meta_learner.fit(train_preds, y_train)
        
        print(f"Meta-learner coefficients: {self.meta_learner.coef_[0]}")
        
        return self.meta_learner
    
    def evaluate_models(self, X_test, y_test):
        """Evaluate all models and ensemble"""
        print("\n=== EVALUATING MODELS ===")
        
        X_test_scaled = self.scaler.transform(X_test)
        
        # Evaluate base learners
        models = {
            'Random Forest': self.rf_model,
            'XGBoost': self.xgb_model,
            'LightGBM': self.lgb_model
        }
        
        results = {}
        for name, model in models.items():
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            
            results[name] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'auc_roc': roc_auc_score(y_test, y_pred_proba),
                'f1_score': f1_score(y_test, y_pred),
                'precision': (y_pred * y_test).sum() / max(y_pred.sum(), 1),
                'recall': (y_pred * y_test).sum() / max(y_test.sum(), 1),
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
        
        # Evaluate ensemble
        test_preds = np.column_stack([
            self.rf_model.predict_proba(X_test_scaled)[:, 1],
            self.xgb_model.predict_proba(X_test_scaled)[:, 1],
            self.lgb_model.predict_proba(X_test_scaled)[:, 1]
        ])
        
        y_ensemble_pred = self.meta_learner.predict(test_preds)
        y_ensemble_proba = self.meta_learner.predict_proba(test_preds)[:, 1]
        
        results['Ensemble'] = {
            'accuracy': accuracy_score(y_test, y_ensemble_pred),
            'auc_roc': roc_auc_score(y_test, y_ensemble_proba),
            'f1_score': f1_score(y_test, y_ensemble_pred),
            'precision': (y_ensemble_pred * y_test).sum() / max(y_ensemble_pred.sum(), 1),
            'recall': (y_ensemble_pred * y_test).sum() / max(y_test.sum(), 1),
            'predictions': y_ensemble_pred,
            'probabilities': y_ensemble_proba
        }
        
        self.results = results
        return results
    
    def analyze_feature_importance(self, X_test):
        """Analyze feature importance using multiple methods"""
        print("\n=== FEATURE IMPORTANCE ANALYSIS ===")
        
        X_test_scaled = self.scaler.transform(X_test)
        
        # Random Forest feature importance
        rf_importance = self.rf_model.feature_importances_
        
        # XGBoost feature importance
        xgb_importance = self.xgb_model.feature_importances_
        
        # LightGBM feature importance
        lgb_importance = self.lgb_model.feature_importances_
        
        self.feature_importance = {
            'Random Forest': dict(zip(self.feature_names, rf_importance)),
            'XGBoost': dict(zip(self.feature_names, xgb_importance)),
            'LightGBM': dict(zip(self.feature_names, lgb_importance))
        }
        
        return self.feature_importance
    
    def create_visualizations(self, save_path='imgs/'):
        """Create comprehensive visualizations"""
        print("\n=== CREATING VISUALIZATIONS ===")
        
        os.makedirs(save_path, exist_ok=True)
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Model Performance Comparison
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        models = list(self.results.keys())
        metrics = ['accuracy', 'auc_roc', 'f1_score']
        
        for i, metric in enumerate(metrics):
            values = [self.results[model][metric] for model in models]
            axes[0, i].bar(models, values, alpha=0.7, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
            axes[0, i].set_title(f'{metric.replace("_", " ").title()} Comparison')
            axes[0, i].set_ylabel(metric.replace("_", " ").title())
            axes[0, i].tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for j, v in enumerate(values):
                axes[0, i].text(j, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # ROC Curves
        axes[1, 0].set_title('ROC Curves')
        for i, model in enumerate(models):
            fpr, tpr, _ = roc_curve(self.y_test, self.results[model]['probabilities'])
            auc = self.results[model]['auc_roc']
            axes[1, 0].plot(fpr, tpr, label=f'{model} (AUC = {auc:.3f})', linewidth=2)
        
        axes[1, 0].plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier')
        axes[1, 0].set_xlabel('False Positive Rate')
        axes[1, 0].set_ylabel('True Positive Rate')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Precision-Recall Curves
        axes[1, 1].set_title('Precision-Recall Curves')
        for i, model in enumerate(models):
            precision, recall, _ = precision_recall_curve(self.y_test, self.results[model]['probabilities'])
            axes[1, 1].plot(recall, precision, label=f'{model}', linewidth=2)
        
        axes[1, 1].set_xlabel('Recall')
        axes[1, 1].set_ylabel('Precision')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{save_path}model_performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Feature Importance Plot
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        
        for i, (model_name, importance) in enumerate(self.feature_importance.items()):
            sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
            features, importances = zip(*sorted_features[:10])  # Top 10 features
            
            axes[i].barh(range(len(features)), importances, color=colors[i], alpha=0.7)
            axes[i].set_yticks(range(len(features)))
            axes[i].set_yticklabels(features)
            axes[i].set_xlabel('Feature Importance')
            axes[i].set_title(f'{model_name} Feature Importance')
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{save_path}feature_importance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Calibration Plot
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        
        for i, model in enumerate(models):
            prob_true, prob_pred = calibration_curve(self.y_test, self.results[model]['probabilities'], 
                                                   n_bins=10)
            ax.plot(prob_pred, prob_true, marker='o', label=f'{model}', linewidth=2)
        
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect Calibration')
        ax.set_xlabel('Mean Predicted Probability')
        ax.set_ylabel('Fraction of Positives')
        ax.set_title('Model Calibration Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.savefig(f'{save_path}calibration_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Confusion Matrices
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for i, (model, results) in enumerate(self.results.items()):
            cm = confusion_matrix(self.y_test, results['predictions'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
            axes[i].set_title(f'{model}\nAccuracy: {results["accuracy"]:.3f}')
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('Actual')
        
        plt.tight_layout()
        plt.savefig(f'{save_path}confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Visualizations saved to {save_path}")
    
    def save_results(self, save_path='results/'):
        """Save all results to files"""
        print("\n=== SAVING RESULTS ===")
        os.makedirs(save_path, exist_ok=True)
        
        # Save model performance metrics
        results_df = pd.DataFrame(self.results).T
        results_df.to_csv(f'{save_path}model_performance_metrics.csv')
        
        # Save feature importance
        importance_df = pd.DataFrame(self.feature_importance)
        importance_df.to_csv(f'{save_path}feature_importance.csv')
        
        # Save detailed results as JSON
        detailed_results = {}
        for model, metrics in self.results.items():
            detailed_results[model] = {k: v.tolist() if isinstance(v, np.ndarray) else v 
                                     for k, v in metrics.items()}
        
        with open(f'{save_path}detailed_results.json', 'w') as f:
            json.dump(detailed_results, f, indent=2)
        
        print(f"Results saved to {save_path}")
    
    def run_complete_pipeline(self, data_path):
        """Run the complete prediction pipeline"""
        print("=" * 80)
        print("DIABETES PREDICTION HYBRID ENSEMBLE FRAMEWORK")
        print("=" * 80)
        
        # Load and preprocess data
        X, y = self.load_and_preprocess_data(data_path)
        
        # Split data
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )
        
        # Handle class imbalance
        X_train_resampled, y_train_resampled = self.handle_class_imbalance(X_temp, y_temp)
        
        # Split for stacking validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_resampled, y_train_resampled, test_size=0.2, 
            random_state=self.random_state, stratify=y_train_resampled
        )
        
        # Store test data for later use
        self.X_test = X_test
        self.y_test = y_test
        
        print(f"\nData split summary:")
        print(f"Training set size: {len(X_train)}")
        print(f"Validation set size: {len(X_val)}")
        print(f"Test set size: {len(X_test)}")
        
        # Train base learners
        self.train_base_learners(X_train, y_train)
        
        # Create stacking predictions
        train_preds, val_preds = self.create_stacking_predictions(X_train, y_train, X_val)
        
        # Train meta-learner
        self.train_meta_learner(train_preds, y_train)
        
        # Evaluate models
        results = self.evaluate_models(X_test, y_test)
        
        # Analyze feature importance
        feature_importance = self.analyze_feature_importance(X_test)
        
        # Create visualizations
        self.create_visualizations()
        
        # Save results
        self.save_results()
        
        print("\n" + "=" * 80)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        
        return results, feature_importance

def main():
    """Main execution function"""
    print("Starting Diabetes Prediction Ensemble Framework...")
    
    # Download dataset if not exists
    import urllib.request
    import os
    
    dataset_path = 'data/pima_indians_diabetes.csv'
    os.makedirs('data', exist_ok=True)
    
    if not os.path.exists(dataset_path):
        print("Downloading Pima Indians Diabetes Dataset...")
        url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv'
        urllib.request.urlretrieve(url, dataset_path)
        print("Dataset downloaded successfully!")
    
    # Initialize and run the ensemble
    ensemble = DiabetesPredictionEnsemble(random_state=RANDOM_STATE)
    results, feature_importance = ensemble.run_complete_pipeline(dataset_path)
    
    # Print summary results
    print("\n" + "=" * 80)
    print("FINAL RESULTS SUMMARY")
    print("=" * 80)
    
    for model_name, metrics in results.items():
        print(f"\n{model_name}:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  AUC-ROC: {metrics['auc_roc']:.4f}")
        print(f"  F1-Score: {metrics['f1_score']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
    
    print("\nTop 5 features by average importance:")
    avg_importance = {}
    for feature in ensemble.feature_names:
        avg_importance[feature] = np.mean([feature_importance[model][feature] 
                                         for model in feature_importance.keys()])
    
    sorted_features = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)
    for i, (feature, importance) in enumerate(sorted_features[:5]):
        print(f"  {i+1}. {feature}: {importance:.4f}")
    
    return ensemble, results, feature_importance

if __name__ == "__main__":
    ensemble, results, feature_importance = main()