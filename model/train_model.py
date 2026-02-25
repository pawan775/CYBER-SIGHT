"""
Cyber-Sight: ML Model Training Module
=====================================
This module trains machine learning models for:
1. Cyber threat type classification (phishing, malware, hacking, safe)
2. Risk level prediction (low, medium, high)
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.preprocessing import DataPreprocessor


class CyberThreatModelTrainer:
    """
    Trainer class for cyber threat detection ML models.
    Trains multiple models and selects the best performer.
    """
    
    def __init__(self, data_path: str = None):
        """
        Initialize the model trainer.
        
        Args:
            data_path: Path to the training dataset
        """
        self.data_path = data_path or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'data', 'cybercrime_dataset.csv'
        )
        self.preprocessor = DataPreprocessor()
        self.models = {}
        self.best_attack_model = None
        self.best_risk_model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = []
        self.training_stats = {}
        
    def load_and_prepare_data(self):
        """Load dataset and prepare features."""
        print("="*60)
        print("CYBER-SIGHT ML MODEL TRAINING")
        print("="*60)
        print(f"\n📂 Loading dataset from: {self.data_path}")
        
        # Load dataset
        df = self.preprocessor.load_dataset(self.data_path)
        
        # Display dataset info
        print(f"\n📊 Dataset Statistics:")
        print(f"   Total samples: {len(df)}")
        print(f"   Columns: {list(df.columns)}")
        
        if 'attack_type' in df.columns:
            print(f"\n   Attack Type Distribution:")
            for attack, count in df['attack_type'].value_counts().items():
                print(f"      {attack}: {count} ({count/len(df)*100:.1f}%)")
        
        if 'risk_level' in df.columns:
            print(f"\n   Risk Level Distribution:")
            for risk, count in df['risk_level'].value_counts().items():
                print(f"      {risk}: {count} ({count/len(df)*100:.1f}%)")
        
        # Preprocess data
        print("\n🔧 Preprocessing data...")
        X, y_attack, y_risk = self.preprocessor.preprocess_dataset(df)
        
        # Store feature columns and label encoders
        self.feature_columns = self.preprocessor.get_feature_names()
        self.label_encoders = self.preprocessor.label_encoders.copy()
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, y_attack, y_risk, df
    
    def train_models(self, X, y_attack, y_risk, test_size=0.2):
        """
        Train multiple ML models and select the best performers.
        
        Args:
            X: Scaled feature array
            y_attack: Attack type labels
            y_risk: Risk level labels
            test_size: Proportion of data for testing
        """
        print("\n" + "="*60)
        print("MODEL TRAINING")
        print("="*60)
        
        # Split data
        X_train, X_test, y_attack_train, y_attack_test, y_risk_train, y_risk_test = \
            train_test_split(X, y_attack, y_risk, test_size=test_size, random_state=42, stratify=y_attack)
        
        print(f"\n📊 Data Split:")
        print(f"   Training samples: {len(X_train)}")
        print(f"   Testing samples: {len(X_test)}")
        
        # Define models to train
        model_configs = {
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            ),
            'Logistic Regression': LogisticRegression(
                max_iter=1000,
                random_state=42
            ),
            'SVM': SVC(
                kernel='rbf',
                probability=True,
                random_state=42
            )
        }
        
        # ============ TRAIN ATTACK TYPE CLASSIFIER ============
        print("\n" + "-"*60)
        print("🎯 TRAINING ATTACK TYPE CLASSIFIER")
        print("-"*60)
        
        attack_results = {}
        
        for name, model in model_configs.items():
            print(f"\n   Training {name}...")
            
            try:
                # Train model
                model.fit(X_train, y_attack_train)
                
                # Make predictions
                y_pred = model.predict(X_test)
                
                # Calculate metrics
                accuracy = accuracy_score(y_attack_test, y_pred)
                precision = precision_score(y_attack_test, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_attack_test, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_attack_test, y_pred, average='weighted', zero_division=0)
                
                # Cross-validation score
                cv_scores = cross_val_score(model, X_train, y_attack_train, cv=5)
                
                attack_results[name] = {
                    'model': model,
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std()
                }
                
                print(f"      Accuracy: {accuracy:.4f}")
                print(f"      F1 Score: {f1:.4f}")
                print(f"      CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
                
            except Exception as e:
                print(f"      [!] Error training {name}: {e}")
        
        # Select best attack model
        best_attack_name = max(attack_results, key=lambda x: attack_results[x]['f1_score'])
        self.best_attack_model = attack_results[best_attack_name]['model']
        print(f"\n   [OK] Best Attack Classifier: {best_attack_name}")
        print(f"     F1 Score: {attack_results[best_attack_name]['f1_score']:.4f}")
        
        # Detailed classification report for best model
        y_pred_best = self.best_attack_model.predict(X_test)
        print(f"\n   Classification Report:")
        target_names = list(self.label_encoders['attack_type'].classes_)
        print(classification_report(y_attack_test, y_pred_best, target_names=target_names))
        
        # ============ TRAIN RISK LEVEL CLASSIFIER ============
        print("\n" + "-"*60)
        print("[!] TRAINING RISK LEVEL CLASSIFIER")
        print("-"*60)
        
        risk_results = {}
        
        for name, model_class in [
            ('Random Forest', RandomForestClassifier(n_estimators=100, random_state=42)),
            ('Gradient Boosting', GradientBoostingClassifier(n_estimators=100, random_state=42)),
            ('Logistic Regression', LogisticRegression(max_iter=1000, random_state=42))
        ]:
            print(f"\n   Training {name}...")
            
            try:
                # Train model
                model_class.fit(X_train, y_risk_train)
                
                # Make predictions
                y_pred = model_class.predict(X_test)
                
                # Calculate metrics
                accuracy = accuracy_score(y_risk_test, y_pred)
                f1 = f1_score(y_risk_test, y_pred, average='weighted', zero_division=0)
                
                risk_results[name] = {
                    'model': model_class,
                    'accuracy': accuracy,
                    'f1_score': f1
                }
                
                print(f"      Accuracy: {accuracy:.4f}")
                print(f"      F1 Score: {f1:.4f}")
                
            except Exception as e:
                print(f"      ⚠ Error training {name}: {e}")
        
        # Select best risk model
        best_risk_name = max(risk_results, key=lambda x: risk_results[x]['f1_score'])
        self.best_risk_model = risk_results[best_risk_name]['model']
        print(f"\n   [OK] Best Risk Classifier: {best_risk_name}")
        print(f"     F1 Score: {risk_results[best_risk_name]['f1_score']:.4f}")
        
        # Store training statistics
        self.training_stats = {
            'attack_model': best_attack_name,
            'attack_accuracy': attack_results[best_attack_name]['accuracy'],
            'attack_f1': attack_results[best_attack_name]['f1_score'],
            'risk_model': best_risk_name,
            'risk_accuracy': risk_results[best_risk_name]['accuracy'],
            'risk_f1': risk_results[best_risk_name]['f1_score'],
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'features': len(self.feature_columns),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Feature importance (if available)
        if hasattr(self.best_attack_model, 'feature_importances_'):
            print("\n   📊 Feature Importance (Attack Model):")
            importances = self.best_attack_model.feature_importances_
            indices = np.argsort(importances)[::-1]
            for i in range(min(10, len(self.feature_columns))):
                print(f"      {i+1}. {self.feature_columns[indices[i]]}: {importances[indices[i]]:.4f}")
        
        return attack_results, risk_results
    
    def save_models(self, output_dir: str = None):
        """
        Save trained models and preprocessing objects.
        
        Args:
            output_dir: Directory to save models
        """
        if output_dir is None:
            output_dir = os.path.dirname(os.path.abspath(__file__))
        
        model_path = os.path.join(output_dir, 'threat_model.pkl')
        
        print("\n" + "="*60)
        print("SAVING MODELS")
        print("="*60)
        
        # Save all components in a single file
        model_data = {
            'attack_model': self.best_attack_model,
            'risk_model': self.best_risk_model,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_columns': self.feature_columns,
            'training_stats': self.training_stats
        }
        
        joblib.dump(model_data, model_path)
        print(f"\n[OK] Models saved to: {model_path}")
        print(f"   File size: {os.path.getsize(model_path) / 1024:.2f} KB")
        
        # Save training report
        report_path = os.path.join(output_dir, 'training_report.txt')
        with open(report_path, 'w') as f:
            f.write("CYBER-SIGHT MODEL TRAINING REPORT\n")
            f.write("="*50 + "\n\n")
            f.write(f"Training Date: {self.training_stats['timestamp']}\n\n")
            f.write("ATTACK TYPE CLASSIFIER\n")
            f.write("-"*30 + "\n")
            f.write(f"Model: {self.training_stats['attack_model']}\n")
            f.write(f"Accuracy: {self.training_stats['attack_accuracy']:.4f}\n")
            f.write(f"F1 Score: {self.training_stats['attack_f1']:.4f}\n\n")
            f.write("RISK LEVEL CLASSIFIER\n")
            f.write("-"*30 + "\n")
            f.write(f"Model: {self.training_stats['risk_model']}\n")
            f.write(f"Accuracy: {self.training_stats['risk_accuracy']:.4f}\n")
            f.write(f"F1 Score: {self.training_stats['risk_f1']:.4f}\n\n")
            f.write("DATASET INFO\n")
            f.write("-"*30 + "\n")
            f.write(f"Training Samples: {self.training_stats['training_samples']}\n")
            f.write(f"Test Samples: {self.training_stats['test_samples']}\n")
            f.write(f"Features: {self.training_stats['features']}\n")
            f.write(f"Feature Names: {', '.join(self.feature_columns)}\n")
        
        print(f"[OK] Training report saved to: {report_path}")
        
        return model_path
    
    def train_and_save(self):
        """Complete training pipeline."""
        # Load and prepare data
        X, y_attack, y_risk, df = self.load_and_prepare_data()
        
        # Train models
        self.train_models(X, y_attack, y_risk)
        
        # Save models
        model_path = self.save_models()
        
        print("\n" + "="*60)
        print("[SUCCESS] TRAINING COMPLETE!")
        print("="*60)
        print(f"\nModel ready for deployment.")
        print(f"Use the model with:")
        print(f"   model_data = joblib.load('{model_path}')")
        
        return model_path


def load_model(model_path: str = None):
    """
    Load a trained model.
    
    Args:
        model_path: Path to the model file
        
    Returns:
        Dictionary containing model components
    """
    if model_path is None:
        model_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'threat_model.pkl'
        )
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    return joblib.load(model_path)


def predict_threat(url_features: dict, model_data: dict) -> dict:
    """
    Predict threat type and risk level for given URL features.
    
    Args:
        url_features: Dictionary of extracted URL features
        model_data: Loaded model data dictionary
        
    Returns:
        Dictionary with predictions
    """
    # Extract components
    attack_model = model_data['attack_model']
    risk_model = model_data['risk_model']
    scaler = model_data['scaler']
    label_encoders = model_data['label_encoders']
    feature_columns = model_data['feature_columns']
    
    # Prepare features array
    features = [url_features.get(col, 0) for col in feature_columns]
    features_scaled = scaler.transform([features])
    
    # Make predictions
    attack_pred = attack_model.predict(features_scaled)[0]
    attack_proba = attack_model.predict_proba(features_scaled)[0]
    
    risk_pred = risk_model.predict(features_scaled)[0]
    risk_proba = risk_model.predict_proba(features_scaled)[0]
    
    # Decode predictions
    attack_type = label_encoders['attack_type'].inverse_transform([attack_pred])[0]
    risk_level = label_encoders['risk_level'].inverse_transform([risk_pred])[0]
    
    return {
        'attack_type': attack_type,
        'attack_confidence': max(attack_proba),
        'risk_level': risk_level,
        'risk_confidence': max(risk_proba),
        'attack_probabilities': dict(zip(
            label_encoders['attack_type'].classes_,
            attack_proba
        )),
        'risk_probabilities': dict(zip(
            label_encoders['risk_level'].classes_,
            risk_proba
        ))
    }


if __name__ == "__main__":
    print("\n" + "== "*20 + "\n")
    print("     CYBER-SIGHT: ML MODEL TRAINER")
    print("     Global Cyber Crime Detection System")
    print("\n" + "== "*20 + "\n")
    
    # Initialize trainer
    trainer = CyberThreatModelTrainer()
    
    # Run training pipeline
    try:
        model_path = trainer.train_and_save()
        print(f"\n[SUCCESS] Model training successful!")
        print(f"   Saved to: {model_path}")
    except Exception as e:
        print(f"\n[ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()
