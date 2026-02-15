"""
Train all 6 classification models on UCI Bank Marketing Dataset.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import joblib


def load_and_preprocess_data(data_path):
    """
    Load and preprocess the Bank Marketing Dataset.
    
    Parameters:
    -----------
    data_path : str
        Path to the dataset file.
    
    Returns:
    --------
    tuple : (X_train, X_test, y_train, y_test, feature_names)
    """
    
    # Load dataset with semicolon separator
    df = pd.read_csv(data_path, sep=';')
    
    print(f"Dataset shape: {df.shape}")
    print(f"Target distribution:\n{df['y'].value_counts()}")
    
    # Encode target variable
    df['y'] = (df['y'] == 'yes').astype(int)
    
    # Separate features and target
    X = df.drop('y', axis=1)
    y = df['y']
    
    # One-hot encode categorical features
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    
    print(f"Features after encoding: {X.shape[1]}")
    
    # Train-test split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrame for consistency
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, X.columns.tolist()


def train_all_models(X_train, X_test, y_train, y_test, models_dir):
    """
    Train all 6 classification models.
    
    Parameters:
    -----------
    X_train : DataFrame
        Training features.
    X_test : DataFrame
        Testing features.
    y_train : Series
        Training target.
    y_test : Series
        Testing target.
    models_dir : str
        Directory to save trained models.
    """
    
    # Create models directory if it doesn't exist
    os.makedirs(models_dir, exist_ok=True)
    
    models = {
        'logistic': LogisticRegression(max_iter=1000, random_state=42),
        'dt': DecisionTreeClassifier(random_state=42),
        'knn': KNeighborsClassifier(n_neighbors=5),
        'nb': GaussianNB(),
        'rf': RandomForestClassifier(n_estimators=100, random_state=42),
        'xgb': XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss', verbosity=0)
    }
    
    results = {}
    
    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")
        
        # Train model
        model.fit(X_train, y_train)
        
        # Evaluate
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        
        print(f"  Train Accuracy: {train_score:.4f}")
        print(f"  Test Accuracy: {test_score:.4f}")
        
        # Save model
        model_path = os.path.join(models_dir, f"{model_name}.pkl")
        joblib.dump(model, model_path)
        print(f"  Saved to {model_path}")
        
        results[model_name] = {
            'model': model,
            'train_score': train_score,
            'test_score': test_score
        }
    
    return results


def main():
    """Main execution function."""
    
    # Paths
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'bank-additional-full.csv')
    models_dir = os.path.join(os.path.dirname(__file__), 'saved_models')
    
    print("="*60)
    print("Loading and preprocessing data...")
    print("="*60)
    
    X_train, X_test, y_train, y_test, feature_names = load_and_preprocess_data(data_path)
    
    print("\n" + "="*60)
    print("Training all 6 classification models...")
    print("="*60)
    
    results = train_all_models(X_train, X_test, y_train, y_test, models_dir)
    
    print("\n" + "="*60)
    print("All models trained and saved successfully!")
    print("="*60)
    print(f"\nModels saved in: {models_dir}")
    print("Files created:")
    for model_name in results.keys():
        print(f"  - {model_name}.pkl")


if __name__ == "__main__":
    main()
