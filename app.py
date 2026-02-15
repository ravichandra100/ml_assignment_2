"""
Streamlit application for binary classification prediction.
Deploy Bank Marketing Classification models with interactive UI.
"""

import os
import io
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from model.evaluate_models import compute_metrics, get_confusion_matrix, get_classification_report
import matplotlib.pyplot as plt
import seaborn as sns


# Set page config
st.set_page_config(
    page_title="Bank Marketing Classification",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("📊 Bank Marketing Classification")
st.markdown("**Binary Classification with 6 ML Models**")
st.markdown("---")


def load_models():
    """Load all trained models from saved_models directory."""
    models_dir = os.path.join(os.path.dirname(__file__), 'model', 'saved_models')
    models = {}
    model_files = ['logistic.pkl', 'dt.pkl', 'knn.pkl', 'nb.pkl', 'rf.pkl', 'xgb.pkl']
    
    for model_file in model_files:
        model_path = os.path.join(models_dir, model_file)
        if os.path.exists(model_path):
            model_name = model_file.replace('.pkl', '')
            models[model_name] = joblib.load(model_path)
    
    return models


def preprocess_input_data(df, training_columns):
    """
    Preprocess input data to match training format.
    
    Parameters:
    -----------
    df : DataFrame
        Input data to preprocess.
    training_columns : list
        Column names from training data.
    
    Returns:
    --------
    DataFrame : Preprocessed data matching training format.
    """
    
    # One-hot encode categorical features
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    # Align with training columns
    for col in training_columns:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    
    # Keep only training columns in same order
    df_encoded = df_encoded[training_columns]
    
    # Standardize
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_encoded)
    df_scaled = pd.DataFrame(df_scaled, columns=training_columns)
    
    return df_scaled


@st.cache_resource
def load_all_models():
    """Cache loaded models."""
    return load_models()


def load_default_data():
    """Load default data from data/ folder."""
    data_path = os.path.join(os.path.dirname(__file__), 'data', 'bank-additional-full.csv')
    if os.path.exists(data_path):
        return pd.read_csv(data_path, sep=';')
    return None


def main():
    """Main Streamlit application."""
    
    # Load models
    models = load_all_models()
    
    if not models:
        st.error("❌ No trained models found in model/saved_models/")
        st.info("Please run `python model/train_models.py` first to train models.")
        return
    
    # Sidebar
    st.sidebar.header("⚙️ Configuration")
    
    model_name = st.sidebar.selectbox(
        "Select Model",
        list(models.keys()),
        format_func=lambda x: {
            'logistic': 'Logistic Regression',
            'dt': 'Decision Tree',
            'knn': 'KNN',
            'nb': 'Naive Bayes',
            'rf': 'Random Forest',
            'xgb': 'XGBoost'
        }.get(x, x)
    )
    
    selected_model = models[model_name]
    
    # Main content
    st.subheader(f"Model: {model_name.upper()}")
    
    # Load data - try default first, then allow upload
    st.markdown("### 📁 Data Source")
    
    df = load_default_data()
    data_source = st.radio("Choose data source:", ["Use Default Data", "Upload CSV File"])
    
    if data_source == "Upload CSV File":
        uploaded_file = st.file_uploader("Upload CSV file", type='csv')
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file, sep=';')
        else:
            df = None
    
    if df is not None:
        try:
            st.success(f"✅ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
            
            # Display preview
            with st.expander("📋 Data Preview"):
                st.dataframe(df.head(10), use_container_width=True)
            
            # Check if target column exists
            if 'y' in df.columns:
                st.markdown("### 📊 Model Evaluation")
                
                # Prepare data
                X = df.drop('y', axis=1)
                y_true = (df['y'] == 'yes').astype(int)
                
                # Get training feature names (approximate from model)
                # For simplicity, we'll preprocess and predict
                try:
                    # One-hot encode
                    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
                    X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
                    
                    # Standardize
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X_encoded)
                    X_scaled = pd.DataFrame(X_scaled, columns=X_encoded.columns)
                    
                    # Make predictions
                    y_pred = selected_model.predict(X_scaled)
                    
                    # Get probabilities if available
                    if hasattr(selected_model, 'predict_proba'):
                        y_proba = selected_model.predict_proba(X_scaled)[:, 1]
                    else:
                        y_proba = None
                    
                    # Compute metrics
                    metrics = compute_metrics(y_true, y_pred, y_proba)
                    
                    # Display metrics in columns
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Accuracy", f"{metrics['accuracy']:.4f}")
                    with col2:
                        st.metric("AUC", f"{metrics['auc']:.4f}")
                    with col3:
                        st.metric("F1 Score", f"{metrics['f1']:.4f}")
                    
                    col4, col5, col6 = st.columns(3)
                    with col4:
                        st.metric("Precision", f"{metrics['precision']:.4f}")
                    with col5:
                        st.metric("Recall", f"{metrics['recall']:.4f}")
                    with col6:
                        st.metric("MCC", f"{metrics['mcc']:.4f}")
                    
                    # Classification Report
                    st.markdown("### 📄 Classification Report")
                    report_text = get_classification_report(y_true, y_pred)
                    st.text(report_text)
                    
                    # Confusion Matrix
                    st.markdown("### 📈 Confusion Matrix")
                    cm = get_confusion_matrix(y_true, y_pred)
                    
                    fig, ax = plt.subplots(figsize=(6, 5))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax,
                                xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
                    ax.set_xlabel('Predicted')
                    ax.set_ylabel('Actual')
                    ax.set_title(f'Confusion Matrix - {model_name.upper()}')
                    st.pyplot(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"❌ Error during prediction: {str(e)}")
            
            else:
                st.warning("⚠️ Target column 'y' not found. Use data with target variable for evaluation.")
                
                # Allow prediction without target
                st.markdown("### 🔮 Make Predictions")
                try:
                    X = df
                    
                    # One-hot encode
                    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
                    X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
                    
                    # Standardize
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X_encoded)
                    X_scaled = pd.DataFrame(X_scaled, columns=X_encoded.columns)
                    
                    # Make predictions
                    y_pred = selected_model.predict(X_scaled)
                    
                    # Get probabilities if available
                    if hasattr(selected_model, 'predict_proba'):
                        y_proba = selected_model.predict_proba(X_scaled)[:, 1]
                        results_df = pd.DataFrame({
                            'Prediction': y_pred,
                            'Probability_Yes': y_proba
                        })
                    else:
                        results_df = pd.DataFrame({'Prediction': y_pred})
                    
                    st.dataframe(results_df, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"❌ Error during prediction: {str(e)}")
        
        except Exception as e:
            st.error(f"❌ Error processing data: {str(e)}")
    
    else:
        st.info("⚠️ No data available. Please check that 'bank-additional-full.csv' exists in the 'data/' folder or upload a CSV file.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **Project:** ML Assignment 2 - Binary Classification  
    **Dataset:** UCI Bank Marketing Dataset  
    **Models:** Logistic Regression, Decision Tree, KNN, Naive Bayes, Random Forest, XGBoost  
    **Lab:** BITS Lab
    """)


if __name__ == "__main__":
    main()
