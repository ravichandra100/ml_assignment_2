"""
Evaluate trained classification models using multiple metrics.
"""

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report
)


def compute_metrics(y_true, y_pred, y_pred_proba=None):
    """
    Compute comprehensive metrics for binary classification.
    
    Parameters:
    -----------
    y_true : array-like
        True binary labels (0 or 1).
    y_pred : array-like
        Predicted binary labels (0 or 1).
    y_pred_proba : array-like, optional
        Predicted probabilities for positive class.
        If None, AUC will be computed from y_pred.
    
    Returns:
    --------
    dict : Dictionary containing all required metrics.
        Keys: 'accuracy', 'auc', 'precision', 'recall', 'f1', 'mcc'
    """
    
    metrics = {}
    
    # Accuracy
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    
    # AUC
    if y_pred_proba is not None:
        metrics['auc'] = roc_auc_score(y_true, y_pred_proba)
    else:
        metrics['auc'] = roc_auc_score(y_true, y_pred)
    
    # Precision
    metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
    
    # Recall
    metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
    
    # F1 Score
    metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)
    
    # Matthews Correlation Coefficient
    metrics['mcc'] = matthews_corrcoef(y_true, y_pred)
    
    return metrics


def get_confusion_matrix(y_true, y_pred):
    """
    Compute confusion matrix.
    
    Parameters:
    -----------
    y_true : array-like
        True binary labels.
    y_pred : array-like
        Predicted binary labels.
    
    Returns:
    --------
    ndarray : 2D confusion matrix.
    """
    return confusion_matrix(y_true, y_pred)


def get_classification_report(y_true, y_pred, target_names=None):
    """
    Generate detailed classification report.
    
    Parameters:
    -----------
    y_true : array-like
        True binary labels.
    y_pred : array-like
        Predicted binary labels.
    target_names : list, optional
        Names for classes (e.g., ['No', 'Yes']).
    
    Returns:
    --------
    str : Classification report as formatted string.
    """
    if target_names is None:
        target_names = ['No', 'Yes']
    return classification_report(y_true, y_pred, target_names=target_names)
