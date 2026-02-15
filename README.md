# ML Assignment 2: Implement multi classification models

## Problem Statement

The objective of this project is to develop a **binary classification system** that predicts whether a bank client will subscribe to a term deposit based on their marketing campaign data. This assignment involves implementing and comparing **6 different machine learning models** on the UCI Bank Marketing Dataset and deploying the solution using a **Streamlit web application**.

The problem addresses a real-world business scenario where banks need to identify potential customers likely to subscribe to term deposits, enabling targeted marketing efforts and improved resource allocation.

---
## Model Performance Comparison

### Performance Metrics Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---------------|----------|-----|-----------|--------|----------|-----|
| Logistic Regression | 0.9112 | 0.9362 | 0.6688 | 0.4196 | 0.5157 | 0.4851 |
| Decision Tree | 0.9602 | 0.9086 | 0.8119 | 0.8420 | 0.8267 | 0.8044 |
| KNN | 0.9168 | 0.9300 | 0.7217 | 0.4252 | 0.5351 | 0.5136 |
| Naive Bayes | 0.8871 | 0.8418 | 0.2222 | 0.0009 | 0.0017 | 0.0072 |
| Random Forest (Ensemble) | 0.9803 | 0.9938 | 0.9463 | 0.8744 | 0.9089 | 0.8988 |
| XGBoost (Ensemble) | 0.8813 | 0.9397 | 0.4841 | 0.8157 | 0.6076 | 0.5687 |

### Model Performance Observations

| ML Model Name | Observation about Model Performance |
|---------------|-------------------------------------|
| **Logistic Regression** | Demonstrates good overall accuracy (91.12%) and excellent AUC (0.9362), indicating strong discriminative ability. However, the recall is low (0.4196), meaning it misses ~58% of actual positive cases (term deposit subscribers). The model is conservative in positive predictions (high precision 0.6688). This linear model provides reasonable baseline performance but struggles with the non-linear decision boundary of this classification problem. |
| **Decision Tree** | Strong performer with 96.02% accuracy and balanced precision-recall (0.8119 / 0.8420). The F1 score of 0.8267 indicates good balance between precision and recall. Decision trees effectively capture the non-linear relationships in the data. However, the AUC (0.9086) is slightly lower than some other models, and the model may overfit on training data due to its high complexity without pruning. |
| **KNN** | Achieves 91.68% accuracy with high AUC (0.9300), showing good class separation ability. However, similar to Logistic Regression, it has low recall (0.4252) and moderate F1 score (0.5351), indicating it's conservative in positive predictions. As a distance-based lazy learner, KNN is sensitive to feature scaling but benefits from StandardScaler preprocessing. Memory-intensive for larger datasets. |
| **Naive Bayes** | Poor performance with extremely low recall (0.0009) and F1 score (0.0017), despite moderate accuracy (88.71%). This indicates the model almost never predicts positive cases. The assumption of feature independence in Gaussian Naive Bayes is violated in this dataset with correlated features. The MCC (0.0072) is near zero, meaning the model provides little better than random classification for the minority class. Not recommended for this problem. |
| **Random Forest (Ensemble) | **BEST OVERALL PERFORMER** with the highest accuracy (98.03%) and AUC (0.9938). Excellent balanced performance: precision 0.9463, recall 0.8744, F1 0.9089, and MCC 0.8988. The ensemble method effectively reduces overfitting while capturing complex non-linear relationships. Multiple decision trees voting reduces variance. This model best identifies both negative and positive cases with minimal false positives and false negatives. Recommended for production deployment. |
| **XGBoost (Ensemble)** | Gradient boosting ensemble with strong AUC (0.9397) and high recall (0.8157), effectively identifying most positive cases. However, precision is lower (0.4841), resulting in higher false positive rate. Accuracy is lower (88.13%) than Random Forest. The F1 score (0.6076) shows good but not excellent balance. XGBoost excels at handling complex patterns but in this case is outperformed by Random Forest. Still a strong alternative model. |

---

## Dataset Description

**UCI Bank Marketing Dataset**

- **Source:** UCI Machine Learning Repository
- **Format:** CSV file with semicolon separator (`;`)
- **Total Records:** 45,211 instances
- **Total Features:** 20 input attributes + 1 target variable
- **Target Variable:** `y` (binary: 'yes' or 'no') - whether client subscribed to a term deposit
- **Feature Categories:**
  - **Demographic:** age, job, marital status, education, housing loan, personal loan
  - **Campaign Related:** number of contacts, contact type, day of week, duration of call
  - **Economic Indicators:** employment variation rate, consumer price index, consumer confidence index, euribor rate, number of employees
  - **Previous Campaign:** number of contacts in previous campaign, outcome of previous campaign

### Data Preprocessing Steps
1. **Target Encoding:** Binary transformation (0 = No subscription, 1 = Yes subscription)
2. **Categorical Encoding:** One-hot encoding applied to all categorical features
3. **Feature Scaling:** StandardScaler used to normalize numerical features to zero mean and unit variance
4. **Train-Test Split:** 80% training set, 20% testing set with stratified sampling to maintain class distribution

---

## Models Used

| # | Model | Algorithm Type | Key Characteristics |
|---|-------|-----------------|-------------------|
| 1 | **Logistic Regression** | Linear | Probabilistic, interpretable, baseline model |
| 2 | **Decision Tree** | Tree-based | Non-parametric, prone to overfitting, easy to visualize |
| 3 | **K-Nearest Neighbors (kNN)** | Distance-based | Lazy learner, memory-intensive, no training phase |
| 4 | **Naive Bayes** | Probabilistic | Fast training, assumes feature independence |
| 5 | **Random Forest** | Ensemble | Robust ensemble method, handles non-linearity well |
| 6 | **XGBoost** | Gradient Boosting | Sequential ensemble, powerful, often achieves best performance |

---

## Model Evaluation Metrics

All models are comprehensively evaluated using the following 6 metrics:

1. **Accuracy**
   - Formula: $\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$
   - Measures overall correctness of predictions
   - Range: [0, 1], higher is better

2. **AUC (Area Under ROC Curve)**
   - Measures trade-off between true positive rate and false positive rate
   - Range: [0, 1], where 0.5 = random classifier, 1.0 = perfect classifier
   - Robust to class imbalance
   - Measures discriminative ability across thresholds

3. **Precision**
   - Formula: $\text{Precision} = \frac{TP}{TP + FP}$
   - Of all positive predictions, how many were actually correct
   - Important when false positives are costly

4. **Recall (Sensitivity)**
   - Formula: $\text{Recall} = \frac{TP}{TP + FN}$
   - Of all actual positives, how many did the model find
   - Important when false negatives are costly

5. **F1 Score**
   - Formula: $\text{F1} = 2 \cdot \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$
   - Harmonic mean balancing Precision and Recall
   - Useful for imbalanced datasets

6. **MCC (Matthews Correlation Coefficient)**
   - Formula: $\text{MCC} = \frac{TP \cdot TN - FP \cdot FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}$
   - Correlation coefficient between predicted and actual values
   - Handles class imbalance better than accuracy alone
   - Range: [-1, 1], where 1 = perfect, 0 = random, -1 = worst

**Legend:** TP = True Positives, TN = True Negatives, FP = False Positives, FN = False Negatives

---

## Project Structure

```
ml-assignment-2/
├── app.py                          # Streamlit web application
├── requirements.txt                # Python dependencies
├── README.md                       # This file
│
├── data/
│   └── bank-additional-full.csv   # UCI Bank Marketing Dataset
│
└── model/
    ├── train_models.py            # Training script for all 6 models
    ├── evaluate_models.py         # Evaluation metrics functions
    └── saved_models/
        ├── logistic.pkl           # Logistic Regression model
        ├── dt.pkl                 # Decision Tree model
        ├── knn.pkl                # KNN model
        ├── nb.pkl                 # Naive Bayes model
        ├── rf.pkl                 # Random Forest model
        └── xgb.pkl                # XGBoost model
```

---

## Installation & Setup

### Prerequisites
- Python 3.10+
- pip or conda

### Step 1: Clone/Navigate to Project
```bash
cd ml_assignment_2
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Place Dataset
Ensure `bank-additional-full.csv` is in the `data/` folder.

### Step 4: Train Models
```bash
python model/train_models.py
```

This will:
- Load and preprocess the dataset
- Train all 6 models
- Save trained models to `model/saved_models/`
- Display training and test accuracy for each model

---

## Running the Streamlit App

### Local Deployment
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

### Streamlit Community Cloud Deployment

1. Push project to GitHub repository
2. Go to [Streamlit Cloud](https://share.streamlit.io/)
3. Click "New app" → Select repository → Select `app.py` as main file
4. Click "Deploy"

**Important:** Ensure `bank-additional-full.csv` is in the `data/` folder in your GitHub repo.

---

## Using the Streamlit App

1. **Select a Model:** Choose from 6 trained models in the sidebar
2. **Upload CSV:** Upload a CSV file with the same format as the training data
3. **View Results:**
   - Display of 6 comprehensive metrics
   - Classification report (Precision, Recall, F1 for each class)
   - Confusion matrix visualization

## Technologies & Libraries

- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-learn, XGBoost
- **Visualization:** Matplotlib, Seaborn
- **Web Framework:** Streamlit
- **Model Persistence:** Joblib

---

## Authors & Attribution

**Lab:** BITS Lab  
**Assignment:** ML Assignment 2  
**Date:** February 2026  

---

## References

- UCI Bank Marketing Dataset: https://archive.ics.uci.edu/ml/datasets/Bank+Marketing
- Streamlit Documentation: https://docs.streamlit.io/
- Scikit-learn Metrics: https://scikit-learn.org/stable/modules/model_evaluation.html
- XGBoost Documentation: https://xgboost.readthedocs.io/

---

## License

Educational use only.
