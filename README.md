# Bank Marketing Campaign Prediction – ML Assignment 2

---

## a. Problem Statement

The objective of this project is to predict whether a bank client will subscribe to a **term deposit** (`yes` / `no`) based on historical data from direct marketing campaigns (primarily phone calls) conducted by a Portuguese banking institution.

This is a **binary classification problem** aimed at improving marketing efficiency by:
- Identifying potential subscribers more accurately  
- Reducing unnecessary marketing calls  
- Increasing campaign success rate  

Multiple machine learning classification models are implemented and evaluated to determine the best-performing model for this task.

---

## b. Dataset Description  [1 Mark]

The dataset used is the **Bank Marketing Dataset** from the UCI Machine Learning Repository.

- **Source**: Bank Marketing Dataset  
- **URL**: https://archive.ics.uci.edu/dataset/222/bank+marketing  
- **Number of Instances**: 45,211  
- **Number of Features**: 16 input attributes + 1 target variable  
- **Target Variable**:  
  - `y` → Binary  
    - `yes`: Client subscribed to a term deposit  
    - `no`: Client did not subscribe  

### Feature Categories

**Client-related features:**
- `age` (numeric)  
- `job` (categorical)  
- `marital` (categorical)  
- `education` (categorical)  
- `default` (categorical)  
- `balance` (numeric)  
- `housing` (categorical)  
- `loan` (categorical)  

**Campaign-related features:**
- `contact` (categorical)  
- `day` (numeric)  
- `month` (categorical)  
- `duration` (numeric)  
- `campaign` (numeric)  
- `pdays` (numeric)  
- `previous` (numeric)  
- `poutcome` (categorical)  

### Dataset Characteristics
- Highly **imbalanced dataset** (~11–12% positive class)  
- No significant missing values in cleaned version  
- Meets requirement of ≥500 instances and ≥12 features  
- Suitable for binary classification modeling  

---

## c. Models Used  [6 Marks]

Six classification models were implemented and evaluated using an 80/20 train-test split.

### Evaluation Metrics Used
- Accuracy  
- AUC (Area Under ROC Curve)  
- Precision  
- Recall  
- F1 Score  
- MCC (Matthews Correlation Coefficient)  

---

## Model Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|--------------|----------|-----|-----------|--------|-----|-----|
| Logistic Regression | 0.9112 | 0.9362 | 0.6688 | 0.4196 | 0.5157 | 0.4851 |
| Decision Tree | 0.9602 | 0.9086 | 0.8119 | 0.8420 | 0.8267 | 0.8044 |
| kNN | 0.9168 | 0.9300 | 0.7217 | 0.4252 | 0.5351 | 0.5136 |
| Naive Bayes | 0.8871 | 0.8418 | 0.2222 | 0.0009 | 0.0017 | 0.0072 |
| Random Forest (Ensemble) | **0.9803** | **0.9938** | **0.9463** | **0.8744** | **0.9089** | **0.8988** |
| XGBoost (Ensemble) | 0.8813 | 0.9397 | 0.4841 | 0.8157 | 0.6076 | 0.5687 |

---

## d. Observations on Model Performance  [3 Marks]

| ML Model Name | Observation about Model Performance |
|--------------|-------------------------------------|
| Logistic Regression | Achieves good overall accuracy (91.12%) and excellent AUC (0.9362), indicating strong discriminative ability. However, recall is low (0.4196), meaning many actual subscribers are missed. Provides a strong baseline but struggles with non-linear relationships. |
| Decision Tree | Strong performance with 96.02% accuracy and balanced precision-recall. Captures non-linear patterns effectively. May overfit without pruning but performs well overall. |
| kNN | Good accuracy and AUC but low recall, similar to Logistic Regression. Sensitive to feature scaling and computationally intensive for larger datasets. |
| Naive Bayes | Very poor performance with extremely low recall and F1 score. The independence assumption is violated due to correlated features, making it unsuitable for this dataset. |
| Random Forest (Ensemble) | **Best overall performer.** Highest accuracy (98.03%), highest AUC (0.9938), and best balance between precision and recall. Effectively reduces overfitting and captures complex non-linear relationships. Recommended for deployment. |
| XGBoost (Ensemble) | Strong recall (0.8157) and high AUC (0.9397), effectively identifying positive cases. However, lower precision results in more false positives. Performs well but is outperformed by Random Forest. |

---

## Final Conclusion

Among all evaluated models, **Random Forest (Ensemble)** demonstrates the best overall performance across all evaluation metrics and is the most suitable model for predicting term deposit subscription in this dataset.
