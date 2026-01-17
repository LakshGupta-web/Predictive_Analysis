# Program 1: Impact of Sampling Techniques on Imbalanced Datasets

## Objective
The objective of this assignment is to understand the importance of sampling techniques in handling highly imbalanced datasets and to analyze how different sampling strategies affect the performance of various machine learning models.

---

## Problem Statement
In real-world classification problems, datasets are often imbalanced, where one class significantly outnumbers the other. Such imbalance can lead to biased models that perform poorly on the minority class.

In this assignment, a highly imbalanced **credit card fraud detection dataset** is used. The task is to:
- Balance the dataset using different sampling techniques
- Apply multiple machine learning models
- Evaluate how sampling strategies influence model accuracy

---

## Dataset
- **Name:** Creditcard_data.csv  
- **Source:**  
  https://github.com/AnjulaMehto/Sampling_Assignment/blob/main/Creditcard_data.csv  
- **Target Variable:** `Class`  
  - `0` → Non-Fraud  
  - `1` → Fraud  

The dataset is highly imbalanced, with fraud cases being significantly fewer than non-fraud cases.

---

## Sampling Techniques Used

| Sampling ID | Technique Description |
|------------|----------------------|
| Sampling1 | Random Under Sampling |
| Sampling2 | Random Over Sampling |
| Sampling3 | SMOTE (Synthetic Minority Oversampling Technique) |
| Sampling4 | Tomek Links |
| Sampling5 | SMOTE + ENN (Hybrid Sampling) |

Each sampling technique was applied **only on the training dataset** to maintain correct machine learning practice.

---

## Machine Learning Models Used

| Model ID | Algorithm |
|--------|----------|
| M1 | Logistic Regression |
| M2 | K-Nearest Neighbors |
| M3 | Support Vector Machine |
| M4 | Random Forest Classifier |
| M5 | Gradient Boosting Classifier |

---

## Experimental Setup
- Data was split using **stratified train-test split**
- Features were standardized using `StandardScaler`
- Accuracy was used as the evaluation metric
- Sampling techniques were applied to the training data only
- The test dataset was kept untouched

---

## Accuracy Results

| Model | Sampling1 | Sampling2 | Sampling3 | Sampling4 | Sampling5 |
|------|----------|----------|----------|----------|----------|
| M1 (LogReg) | 58.71 | 90.97 | 91.61 | 98.71 | 90.32 |
| M2 (KNN) | 69.03 | 96.77 | 92.26 | 98.71 | 91.61 |
| M3 (SVM) | 52.26 | 95.48 | 96.13 | 98.71 | 95.48 |
| M4 (RF) | 56.77 | 99.35 | 98.71 | 98.71 | 98.71 |
| M5 (GB) | 35.48 | 99.35 | 98.71 | 98.71 | 98.06 |

---

## Best Sampling Technique per Model

| Model | Best Sampling Technique | Accuracy (%) |
|------|------------------------|--------------|
| M1 | Sampling4 | 98.71 |
| M2 | Sampling4 | 98.71 |
| M3 | Sampling4 | 98.71 |
| M4 | Sampling2 | 99.35 |
| M5 | Sampling1 | 99.35 |

---

## Observations and Discussion
- No single sampling technique performs best for all models.
- Undersampling worked exceptionally well for SVM and ensemble-based models.
- SMOTE improved performance for distance-based models like KNN.
- Hybrid sampling methods helped linear models handle imbalance better.
- Sampling plays a crucial role in improving performance on imbalanced datasets.

---

## Repository Structure
Sampling_Assignment/
│
├── Creditcard_data.csv
├── sampling_assignment.ipynb
├── accuracy_results.csv
├── README.md


---

## Conclusion
This experiment demonstrates that handling class imbalance is essential for building reliable machine learning models. The choice of sampling technique should be based on the nature of the model and the dataset, as different models respond differently to various sampling strategies.

---

## Author
**Laksh Gupta**

---

## Tools & Libraries Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Imbalanced-learn (imblearn)

