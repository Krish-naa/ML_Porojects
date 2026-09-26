# Breast Cancer Detection

Predict whether a tumor is malignant or benign from 30 cell-nucleus measurements —
a binary medical classification task.

## Approach
- Dataset: Wisconsin Breast Cancer (built into scikit-learn).
- Pipeline: standard scaling + Logistic Regression.
- Evaluate with accuracy, ROC AUC, and a full classification report.

## Run
```bash
pip install -r requirements.txt
python main.py
```

## Concepts
Binary classification, logistic regression, ROC AUC, precision/recall.
