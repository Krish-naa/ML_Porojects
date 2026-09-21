"""
Breast Cancer Detection
-----------------------
Binary classification: predict whether a tumor is malignant or benign from
cell-nucleus measurements, using Logistic Regression.
"""

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report


def main():
    data = load_breast_cancer()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    print(f"Accuracy : {accuracy_score(y_test, preds):.3f}")
    print(f"ROC AUC  : {roc_auc_score(y_test, proba):.3f}\n")
    print(classification_report(y_test, preds, target_names=data.target_names))


if __name__ == "__main__":
    main()
