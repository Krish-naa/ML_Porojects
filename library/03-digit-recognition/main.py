"""
Handwritten Digit Recognition
------------------------------
Classify 8x8 grayscale images of handwritten digits (0-9) using a
Support Vector Machine.
"""

from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, confusion_matrix


def main():
    digits = load_digits()
    X, y = digits.data, digits.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    model = make_pipeline(StandardScaler(), SVC(kernel="rbf", gamma="scale"))
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print(f"Accuracy: {acc:.3f}\n")
    print("Confusion matrix:")
    print(confusion_matrix(y_test, preds))


if __name__ == "__main__":
    main()
