"""
Iris Flower Classification
---------------------------
Classic beginner ML project: predict the species of an iris flower from
its sepal/petal measurements using a Random Forest classifier.
"""

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


def main():
    data = load_iris()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print(f"Accuracy: {acc:.3f}\n")
    print(classification_report(y_test, preds, target_names=data.target_names))

    # Feature importance
    print("Feature importances:")
    for name, imp in sorted(
        zip(data.feature_names, model.feature_importances_),
        key=lambda t: t[1],
        reverse=True,
    ):
        print(f"  {name:20s} {imp:.3f}")


if __name__ == "__main__":
    main()
