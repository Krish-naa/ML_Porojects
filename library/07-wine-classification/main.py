"""
Wine Cultivar Classification
----------------------------
Predict which of three cultivars a wine came from, based on 13 chemical
measurements, using K-Nearest Neighbors.
"""

from sklearn.datasets import load_wine
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report


def main():
    data = load_wine()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))

    cv = cross_val_score(model, X_train, y_train, cv=5)
    print(f"5-fold CV accuracy: {cv.mean():.3f} (+/- {cv.std():.3f})")

    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print(f"Test accuracy    : {accuracy_score(y_test, preds):.3f}\n")
    print(classification_report(y_test, preds, target_names=data.target_names))


if __name__ == "__main__":
    main()
