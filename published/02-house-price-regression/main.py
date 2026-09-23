"""
California House Price Regression
---------------------------------
Predict median house value for California districts from features like
median income, house age, and location, using Gradient Boosting.
"""

from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score


def main():
    data = fetch_california_housing()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = GradientBoostingRegressor(random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print(f"MAE : {mae:.3f} (in $100,000s)")
    print(f"R^2 : {r2:.3f}\n")

    print("Feature importances:")
    for name, imp in sorted(
        zip(data.feature_names, model.feature_importances_),
        key=lambda t: t[1],
        reverse=True,
    ):
        print(f"  {name:12s} {imp:.3f}")


if __name__ == "__main__":
    main()
