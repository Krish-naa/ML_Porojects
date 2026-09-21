# California House Price Regression

Predict the median house value for California districts based on features such as
median income, average rooms, house age, and geographic location.

## Approach
- Dataset: California Housing (built into scikit-learn, fetched once and cached).
- Model: Gradient Boosting Regressor.
- Evaluate with MAE and R², and print feature importances.

## Run
```bash
pip install -r requirements.txt
python main.py
```

## Concepts
Supervised regression, gradient boosting, error metrics (MAE, R²).
