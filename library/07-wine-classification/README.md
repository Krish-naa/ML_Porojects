# Wine Cultivar Classification

Predict which of three cultivars a wine came from, based on 13 chemical measurements.

## Approach
- Dataset: Wine (built into scikit-learn).
- Pipeline: standard scaling + K-Nearest Neighbors.
- Use 5-fold cross-validation, then evaluate on a held-out test set.

## Run
```bash
pip install -r requirements.txt
python main.py
```

## Concepts
Multi-class classification, KNN, cross-validation, feature scaling.
