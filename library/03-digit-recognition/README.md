# Handwritten Digit Recognition

Classify 8x8 grayscale images of handwritten digits (0–9) using a Support Vector Machine.

## Approach
- Dataset: `load_digits` (built into scikit-learn).
- Pipeline: standard scaling + SVM with an RBF kernel.
- Evaluate with accuracy and a confusion matrix.

## Run
```bash
pip install -r requirements.txt
python main.py
```

## Concepts
Image classification, feature scaling, SVMs, pipelines, confusion matrices.
