# SMS Spam Detection

Classify short text messages as **spam** or **ham** (not spam) — a classic NLP task
used in email and messaging filters.

## Approach
- Data: a small built-in set of labeled spam/ham messages (no downloads).
- Pipeline: TF-IDF vectorizer + Logistic Regression.
- Train on labeled messages, then evaluate on a separate set of unseen messages.

## Run
```bash
pip install -r requirements.txt
python main.py
```

## Concepts
Natural language processing, TF-IDF, logistic regression, text classification.
