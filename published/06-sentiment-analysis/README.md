# Text Sentiment Analysis

Classify short movie-review sentences as positive or negative — a first step into NLP.

## Approach
- Data: a small built-in set of labeled sentences (no downloads).
- Pipeline: TF-IDF vectorizer + Multinomial Naive Bayes.
- Evaluate on a held-out split and run predictions on new sentences.

## Run
```bash
pip install -r requirements.txt
python main.py
```

## Concepts
Natural language processing, TF-IDF, Naive Bayes, text classification.
