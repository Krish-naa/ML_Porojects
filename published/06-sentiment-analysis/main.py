"""
Text Sentiment Analysis
-----------------------
Classify short movie-review-style sentences as positive or negative using
TF-IDF features and Naive Bayes. Uses a small built-in dataset so it runs
anywhere with no downloads.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report

# Training data
POSITIVE = [
    "I loved this movie, it was fantastic",
    "An absolute masterpiece, brilliant acting",
    "Wonderful story and great direction",
    "Amazing film, I enjoyed every minute",
    "Superb performances and a touching plot",
    "The best movie I have seen this year",
    "Delightful and heartwarming, highly recommend",
    "Excellent script and beautiful cinematography",
    "A fun, thrilling and satisfying experience",
    "Truly inspiring and emotionally powerful",
]

NEGATIVE = [
    "I hated this movie, it was terrible",
    "A boring and dull waste of time",
    "Awful acting and a weak plot",
    "The worst film I have ever seen",
    "Disappointing and painfully slow",
    "Poorly written and badly directed",
    "A complete mess, I regret watching it",
    "Uninspired, forgettable and tedious",
    "Bad pacing and a predictable ending",
    "Terrible dialogue and flat characters",
]

# A separate held-out set of unseen sentences to evaluate honestly
TEST_TEXTS = [
    "what a great and enjoyable film",
    "a brilliant and wonderful experience",
    "the acting was superb and inspiring",
    "this was a dull and awful movie",
    "a boring, terrible waste of time",
    "poorly directed and painfully slow",
]
TEST_LABELS = [1, 1, 1, 0, 0, 0]


def main():
    texts = POSITIVE + NEGATIVE
    labels = [1] * len(POSITIVE) + [0] * len(NEGATIVE)

    model = make_pipeline(TfidfVectorizer(), MultinomialNB())
    model.fit(texts, labels)

    preds = model.predict(TEST_TEXTS)
    print(f"Held-out accuracy: {accuracy_score(TEST_LABELS, preds):.3f}\n")
    print(classification_report(TEST_LABELS, preds, target_names=["negative", "positive"]))

    print("Sample predictions:")
    for text, pred in zip(TEST_TEXTS, preds):
        sentiment = "positive" if pred == 1 else "negative"
        print(f"  '{text}' -> {sentiment}")


if __name__ == "__main__":
    main()
