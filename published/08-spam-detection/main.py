"""
SMS Spam Detection
------------------
Classify short text messages as spam or ham (not spam) using TF-IDF
features and a Logistic Regression classifier. Uses a small built-in
dataset so it runs anywhere with no downloads.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report

# Training messages
SPAM = [
    "Congratulations! You won a free prize, claim now",
    "WINNER!! Click this link to collect your reward",
    "Free entry in a weekly competition, text WIN to 80085",
    "Urgent! Your account has been selected for a cash bonus",
    "Get cheap loans now, no credit check required",
    "You have been pre-approved for a credit card, apply today",
    "Claim your free vacation voucher before it expires",
    "Limited offer! Buy one get one free, click here now",
    "Your mobile number won 1000 dollars, reply to claim",
    "Hot singles in your area want to chat, join free",
]

HAM = [
    "Hey, are we still meeting for lunch today?",
    "Can you send me the notes from class?",
    "I'll be home in about twenty minutes",
    "Happy birthday! Hope you have a great day",
    "Don't forget to pick up milk on your way back",
    "The meeting got moved to three o'clock",
    "Thanks for helping me with the project yesterday",
    "Let me know when you reach the station",
    "Mom called, she wants you to call her back",
    "Great game last night, we should play again soon",
]

# Separate unseen messages for honest evaluation
TEST_TEXTS = [
    "You won a free gift card, click to claim now",
    "Urgent offer, apply for your cash prize today",
    "reply now to win a brand new phone for free",
    "Are you coming to the party this weekend?",
    "I left my charger at your place, can you bring it",
    "Call me when you get a chance, nothing urgent",
]
TEST_LABELS = [1, 1, 1, 0, 0, 0]  # 1 = spam, 0 = ham


def main():
    texts = SPAM + HAM
    labels = [1] * len(SPAM) + [0] * len(HAM)

    model = make_pipeline(TfidfVectorizer(), LogisticRegression(max_iter=1000))
    model.fit(texts, labels)

    preds = model.predict(TEST_TEXTS)
    print(f"Held-out accuracy: {accuracy_score(TEST_LABELS, preds):.3f}\n")
    print(classification_report(TEST_LABELS, preds, target_names=["ham", "spam"]))

    print("Sample predictions:")
    for text, pred in zip(TEST_TEXTS, preds):
        label = "spam" if pred == 1 else "ham"
        print(f"  [{label:4s}] {text}")


if __name__ == "__main__":
    main()
