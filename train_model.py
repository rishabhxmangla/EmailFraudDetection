
"""Train a simple AI model to classify sample emails."""

from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

BASE_DIR = Path(__file__).resolve().parent

emails = [
    "Your meeting is scheduled for tomorrow",
    "Your monthly bank statement is ready",
    "Thank you for your order receipt",
    "Your class timetable has been updated",
    "Please find the project report attached",
    "Your appointment reminder is ready",
    "URGENT verify your account immediately",
    "Your account will be suspended click here",
    "Congratulations you won a prize claim now",
    "Send your OTP and bank details immediately",
    "Pay a processing fee to receive your reward",
    "Confirm your password to unlock your account",
    "Buy gift cards and send the codes urgently",
    "Click this link to claim your lottery money",
]

labels = [
    "legitimate",
    "legitimate",
    "legitimate",
    "legitimate",
    "legitimate",
    "legitimate",
    "fraud",
    "fraud",
    "fraud",
    "fraud",
    "fraud",
    "fraud",
    "fraud",
    "fraud",
]

vectorizer = TfidfVectorizer(ngram_range=(1, 2))
X = vectorizer.fit_transform(emails)

model = LogisticRegression(max_iter=1000)
model.fit(X, labels)

joblib.dump(model, BASE_DIR / "email_fraud_model.joblib")
joblib.dump(vectorizer, BASE_DIR / "email_vectorizer.joblib")

print("AI model trained successfully!")
print("Model and vectorizer saved.")
