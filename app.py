
from flask import Flask, render_template, request
from pathlib import Path
import joblib

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent

# Load the trained AI model and text vectorizer
model = joblib.load(BASE_DIR / "email_fraud_model.joblib")
vectorizer = joblib.load(BASE_DIR / "email_vectorizer.joblib")


def analyze_email(subject, sender, body):
    email_text = (
        f"Subject: {subject}\n"
        f"Sender: {sender}\n"
        f"Message: {body}"
    )

    # Convert email text into numerical features
    features = vectorizer.transform([email_text])

    # Predict email category
    prediction = model.predict(features)[0]

    # Calculate model's fraud score
    probabilities = model.predict_proba(features)[0]
    fraud_index = list(model.classes_).index("fraud")
    score = round(float(probabilities[fraud_index]) * 100)

    # Find some common warning signs
    text = email_text.lower()
    warnings = []

    suspicious_phrases = [
        "verify your account",
        "account will be suspended",
        "send your otp",
        "bank details",
        "won a prize",
        "claim your reward",
        "click here",
        "immediately",
        "gift cards",
    ]

    for phrase in suspicious_phrases:
        if phrase in text:
            warnings.append(phrase)

    # Decide the displayed result
    if prediction == "fraud" or len(warnings) >= 2:
        verdict = "Suspicious Email"
        level = "high"
        advice = (
            "Do not click links or share passwords, OTPs, "
            "or banking details. Verify the sender independently."
        )
    elif len(warnings) == 1 or score >= 40:
        verdict = "Use Caution"
        level = "medium"
        advice = (
            "Check the sender and links carefully before "
            "taking any action."
        )
    else:
        verdict = "Likely Legitimate"
        level = "low"
        advice = (
            "No strong warning was identified by this demo. "
            "Still verify unexpected requests."
        )

    return {
        "verdict": verdict,
        "level": level,
        "score": score,
        "warnings": warnings,
        "advice": advice,
    }


@app.route("/", methods=["GET", "POST"])
def home():
    result = None

    if request.method == "POST":
        subject = request.form.get("subject", "").strip()
        sender = request.form.get("sender", "").strip()
        body = request.form.get("body", "").strip()

        if subject or sender or body:
            result = analyze_email(subject, sender, body)
        else:
            result = {
                "error": "Please enter some email information."
            }

    return render_template("index.html", result=result)


@app.route("/health")
def health():
    return {"status": "ok"}



if __name__ == "__main__":
    import os
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=False
    )

