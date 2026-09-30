from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load trained model and TF-IDF vectorizer
model = joblib.load("models/news_classifier.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")


@app.route("/", methods=["GET", "POST"])
def home():

    result = None
    confidence = None
    fake_probability = None
    real_probability = None
    confidence_level = None
    error_message = None
    news_text = ""

    if request.method == "POST":

        news_text = request.form.get("news_text", "").strip()

        # Validate input
        if not news_text:

            error_message = "Please enter a news headline or article."

        elif len(news_text) < 20:

            error_message = (
                "Please enter a meaningful news headline or article "
                "with at least 20 characters."
            )

        else:

            # Convert text into TF-IDF features
            text_vector = vectorizer.transform([news_text])

            # Make prediction
            prediction = model.predict(text_vector)[0]

            # Get probabilities
            probabilities = model.predict_proba(text_vector)[0]

            fake_probability = probabilities[0] * 100
            real_probability = probabilities[1] * 100

            # Determine prediction
            if prediction == 0:

                result = "FAKE NEWS"
                confidence = fake_probability

            else:

                result = "REAL NEWS"
                confidence = real_probability

            # Confidence interpretation
            if confidence >= 80:

                confidence_level = "High Confidence"

            elif confidence >= 65:

                confidence_level = "Moderate Confidence"

            else:

                confidence_level = "Low Confidence"


    return render_template(
        "index.html",
        result=result,
        confidence=confidence,
        fake_probability=fake_probability,
        real_probability=real_probability,
        confidence_level=confidence_level,
        error_message=error_message,
        news_text=news_text
    )


if __name__ == "__main__":

    print("=" * 60)
    print("       AI NEWS ARTICLE CLASSIFIER")
    print("=" * 60)

    print("\nStarting Flask web application...")

    print("Open this URL in your browser:")
    print("http://127.0.0.1:5000")

    print("\nPress CTRL+C to stop the server.\n")

    app.run(
        debug=True
    )