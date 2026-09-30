import joblib


# ============================================================
# LOAD TRAINED MODEL AND TF-IDF VECTORIZER
# ============================================================

model = joblib.load("models/news_classifier.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_news(news_text):

    # Convert the news text into TF-IDF features
    text_vector = vectorizer.transform([news_text])

    # Make prediction
    prediction = model.predict(text_vector)[0]

    # Get prediction probability
    probabilities = model.predict_proba(text_vector)[0]

    fake_probability = probabilities[0] * 100
    real_probability = probabilities[1] * 100

    # Determine result
    if prediction == 0:
        result = "FAKE NEWS"
        confidence = fake_probability
    else:
        result = "REAL NEWS"
        confidence = real_probability

    return result, confidence, fake_probability, real_probability


# ============================================================
# MAIN PROGRAM
# ============================================================

print("=" * 60)
print("       AI NEWS ARTICLE CLASSIFIER")
print("=" * 60)

print("\nEnter a news article or headline.")
print("Type 'exit' to stop the program.\n")


while True:

    news_text = input("Enter news text: ").strip()

    if news_text.lower() == "exit":
        print("\nProgram closed.")
        break

    if not news_text:
        print("Please enter some news text.\n")
        continue

    result, confidence, fake_probability, real_probability = (
        predict_news(news_text)
    )

    print("\n" + "-" * 60)
    print("PREDICTION:", result)
    print(f"Confidence: {confidence:.2f}%")
    print(f"Fake probability: {fake_probability:.2f}%")
    print(f"Real probability: {real_probability:.2f}%")
    print("-" * 60 + "\n")