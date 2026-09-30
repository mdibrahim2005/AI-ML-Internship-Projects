import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ============================================================
# 1. LOAD DATASET
# ============================================================

fake = pd.read_csv("data/Fake.csv")
true = pd.read_csv("data/True.csv")


# ============================================================
# 2. ADD LABELS
# ============================================================

# 0 = Fake News
# 1 = Real News

fake["label"] = 0
true["label"] = 1


# ============================================================
# 3. COMBINE BOTH DATASETS
# ============================================================

data = pd.concat([fake, true], ignore_index=True)


# ============================================================
# 4. CREATE ONE TEXT COLUMN
# ============================================================

data["content"] = (
    data["title"].fillna("")
    + " "
    + data["text"].fillna("")
)


# ============================================================
# 5. REMOVE EMPTY ARTICLES
# ============================================================

data = data[data["content"].str.strip() != ""]


# ============================================================
# 6. SHUFFLE DATA
# ============================================================

data = data.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)


# ============================================================
# 7. SELECT INPUT AND TARGET
# ============================================================

X = data["content"]
y = data["label"]


# ============================================================
# 8. SPLIT DATA INTO TRAINING AND TESTING
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ============================================================
# 9. CONVERT TEXT INTO NUMERICAL FEATURES
# ============================================================

print("Creating TF-IDF features...")

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_df=0.7
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


print("TF-IDF preparation completed!")
print("Training shape:", X_train_tfidf.shape)
print("Testing shape:", X_test_tfidf.shape)


# ============================================================
# 10. CREATE MACHINE LEARNING MODEL
# ============================================================

print("\nTraining Logistic Regression model...")

model = LogisticRegression(
    max_iter=1000,
    random_state=42
)


# ============================================================
# 11. TRAIN THE MODEL
# ============================================================

model.fit(X_train_tfidf, y_train)

print("Model training completed!")


# ============================================================
# 12. MAKE PREDICTIONS
# ============================================================

y_pred = model.predict(X_test_tfidf)


# ============================================================
# 13. CALCULATE ACCURACY
# ============================================================

accuracy = accuracy_score(y_test, y_pred)


# ============================================================
# 14. DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 60)
print("MODEL RESULTS")
print("=" * 60)

print(f"\nAccuracy: {accuracy * 100:.2f}%")


# ============================================================
# 15. CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Fake News", "Real News"]
    )
)


# ============================================================
# 16. CONFUSION MATRIX
# ============================================================

print("Confusion Matrix:")

cm = confusion_matrix(y_test, y_pred)

print(cm)


# ============================================================
# 17. SAVE TRAINED MODEL
# ============================================================

joblib.dump(
    model,
    "models/news_classifier.pkl"
)


# ============================================================
# 18. SAVE TF-IDF VECTORIZER
# ============================================================

joblib.dump(
    vectorizer,
    "models/tfidf_vectorizer.pkl"
)


# ============================================================
# 19. FINAL MESSAGE
# ============================================================

print("\n" + "=" * 60)
print("MODEL SAVED SUCCESSFULLY!")
print("=" * 60)

print("\nSaved files:")
print("1. models/news_classifier.pkl")
print("2. models/tfidf_vectorizer.pkl")

print("\nProject Step 8 completed successfully!")