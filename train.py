import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

# -------------------------
# 1. Sample dataset
# -------------------------
data = {
    "text": [
        "I love this product",
        "This is amazing",
        "Very good experience",
        "I am happy with this",
        "Worst purchase ever",
        "I hate it",
        "Very bad quality",
        "Not worth the money"
    ],
    "label": [1, 1, 1, 1, 0, 0, 0, 0]  # 1 = Positive, 0 = Negative
}

df = pd.DataFrame(data)

# -------------------------
# 2. Train/Test Split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42
)

# -------------------------
# 3. Create Pipeline
# -------------------------
model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("clf", LogisticRegression())
])

# -------------------------
# 4. Train Model
# -------------------------
model.fit(X_train, y_train)

# -------------------------
# 5. Evaluate Model
# -------------------------
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nReport:\n", classification_report(y_test, y_pred))

# -------------------------
# 6. Save Model
# -------------------------
joblib.dump(model, "sentiment_model.pkl")
print("\nModel saved as sentiment_model.pkl")