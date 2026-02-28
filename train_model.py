import pandas as pd
import re
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

# Load data
data = pd.read_csv("d:/fn/DATASET.csv")

# Remove empty rows
data = data.dropna(subset=["text", "label"])
data = data.reset_index(drop=True)

# Clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z ]', '', text)
    return text

data["text"] = data["text"].apply(clean_text)

# Remove short samples
data = data[data["text"].str.len() > 20]

X = data["text"]
y = data["label"]

# Vectorization
vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1,3),
    max_df=0.8,
    min_df=2,
    sublinear_tf=True
)

X_vectorized = vectorizer.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Model
model = LinearSVC()
model.fit(X_train, y_train)

# Accuracy
pred = model.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, pred))

# Save
joblib.dump(model, "model.joblib")
joblib.dump(vectorizer, "vectorizer.joblib")
