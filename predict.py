import joblib

# Load saved model
model = joblib.load("model.joblib")
vectorizer = joblib.load("vectorizer.joblib")

# User input
news = input("Enter news article: ")

# Transform text
news_vector = vectorizer.transform([news])

# Predict
prediction = model.predict(news_vector)

print("\nPrediction:", prediction[0])

if prediction[0] == "FAKE":
    print("⚠️ This news may be misleading.")
else:
    print("✅ This news appears reliable.")
