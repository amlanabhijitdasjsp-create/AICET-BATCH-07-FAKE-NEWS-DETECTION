import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv("dataset/news.csv")

# Count labels
counts = data["label"].value_counts()

# Plot
plt.figure()
counts.plot(kind="bar")

plt.title("Fake vs Real News Distribution")
plt.xlabel("News Type")
plt.ylabel("Count")

plt.show()
