# Task-04: Sentiment Analysis on Social Media Data
# Prodigy Infotech Internship

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# -----------------------------
# 1. Load Dataset
# -----------------------------
file_path = r"T:\GitHub\Financial report\Prodigy_Infotech\Sentiment Pattern in Social Media\twitter_sentiments.csv"
data = pd.read_csv(file_path)

# Rename columns properly
data.columns = ["id", "topic", "sentiment", "content"]

print("✅ First 5 rows of cleaned data:")
print(data.head())

# -----------------------------
# 2. Data Cleaning
# -----------------------------
# Drop rows with missing text/sentiment
data = data.dropna(subset=['content', 'sentiment'])

print("\n✅ Class distribution:")
print(data['sentiment'].value_counts())

# -----------------------------
# 3. Feature Extraction (TF-IDF)
# -----------------------------
X = data['content']
y = data['sentiment']

vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_tfidf = vectorizer.fit_transform(X)

# -----------------------------
# 4. Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------
# 5. Train Model
# -----------------------------
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# -----------------------------
# 6. Evaluate Model
# -----------------------------
y_pred = model.predict(X_test)

print("\n✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n✅ Classification Report:\n", classification_report(y_test, y_pred))
print("\n✅ Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# -----------------------------
# 7. Visualization
# -----------------------------
sentiment_counts = data['sentiment'].value_counts()

plt.figure(figsize=(6, 6))
plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=90)
plt.title("Sentiment Distribution in Social Media Data")
plt.show()
