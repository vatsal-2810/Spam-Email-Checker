# spam_classifier.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. Load Dataset
# You can use SMS Spam Collection dataset: https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset
# Dataset has 2 columns: ['label', 'message']
df = pd.read_csv("spam.csv", encoding='latin-1')[['v1','v2']]
df.columns = ['label', 'message']

# Convert labels: ham -> 0, spam -> 1
df['label'] = df['label'].map({'ham':0, 'spam':1})

# 2. Split Data
X_train, X_test, y_train, y_test = train_test_split(
    df['message'], df['label'], test_size=0.2, random_state=42
)

# 3. Text Vectorization (TF-IDF)
vectorizer = TfidfVectorizer(stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 4. Train Naive Bayes Classifier
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# 5. Evaluate Model
y_pred = model.predict(X_test_tfidf)

print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n📊 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\n📋 Classification Report:\n", classification_report(y_test, y_pred))

# 6. Test on Custom Messages
sample_emails = [
    "Congratulations! You have won a $1000 Walmart gift card. Click here to claim.",
    "Hey, are we still meeting for lunch tomorrow?",
    "Your account has been suspended. Please verify your details."
]

sample_tfidf = vectorizer.transform(sample_emails)
predictions = model.predict(sample_tfidf)

for msg, pred in zip(sample_emails, predictions):
    print(f"\n📩 Email: {msg}\n🔍 Prediction: {'Spam' if pred==1 else 'Not Spam'}")
