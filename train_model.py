# train_model.py

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pickle

# --- Step 1: Sample Labeled Emails ---
emails = [
    "Huge SALE! Get 50% off on all items. Shop now!",
    "Your Amazon order has been shipped",
    "Meeting scheduled for 3PM today",
    "Win a free iPhone by clicking here!",
    "Don't miss this limited-time offer!",
    "Re: Follow-up on our project discussion",
    "Earn money working from home easily",
    "Your Uber receipt from your recent ride",
    "Congratulations! You've been selected for a prize",
    "Monthly report attached"
]

labels = [1, 0, 0, 1, 1, 0, 1, 0, 1, 0]  # 1 = Promotion, 0 = Normal

# --- Step 2: Vectorize Text ---
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(emails)

# --- Step 3: Train/Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2)

# --- Step 4: Train the Model -
model = LogisticRegression()
model.fit(X_train, y_train)

# --- Step 5: Evaluate Model (optional) ---
print("Model Accuracy on Test Data:", model.score(X_test, y_test))

# --- Step 6: Save Model and Vectorizer ---
with open('model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

with open('vectorizer.pkl', 'wb') as vec_file:
    pickle.dump(vectorizer, vec_file)

print("✅ Model and vectorizer saved successfully.")
