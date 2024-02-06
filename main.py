import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

# Load the text corpus data
response = requests.get('https://raw.githubusercontent.com/selva86/datasets/master/newsgroups.json')
data = pd.read_json(response.text)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data['content'], data['target'], test_size=0.2, random_state=42)

# Create the TF-IDF vectorizer with 2-6 character n-grams
tfidf_vectorizer = TfidfVectorizer(ngram_range=(2, 6))

# Fit the vectorizer on the training data and transform the training and testing data
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Train a Naive Bayes classifier on the TF-IDF transformed training data
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train_tfidf, y_train)

# Predict the labels for the testing data
y_pred = nb_classifier.predict(X_test_tfidf)

# Calculate the accuracy and print the classification report
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)
print(classification_report(y_test, y_pred))
