# Import the required libraries
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

# Sample training data
texts = ["love", "great", "hate", "terrible", "amazing"]
labels = ["positive", "positive", "negative", "negative", "positive"]

# Preprocess the text data
vectorizer = CountVectorizer()
features = vectorizer.fit_transform(texts)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Train the classifier
classifier = LinearSVC()
classifier.fit(X_train, y_train)

# Predict sentiment for new words
def predict_sentiment(word):
    word_features = vectorizer.transform([word])
    prediction = classifier.predict(word_features)
    return prediction[0]

# Test the app
word_to_predict = "this is terrible..... i hate it, it should be amazing"
predicted_sentiment = predict_sentiment(word_to_predict)
print(f"Predicted sentiment: {predicted_sentiment}")
