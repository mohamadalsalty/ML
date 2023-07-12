import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

# Load the CSV file with the comma delimiter
df = pd.read_csv('data.csv')

# Separate the code snippets and labels
X = df['code']
y = df['label']

# Convert the code snippets into vectorized features
vectorizer = TfidfVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# Train the Support Vector Machine (SVM) classifier
classifier = SVC(kernel='linear')
classifier.fit(X_vectorized, y)

# Interactive loop to input PHP code
while True:
    php_code = input('Enter PHP code snippet (or "exit" to quit): ')
    if php_code == 'exit':
        break
    else:
        code_vectorized = vectorizer.transform([php_code])
        prediction = classifier.predict(code_vectorized)
        print('Prediction:', prediction[0])
