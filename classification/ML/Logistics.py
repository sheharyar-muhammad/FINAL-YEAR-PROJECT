import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load the functional requirements dataset
functional_data = pd.read_excel('functional.xlsx')

# Load the non-functional requirements dataset
non_functional_data = pd.read_excel('nonfunctional-2.xlsx')

# Label the datasets
functional_data['label'] = 'F'
non_functional_data['label'] = 'NFR'

# Combine the datasets
combined_data = pd.concat(
    [functional_data, non_functional_data], ignore_index=True)

# Shuffle the combined dataset
shuffled_data = combined_data.sample(
    frac=1, random_state=42).reset_index(drop=True)

# Split the dataset into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(
    shuffled_data['Requirements'], shuffled_data['label'], test_size=0.2, random_state=42)


print("Unique values in training dataset:", y_train.unique())
print("Unique values in testing dataset:", y_test.unique())
print("Training set:", X_train.shape)
print("Testing set:", X_test.shape)

# Define the path to save/load the vectorizer
folder_name = "trained_model"
os.makedirs(folder_name, exist_ok=True)
vectorizer_path = os.path.join(folder_name, "vectorizer.pkl")

# Check if the vectorizer exists
if os.path.exists(vectorizer_path):
    # Load the existing vectorizer
    vectorizer = joblib.load(vectorizer_path)
    print("Vectorizer loaded from", vectorizer_path)
else:
    # Initialize and fit a new CountVectorizer
    vectorizer = CountVectorizer()
    vectorizer.fit(X_train)
    # Save the vectorizer
    joblib.dump(vectorizer, vectorizer_path)
    print("Vectorizer trained and saved in", vectorizer_path)

# Convert text data into numerical features using the loaded vectorizer
X_train_vectorized = vectorizer.transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Define the path to save/load the trained model
model_path = os.path.join(folder_name, "logistic_regression_model.pkl")

# Check if the trained model exists
if os.path.exists(model_path):
    # Load the existing trained model
    model = joblib.load(model_path)
    print("Trained model loaded from", model_path)
else:
    # Initialize and train a new logistic regression model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vectorized, y_train)
    # Save the trained model
    joblib.dump(model, model_path)
    print("New model trained and saved in", model_path)

# Make predictions on the testing set
y_pred = model.predict(X_test_vectorized)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)

# Generate classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))
