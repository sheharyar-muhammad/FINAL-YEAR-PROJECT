import os
from sklearn.feature_extraction.text import CountVectorizer
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import joblib

# Load the logistic regression model
logistic_regression_model_path = "classification/trained_model/logistic_regression_model.pkl"
logistic_regression_model = joblib.load(logistic_regression_model_path)

# Load the CountVectorizer used during training
vectorizer_path = "classification/trained_model/vectorizer.pkl"
vectorizer = joblib.load(vectorizer_path)

# Load the BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased', num_labels=2
)

# Define weights for each model (to be determined based on validation performance)
logistic_regression_weight = 0.7
bert_weight = 0.3

# Function to preprocess input text for logistic regression


def preprocess_text_for_lr(text):
    return vectorizer.transform([text])

# Function to predict using logistic regression model


def predict_logistic_regression(text):
    features = preprocess_text_for_lr(text)
    prediction = logistic_regression_model.predict(features)
    return prediction[0]

# Function to predict using BERT-based model


def predict_bert_model(text):
    input_tensor = tokenizer(
        text, padding=True, truncation=True, return_tensors='pt'
    )
    outputs = bert_model(**input_tensor)
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1)
    prediction = torch.argmax(probabilities, dim=1).item()
    return prediction

# Function to combine predictions using weighted voting


def combine_predictions_weighted(text):
    logistic_regression_prediction = predict_logistic_regression(text)
    bert_prediction = predict_bert_model(text)

    # Convert logistic regression prediction to numerical value
    logistic_regression_prediction = 1 if logistic_regression_prediction == 'F' else 0

    # Combine predictions using weighted voting
    weighted_combined_prediction = (
        logistic_regression_weight * logistic_regression_prediction +
        bert_weight * bert_prediction
    )

    # Apply threshold (e.g., 0.5) to determine final prediction
    combined_prediction = (
        'Functional Requirement'
        if weighted_combined_prediction >= 0.5
        else 'Non-Functional Requirement'
    )

    return combined_prediction  # Return the combined prediction
