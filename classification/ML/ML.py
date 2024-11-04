import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
from transformers import BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, TensorDataset
from torch.nn import CrossEntropyLoss
import torch
import os

# Load your functional requirements dataset
fr_data = pd.read_excel('functional.xlsx')

# Load your non-functional requirements dataset
nfr_data = pd.read_excel('nonfunctional-2.xlsx')

# label
fr_data['label'] = 'F'
nfr_data['label'] = 'NFR'

# Combine the datasets into one
combined_data = pd.concat([fr_data, nfr_data], ignore_index=True)

# Shuffle the rows
shuffled_data = combined_data.sample(frac=1).reset_index(drop=True)

# Convert "fr" labels to 1 and "nfr" labels to 0
shuffled_data['label'] = shuffled_data['label'].apply(
    lambda x: 1 if x == 'F' else 0)

# Split the dataset into training and testing sets (80% training, 20% testing)
train_data, test_data = train_test_split(
    shuffled_data, test_size=0.2, random_state=42)

print("Training set:", train_data.shape)
print("Testing set:", test_data.shape)

# Load the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize and preprocess the training data
train_inputs = tokenizer(train_data['Requirements'].tolist(
), padding=True, truncation=True, return_tensors='pt')
train_labels = torch.tensor(train_data['label'].values)

# Tokenize and preprocess the testing data
test_inputs = tokenizer(test_data['Requirements'].tolist(
), padding=True, truncation=True, return_tensors='pt')
test_labels = torch.tensor(test_data['label'].values)

# Load pre-trained BERT model for sequence classification
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased', num_labels=2)

# Create a DataLoader for training
train_dataset = TensorDataset(
    train_inputs['input_ids'], train_inputs['attention_mask'], train_labels)
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
# Create a DataLoader for testing
test_dataset = TensorDataset(
    test_inputs['input_ids'], test_inputs['attention_mask'], test_labels)
test_dataloader = DataLoader(test_dataset, batch_size=8)


# Set up optimizer and learning rate scheduler
optimizer = AdamW(model.parameters(), lr=2e-5)
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=0, num_training_steps=len(train_dataloader)*3)
# Loading the pre-trained model


if 'trained_model' in os.listdir():
    model = BertForSequenceClassification.from_pretrained(
        'trained_model', num_labels=2)
    print("Pre-trained model loaded successfully.")
else:
    print("No pre-trained model found. Training a new model...")

    # Training loop
    epochs = 3
    loss_fn = CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_dataloader:
            optimizer.zero_grad()
            input_ids, attention_mask, labels = batch
            outputs = model(
                input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            scheduler.step()

        avg_loss = total_loss / len(train_dataloader)
        print(f"Average Loss for Epoch {epoch + 1}: {avg_loss}")

        # Calculate and print accuracy on testing set
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in test_dataloader:
            input_ids, attention_mask, labels = batch
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predicted_labels = torch.argmax(logits, dim=1)
            predictions.extend(predicted_labels.cpu().numpy())

    # Compute accuracy
    correct_predictions = (predictions == test_labels.numpy()).sum()
    total_predictions = len(test_labels)
    accuracy = correct_predictions / total_predictions

    # Print accuracy
    print(f"Accuracy on Testing Set: {accuracy:.2%}")

    # Save the trained model
    model.save_pretrained('trained_model')
    print(f"Model trained")


# Continue with user input and prediction code
while True:
    user_statement = input("Enter a statement or (type 'rana' to quit): ")

    if user_statement.lower() == 'rana':
        break

    # Tokenize and preprocess the user input
    user_input = tokenizer(user_statement, padding=True,
                           truncation=True, return_tensors='pt')

    # Make a prediction
    model.eval()
    with torch.no_grad():
        output = model(**user_input)
        logits = output.logits
        predicted_label = torch.argmax(logits, dim=1).item()

    # Print the prediction
    if predicted_label == 1:
        print("Functional Requirement")
    else:
        print("Non-Functional Requirement")
