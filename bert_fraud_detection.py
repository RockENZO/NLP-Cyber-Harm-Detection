"""
BERT-Based Fraud Detection Baseline
==================================

This script implements a BERT-based classifier for fraud/scam detection,
following the recommendations from the project analysis.

Features:
- Fine-tuned BERT for text classification
- Handles class imbalance with weighted loss
- Evaluation metrics focused on F1 and recall
- Easy integration with the traditional baseline models
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizer, BertForSequenceClassification,
    AdamW, get_linear_schedule_with_warmup
)
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class FraudDataset(Dataset):
    """
    Custom dataset for fraud detection with BERT tokenization
    """
    
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts.iloc[idx])
        label = self.labels.iloc[idx]
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class BERTFraudClassifier:
    """
    BERT-based fraud detection classifier
    """
    
    def __init__(self, model_name='bert-base-uncased', num_classes=2, max_length=128):
        self.model_name = model_name
        self.num_classes = num_classes
        self.max_length = max_length
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize tokenizer and model
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_classes
        ).to(self.device)
        
        print(f"Using device: {self.device}")
        print(f"Model: {model_name}")
    
    def prepare_data(self, df, text_column='message', label_column='label'):
        """
        Prepare data for training
        """
        # Encode labels
        label_map = {'normal': 0, 'fraud': 1}
        df['label_encoded'] = df[label_column].map(label_map)
        
        # Split data
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            df[text_column], df['label_encoded'],
            test_size=0.2, random_state=42, stratify=df['label_encoded']
        )
        
        # Create datasets
        train_dataset = FraudDataset(
            train_texts, train_labels, self.tokenizer, self.max_length
        )
        val_dataset = FraudDataset(
            val_texts, val_labels, self.tokenizer, self.max_length
        )
        
        return train_dataset, val_dataset
    
    def create_data_loader(self, dataset, batch_size=16, shuffle=True):
        """
        Create DataLoader for training/validation
        """
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    def train(self, train_loader, val_loader, epochs=3, learning_rate=2e-5):
        """
        Train the BERT model
        """
        # Calculate class weights for imbalanced data
        class_counts = [0, 0]
        for batch in train_loader:
            labels = batch['labels']
            for label in labels:
                class_counts[label.item()] += 1
        
        total_samples = sum(class_counts)
        class_weights = [total_samples / (2 * count) for count in class_counts]
        class_weights = torch.tensor(class_weights, dtype=torch.float).to(self.device)
        
        # Loss function with class weights
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        # Optimizer and scheduler
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )
        
        # Training loop
        self.model.train()
        train_losses = []
        val_accuracies = []
        
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            print("-" * 30)
            
            total_loss = 0
            progress_bar = tqdm(train_loader, desc="Training")
            
            for batch in progress_bar:
                # Move to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                optimizer.zero_grad()
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                loss = criterion(outputs.logits, labels)
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                total_loss += loss.item()
                progress_bar.set_postfix({'loss': loss.item()})
            
            avg_loss = total_loss / len(train_loader)
            train_losses.append(avg_loss)
            
            # Validation
            val_accuracy = self.evaluate(val_loader, verbose=False)
            val_accuracies.append(val_accuracy)
            
            print(f"Average Loss: {avg_loss:.4f}")
            print(f"Validation Accuracy: {val_accuracy:.4f}")
            print()
        
        return train_losses, val_accuracies
    
    def evaluate(self, data_loader, verbose=True):
        """
        Evaluate the model
        """
        self.model.eval()
        predictions = []
        true_labels = []
        probabilities = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                # Get predictions
                logits = outputs.logits
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)
                
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
                probabilities.extend(probs.cpu().numpy())
        
        # Calculate metrics
        accuracy = np.mean(np.array(predictions) == np.array(true_labels))
        
        if verbose:
            print("\n=== BERT Model Evaluation ===")
            print(f"Accuracy: {accuracy:.4f}")
            
            # Classification report
            report = classification_report(
                true_labels, predictions,
                target_names=['Normal', 'Fraud'],
                output_dict=True
            )
            
            print("\nClassification Report:")
            print(classification_report(
                true_labels, predictions,
                target_names=['Normal', 'Fraud']
            ))
            
            # AUC Score
            probs_fraud = [prob[1] for prob in probabilities]
            auc = roc_auc_score(true_labels, probs_fraud)
            print(f"AUC Score: {auc:.4f}")
            
            # Confusion Matrix
            cm = confusion_matrix(true_labels, predictions)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=['Normal', 'Fraud'],
                       yticklabels=['Normal', 'Fraud'])
            plt.title('BERT Model - Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.show()
            
            return {
                'accuracy': accuracy,
                'classification_report': report,
                'auc_score': auc,
                'confusion_matrix': cm
            }
        
        return accuracy
    
    def predict(self, text):
        """
        Predict if a single text is fraud or not
        """
        self.model.eval()
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            prediction = torch.argmax(logits, dim=1)
        
        label_map = {0: 'normal', 1: 'fraud'}
        predicted_label = label_map[prediction.item()]
        confidence = probs[0][prediction.item()].item()
        
        return {
            'text': text,
            'prediction': predicted_label,
            'confidence': confidence,
            'probabilities': {
                'normal': probs[0][0].item(),
                'fraud': probs[0][1].item()
            }
        }
    
    def save_model(self, path):
        """
        Save the trained model
        """
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f"Model saved to {path}")
    
    def load_model(self, path):
        """
        Load a trained model
        """
        self.model = BertForSequenceClassification.from_pretrained(path).to(self.device)
        self.tokenizer = BertTokenizer.from_pretrained(path)
        print(f"Model loaded from {path}")

def create_sample_data():
    """
    Create sample data for BERT training (same as baseline)
    """
    fraud_messages = [
        "URGENT! Your account will be suspended. Click here to verify: suspicious-link.com",
        "Congratulations! You've won $10,000! Send your bank details to claim",
        "FINAL NOTICE: Pay immediately or face legal action. Call now!",
        "Your credit card has been charged $500. If this wasn't you, click here",
        "Limited time offer! Make money fast with this amazing opportunity",
        "ALERT: Suspicious activity detected. Verify your identity now",
        "You have inherited $2 million. Send processing fee to claim",
        "Your package is delayed. Pay additional shipping fees here",
        "Bank security alert! Update your information immediately",
        "Earn $5000 weekly working from home! No experience needed",
        "Phishing attempt: Update your password for security",
        "Fake invoice: Payment overdue for services not rendered",
        "Tech support scam: Your computer is infected, call now",
        "Romance scam: I need money for emergency travel",
        "Investment fraud: Guaranteed 500% returns in 30 days"
    ]
    
    normal_messages = [
        "Hey, are we still meeting for lunch tomorrow?",
        "Thanks for your help with the project presentation",
        "The meeting has been rescheduled to 3 PM",
        "Happy birthday! Hope you have a wonderful day",
        "Can you please send me the quarterly report?",
        "The weather is great today, perfect for a walk",
        "Reminder: Your appointment is scheduled for Friday",
        "Great job on the presentation yesterday!",
        "Let me know if you need any assistance",
        "The new restaurant downtown has excellent reviews",
        "Conference call scheduled for next Tuesday at 2 PM",
        "Please review the attached document and provide feedback",
        "Team building event planned for next month",
        "Thanks for the restaurant recommendation, it was great!",
        "Can we reschedule our meeting to next week?"
    ]
    
    # Create more balanced dataset for BERT training
    messages = fraud_messages + normal_messages
    labels = ['fraud'] * len(fraud_messages) + ['normal'] * len(normal_messages)
    
    # Duplicate data to have more samples for training
    messages = messages * 3
    labels = labels * 3
    
    return pd.DataFrame({
        'message': messages,
        'label': labels
    })

def main():
    """
    Main function to run BERT baseline experiment
    """
    print("=== BERT Fraud Detection Baseline ===\n")
    
    # Check for GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if device.type == 'cpu':
        print("Warning: Training on CPU will be slow. Consider using Google Colab for GPU access.\n")
    
    # Create sample data
    print("1. Creating sample dataset...")
    df = create_sample_data()
    print(f"Dataset size: {len(df)} samples")
    print(f"Label distribution:\n{df['label'].value_counts()}\n")
    
    # Initialize BERT classifier
    print("2. Initializing BERT classifier...")
    classifier = BERTFraudClassifier()
    
    # Prepare data
    print("3. Preparing data...")
    train_dataset, val_dataset = classifier.prepare_data(df)
    
    train_loader = classifier.create_data_loader(train_dataset, batch_size=8)
    val_loader = classifier.create_data_loader(val_dataset, batch_size=8, shuffle=False)
    
    # Train model
    print("4. Training BERT model...")
    print("Note: This may take several minutes, especially on CPU\n")
    
    train_losses, val_accuracies = classifier.train(
        train_loader, val_loader, epochs=2, learning_rate=2e-5
    )
    
    # Evaluate model
    print("5. Evaluating model...")
    results = classifier.evaluate(val_loader)
    
    # Test predictions
    print("\n6. Testing predictions on new samples...")
    test_messages = [
        "URGENT: Your account is suspended. Click here to restore access!",
        "Hey, how was your weekend? Want to grab coffee tomorrow?",
        "You've won a lottery! Send $100 processing fee to claim $50,000!",
        "Meeting reminder: Project review scheduled for 2 PM today"
    ]
    
    for message in test_messages:
        result = classifier.predict(message)
        print(f"\nMessage: {message[:50]}...")
        print(f"Prediction: {result['prediction'].upper()}")
        print(f"Confidence: {result['confidence']:.3f}")
    
    # Plot training progress
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.plot(train_losses)
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    
    ax2.plot(val_accuracies)
    ax2.set_title('Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    
    plt.tight_layout()
    plt.show()
    
    print("\n=== BERT Experiment Complete ===")
    print("Next steps:")
    print("1. Use larger, real datasets for better performance")
    print("2. Experiment with different BERT variants (RoBERTa, DistilBERT)")
    print("3. Implement few-shot learning capabilities")
    print("4. Add ensemble methods combining BERT with traditional models")

if __name__ == "__main__":
    main()