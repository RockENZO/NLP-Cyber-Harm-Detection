"""
Simple Fraud Detection Demo (No External Dependencies)
====================================================

This is a simplified version that demonstrates the concepts without requiring
external libraries. Good for understanding the approach before installing dependencies.
"""

import re
import random
from collections import Counter
import math

class SimpleFraudDetector:
    """
    Simple fraud detector using basic text analysis
    """
    
    def __init__(self):
        # Fraud indicators (keywords commonly found in scam messages)
        self.fraud_keywords = {
            'urgent', 'click', 'verify', 'suspended', 'account', 'winner', 'won',
            'claim', 'prize', 'money', 'bank', 'credit', 'card', 'fee', 'pay',
            'immediately', 'legal', 'action', 'alert', 'security', 'update',
            'confirm', 'details', 'inherit', 'million', 'earn', 'fast',
            'opportunity', 'guaranteed', 'free', 'offer', 'limited', 'time'
        }
        
        # Normal conversation indicators
        self.normal_keywords = {
            'meeting', 'lunch', 'thanks', 'help', 'project', 'schedule',
            'appointment', 'birthday', 'weather', 'restaurant', 'conference',
            'document', 'feedback', 'team', 'event', 'review'
        }
        
        self.trained = False
        self.fraud_score_threshold = 0.3
    
    def preprocess_text(self, text):
        """
        Basic text preprocessing
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation and split into words
        words = re.findall(r'\b[a-z]+\b', text)
        
        return words
    
    def calculate_fraud_score(self, text):
        """
        Calculate fraud score based on keyword presence
        """
        words = self.preprocess_text(text)
        total_words = len(words)
        
        if total_words == 0:
            return 0.0
        
        # Count fraud and normal indicators
        fraud_count = sum(1 for word in words if word in self.fraud_keywords)
        normal_count = sum(1 for word in words if word in self.normal_keywords)
        
        # Calculate scores
        fraud_ratio = fraud_count / total_words
        normal_ratio = normal_count / total_words
        
        # Simple scoring: higher fraud ratio = higher fraud score
        fraud_score = fraud_ratio * 0.8 + (fraud_count > 0) * 0.2 - normal_ratio * 0.3
        
        return max(0.0, min(1.0, fraud_score))
    
    def predict(self, text):
        """
        Predict if text is fraud or normal
        """
        score = self.calculate_fraud_score(text)
        prediction = 'fraud' if score > self.fraud_score_threshold else 'normal'
        
        return {
            'text': text,
            'prediction': prediction,
            'fraud_score': score,
            'confidence': abs(score - 0.5) * 2  # Distance from decision boundary
        }
    
    def evaluate_predictions(self, messages, true_labels):
        """
        Evaluate predictions against true labels
        """
        correct = 0
        predictions = []
        
        for message, true_label in zip(messages, true_labels):
            result = self.predict(message)
            predictions.append(result['prediction'])
            if result['prediction'] == true_label:
                correct += 1
        
        accuracy = correct / len(messages)
        
        # Calculate confusion matrix
        tp = sum(1 for i in range(len(predictions)) 
                if predictions[i] == 'fraud' and true_labels[i] == 'fraud')
        fp = sum(1 for i in range(len(predictions)) 
                if predictions[i] == 'fraud' and true_labels[i] == 'normal')
        tn = sum(1 for i in range(len(predictions)) 
                if predictions[i] == 'normal' and true_labels[i] == 'normal')
        fn = sum(1 for i in range(len(predictions)) 
                if predictions[i] == 'normal' and true_labels[i] == 'fraud')
        
        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': {'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn}
        }

def create_sample_data():
    """
    Create sample fraud and normal messages
    """
    fraud_messages = [
        "URGENT! Your account will be suspended. Click here to verify now!",
        "Congratulations! You've won $10,000! Send your bank details to claim",
        "FINAL NOTICE: Pay immediately or face legal action. Call now!",
        "Your credit card has been charged $500. Click to dispute",
        "Limited time offer! Make money fast with this opportunity",
        "ALERT: Suspicious activity detected. Verify your identity",
        "You have inherited $2 million. Send processing fee to claim",
        "Your package is delayed. Pay additional shipping fees here",
        "Bank security alert! Update your information immediately",
        "Earn $5000 weekly working from home! No experience needed",
        "WINNER! Click to claim your free iPhone now!",
        "Your account will be closed unless you verify today",
        "Guaranteed loan approval! No credit check required",
        "Act fast! Limited time investment opportunity",
        "Emergency: Verify your account to prevent suspension"
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
        "Conference call scheduled for next Tuesday",
        "Please review the attached document",
        "Team building event planned for next month",
        "Thanks for the restaurant recommendation!",
        "Can we reschedule our meeting to next week?"
    ]
    
    # Combine messages and labels
    messages = fraud_messages + normal_messages
    labels = ['fraud'] * len(fraud_messages) + ['normal'] * len(normal_messages)
    
    return messages, labels

def main():
    """
    Main demo function
    """
    print("Simple Fraud Detection Demo")
    print("=" * 40)
    
    # Create detector
    detector = SimpleFraudDetector()
    
    # Load sample data
    messages, labels = create_sample_data()
    print(f"Loaded {len(messages)} sample messages")
    print(f"Fraud messages: {labels.count('fraud')}")
    print(f"Normal messages: {labels.count('normal')}")
    
    # Evaluate on sample data
    print("\nEvaluating on sample data...")
    results = detector.evaluate_predictions(messages, labels)
    
    print(f"\nResults:")
    print(f"Accuracy: {results['accuracy']:.3f}")
    print(f"Precision: {results['precision']:.3f}")
    print(f"Recall: {results['recall']:.3f}")
    print(f"F1-Score: {results['f1_score']:.3f}")
    
    cm = results['confusion_matrix']
    print(f"\nConfusion Matrix:")
    print(f"True Positives (Fraud correctly identified): {cm['tp']}")
    print(f"False Positives (Normal classified as Fraud): {cm['fp']}")
    print(f"True Negatives (Normal correctly identified): {cm['tn']}")
    print(f"False Negatives (Fraud classified as Normal): {cm['fn']}")
    
    # Test on new messages
    print("\n" + "=" * 40)
    print("Testing on new messages:")
    print("=" * 40)
    
    test_messages = [
        "URGENT: Your bank account is compromised! Click here now!",
        "How about we grab coffee this afternoon?",
        "You've won a million dollars! Send $100 fee to claim!",
        "The quarterly report is ready for your review",
        "ALERT: Verify your credit card immediately or face suspension",
        "Thanks for helping me with the coding project yesterday"
    ]
    
    for i, message in enumerate(test_messages, 1):
        result = detector.predict(message)
        print(f"\n{i}. Message: {message}")
        print(f"   Prediction: {result['prediction'].upper()}")
        print(f"   Fraud Score: {result['fraud_score']:.3f}")
        print(f"   Confidence: {result['confidence']:.3f}")
    
    print("\n" + "=" * 40)
    print("Demo Complete!")
    print("=" * 40)
    print("\nThis simple approach shows the basic concepts.")
    print("For better performance, use the full baseline models with:")
    print("- TF-IDF vectorization")
    print("- Machine learning algorithms (SVM, Logistic Regression)")
    print("- BERT-based models for advanced NLP")
    print("\nRun 'python setup.py' to install dependencies and try advanced models.")

if __name__ == "__main__":
    main()