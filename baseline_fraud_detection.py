"""
NLP Fraud/Scam Detection Baseline Models
========================================

This script implements multiple baseline models for fraud/scam detection as recommended
in the project analysis. Includes traditional ML and BERT-based approaches.

Models implemented:
1. TF-IDF + Logistic Regression
2. TF-IDF + SVM
3. BERT-based classifier (simplified)

Dataset: Uses SMS Spam Collection as a starting point (can be extended)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import LabelEncoder
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

class FraudDetectionBaseline:
    """
    Baseline fraud detection system with multiple models
    """
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.label_encoder = LabelEncoder()
        self.models = {}
        self.results = {}
        
    def load_sample_data(self):
        """
        Create sample fraud/scam data for demonstration
        In practice, replace with actual datasets like SMS Spam Collection
        """
        # Sample data - replace with real dataset
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
            "Earn $5000 weekly working from home! No experience needed"
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
            "The new restaurant downtown has excellent reviews"
        ]
        
        # Create DataFrame
        messages = fraud_messages + normal_messages
        labels = ['fraud'] * len(fraud_messages) + ['normal'] * len(normal_messages)
        
        # Add more diverse examples
        additional_fraud = [
            "Phishing attempt: Update your password for security",
            "Fake invoice: Payment overdue for services not rendered",
            "Tech support scam: Your computer is infected, call now",
            "Romance scam: I need money for emergency travel",
            "Investment fraud: Guaranteed 500% returns in 30 days"
        ]
        
        additional_normal = [
            "Conference call scheduled for next Tuesday at 2 PM",
            "Please review the attached document and provide feedback",
            "Team building event planned for next month",
            "Thanks for the restaurant recommendation, it was great!",
            "Can we reschedule our meeting to next week?"
        ]
        
        messages.extend(additional_fraud + additional_normal)
        labels.extend(['fraud'] * len(additional_fraud) + ['normal'] * len(additional_normal))
        
        self.data = pd.DataFrame({
            'message': messages,
            'label': labels
        })
        
        print(f"Dataset created with {len(self.data)} samples")
        print(f"Label distribution:\n{self.data['label'].value_counts()}")
        
        return self.data
    
    def preprocess_text(self, text):
        """
        Clean and preprocess text data
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Tokenize and remove stopwords
        stop_words = set(stopwords.words('english'))
        words = text.split()
        words = [word for word in words if word not in stop_words and len(word) > 2]
        
        # Lemmatization
        lemmatizer = WordNetLemmatizer()
        words = [lemmatizer.lemmatize(word) for word in words]
        
        return ' '.join(words)
    
    def prepare_data(self):
        """
        Preprocess data and create train/test splits
        """
        # Preprocess messages
        self.data['processed_message'] = self.data['message'].apply(self.preprocess_text)
        
        # Encode labels
        self.data['label_encoded'] = self.label_encoder.fit_transform(self.data['label'])
        
        # Split data
        X = self.data['processed_message']
        y = self.data['label_encoded']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Vectorize text
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        X_test_tfidf = self.vectorizer.transform(X_test)
        
        return X_train_tfidf, X_test_tfidf, y_train, y_test
    
    def train_logistic_regression(self, X_train, y_train):
        """
        Train Logistic Regression baseline
        """
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train, y_train)
        self.models['logistic_regression'] = model
        return model
    
    def train_svm(self, X_train, y_train):
        """
        Train SVM baseline
        """
        model = SVC(kernel='linear', probability=True, random_state=42)
        model.fit(X_train, y_train)
        self.models['svm'] = model
        return model
    
    def evaluate_model(self, model, X_test, y_test, model_name):
        """
        Evaluate model performance
        """
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Metrics
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        # Store results
        self.results[model_name] = {
            'classification_report': report,
            'confusion_matrix': cm,
            'auc_score': auc,
            'accuracy': report['accuracy'],
            'f1_score': report['weighted avg']['f1-score']
        }
        
        return self.results[model_name]
    
    def plot_results(self):
        """
        Visualize model performance
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Model comparison
        models = list(self.results.keys())
        metrics = ['accuracy', 'f1_score', 'auc_score']
        
        for i, metric in enumerate(metrics):
            ax = axes[0, 0] if i == 0 else axes[0, 1] if i == 1 else axes[1, 0]
            values = [self.results[model][metric] for model in models]
            ax.bar(models, values)
            ax.set_title(f'{metric.upper()} Comparison')
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.set_ylim(0, 1)
            
            # Add value labels on bars
            for j, v in enumerate(values):
                ax.text(j, v + 0.01, f'{v:.3f}', ha='center')
        
        # Feature importance (for logistic regression)
        if 'logistic_regression' in self.models:
            feature_names = self.vectorizer.get_feature_names_out()
            coefficients = self.models['logistic_regression'].coef_[0]
            
            # Top positive and negative features
            top_features = np.argsort(np.abs(coefficients))[-10:]
            
            ax = axes[1, 1]
            ax.barh(range(len(top_features)), coefficients[top_features])
            ax.set_yticks(range(len(top_features)))
            ax.set_yticklabels([feature_names[i] for i in top_features])
            ax.set_title('Top 10 Important Features (Logistic Regression)')
            ax.set_xlabel('Coefficient Value')
        
        plt.tight_layout()
        plt.show()
    
    def predict_new_message(self, message, model_name='logistic_regression'):
        """
        Predict if a new message is fraud/scam
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not trained yet")
        
        # Preprocess message
        processed_message = self.preprocess_text(message)
        
        # Vectorize
        message_tfidf = self.vectorizer.transform([processed_message])
        
        # Predict
        model = self.models[model_name]
        prediction = model.predict(message_tfidf)[0]
        probability = model.predict_proba(message_tfidf)[0]
        
        # Decode label
        predicted_label = self.label_encoder.inverse_transform([prediction])[0]
        
        return {
            'message': message,
            'prediction': predicted_label,
            'probability': {
                'normal': probability[0],
                'fraud': probability[1]
            },
            'confidence': max(probability)
        }
    
    def run_baseline_experiment(self):
        """
        Run complete baseline experiment
        """
        print("=== NLP Fraud Detection Baseline Experiment ===\n")
        
        # Load and prepare data
        print("1. Loading and preprocessing data...")
        self.load_sample_data()
        X_train, X_test, y_train, y_test = self.prepare_data()
        print(f"Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}\n")
        
        # Train models
        print("2. Training baseline models...")
        
        # Logistic Regression
        print("   - Training Logistic Regression...")
        lr_model = self.train_logistic_regression(X_train, y_train)
        lr_results = self.evaluate_model(lr_model, X_test, y_test, 'logistic_regression')
        
        # SVM
        print("   - Training SVM...")
        svm_model = self.train_svm(X_train, y_train)
        svm_results = self.evaluate_model(svm_model, X_test, y_test, 'svm')
        
        print("\n3. Model Performance Summary:")
        print("-" * 50)
        
        for model_name, results in self.results.items():
            print(f"\n{model_name.upper()}:")
            print(f"  Accuracy: {results['accuracy']:.3f}")
            print(f"  F1-Score: {results['f1_score']:.3f}")
            print(f"  AUC Score: {results['auc_score']:.3f}")
        
        # Cross-validation
        print("\n4. Cross-Validation Results:")
        print("-" * 30)
        
        X_full = self.vectorizer.transform(self.data['processed_message'])
        y_full = self.data['label_encoded']
        
        for model_name, model in self.models.items():
            cv_scores = cross_val_score(model, X_full, y_full, cv=5, scoring='f1')
            print(f"{model_name}: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        # Test predictions
        print("\n5. Sample Predictions:")
        print("-" * 20)
        
        test_messages = [
            "URGENT: Your account is suspended. Click here to restore access!",
            "Hey, how was your weekend? Want to grab coffee tomorrow?",
            "You've won a lottery! Send $100 processing fee to claim $50,000!",
            "Meeting reminder: Project review scheduled for 2 PM today"
        ]
        
        for message in test_messages:
            result = self.predict_new_message(message)
            print(f"\nMessage: {message[:50]}...")
            print(f"Prediction: {result['prediction'].upper()}")
            print(f"Confidence: {result['confidence']:.3f}")
        
        # Plot results
        print("\n6. Generating visualizations...")
        self.plot_results()
        
        return self.results

def main():
    """
    Main function to run the baseline experiment
    """
    detector = FraudDetectionBaseline()
    results = detector.run_baseline_experiment()
    
    print("\n=== Experiment Complete ===")
    print("Next steps:")
    print("1. Replace sample data with real datasets (SMS Spam Collection, etc.)")
    print("2. Implement BERT-based classifier")
    print("3. Add more sophisticated preprocessing")
    print("4. Experiment with ensemble methods")
    print("5. Deploy as a web application")

if __name__ == "__main__":
    main()