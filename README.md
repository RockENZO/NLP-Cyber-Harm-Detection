# NLP Fraud/Scam Detection Baseline Models

[![GitHub Repository](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/RockENZO/NLP-Cyber-Harm-Detection.git)

A comprehensive baseline implementation for fraud and scam detection using Natural Language Processing techniques. This project provides multiple approaches from simple keyword-based detection to advanced BERT-based classification.

## 📁 Repository

🔗 **GitHub Repository**: [https://github.com/RockENZO/NLP-Cyber-Harm-Detection.git](https://github.com/RockENZO/NLP-Cyber-Harm-Detection.git)


## 🎯 Project Overview

This project implements baseline models for detecting fraudulent content (scams, phishing, spam) in text data, based on analysis of existing similar projects. It includes:

- **Traditional ML approaches** (TF-IDF + SVM/Logistic Regression)
- **Deep Learning approaches** (BERT-based classification)
- **Simple rule-based approach** (for quick prototyping)

## 📁 Project Structure

```
nlp project/
├── simple_fraud_demo.py          # Simple demo (no dependencies)
├── baseline_fraud_detection.py   # Traditional ML baselines
├── bert_fraud_detection.py       # BERT-based classifier
├── setup.py                      # Setup and installation script
├── requirements.txt               # Python dependencies
├── README.md                      # This file
├── untitled.txt                  # Project analysis and recommendations
└── Analysis on Existing Similar Projects.pdf
```

## 🚀 Quick Start

### Option 1: Simple Demo (No Installation Required)

Run the simple demo to understand the concepts:

```bash
python simple_fraud_demo.py
```

This demo uses only built-in Python libraries and demonstrates basic fraud detection using keyword analysis.

### Option 2: Full Baseline Models

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Traditional ML Baselines**
   ```bash
   python baseline_fraud_detection.py
   ```

3. **Run BERT Baseline** (requires more computational resources)
   ```bash
   python bert_fraud_detection.py
   ```

### Option 3: Automated Setup

Run the setup script for guided installation and execution:

```bash
python setup.py
```

## 📊 Models Implemented

### 1. Simple Rule-Based Classifier (`simple_fraud_demo.py`)
- **Approach**: Keyword-based scoring
- **Features**: Fraud/normal keyword dictionaries
- **Pros**: Fast, interpretable, no dependencies
- **Cons**: Limited accuracy, can't handle context

### 2. Traditional ML Baselines (`baseline_fraud_detection.py`)
- **TF-IDF + Logistic Regression**
- **TF-IDF + Support Vector Machine (SVM)**
- **Features**: 
  - Text preprocessing (stopword removal, lemmatization)
  - TF-IDF vectorization (5000 features)
  - Cross-validation evaluation
  - Feature importance analysis

### 3. BERT-Based Classifier (`bert_fraud_detection.py`)
- **Model**: BERT-base-uncased fine-tuned for classification
- **Features**:
  - Contextual understanding
  - Class imbalance handling (weighted loss)
  - Pre-trained language model knowledge
  - Transfer learning capabilities

## 📈 Expected Performance

Based on similar projects and baseline implementations:

| Model Type | Expected Accuracy | F1-Score | Notes |
|------------|------------------|----------|-------|
| Simple Rule-Based | 60-70% | 0.6-0.7 | Quick prototype |
| TF-IDF + LogReg | 80-90% | 0.8-0.9 | Good baseline |
| TF-IDF + SVM | 80-90% | 0.8-0.9 | Robust to noise |
| BERT Fine-tuned | 90-95% | 0.9-0.95 | Best performance |

## 🔧 Configuration

### Adjusting the Simple Detector
```python
detector = SimpleFraudDetector()
detector.fraud_score_threshold = 0.3  # Adjust sensitivity
```

### Traditional ML Parameters
```python
# In baseline_fraud_detection.py
vectorizer = TfidfVectorizer(
    max_features=5000,    # Vocabulary size
    stop_words='english', # Remove common words
    ngram_range=(1, 2)    # Use unigrams and bigrams
)
```

### BERT Configuration
```python
# In bert_fraud_detection.py
classifier = BERTFraudClassifier(
    model_name='bert-base-uncased',  # Or 'distilbert-base-uncased' for faster training
    max_length=128,                  # Maximum sequence length
    num_classes=2                    # Binary classification
)
```

## 📊 Sample Results

### Simple Demo Output:
```
Results:
Accuracy: 0.867
Precision: 0.800
Recall: 0.800
F1-Score: 0.800

Testing on new messages:
1. Message: URGENT: Your bank account is compromised! Click here now!
   Prediction: FRAUD
   Fraud Score: 0.650
   Confidence: 0.300
```

### Traditional ML Output:
```
LOGISTIC_REGRESSION:
  Accuracy: 0.889
  F1-Score: 0.889
  AUC Score: 0.944

SVM:
  Accuracy: 0.889
  F1-Score: 0.889
  AUC Score: 0.944
```

## 📋 Data Requirements

### Current Implementation
- Uses synthetic sample data for demonstration
- 15 fraud messages + 15 normal messages
- Easily expandable with real datasets

### Recommended Real Datasets
1. **SMS Spam Collection** - Classic spam detection
2. **Phishing URL Dataset** - URL-based fraud detection
3. **Enron Email Dataset** - Email fraud detection
4. **Social Media Scam Data** - Social platform fraud

### Data Format Expected
```python
data = pd.DataFrame({
    'message': ['text content here', ...],
    'label': ['fraud', 'normal', ...]
})
```

## 🛠️ Extending the Models

### Adding New Features
```python
# Add sentiment analysis
from textblob import TextBlob

def add_sentiment_features(text):
    blob = TextBlob(text)
    return {
        'polarity': blob.sentiment.polarity,
        'subjectivity': blob.sentiment.subjectivity
    }
```

### Custom Preprocessing
```python
def custom_preprocess(text):
    # Add domain-specific preprocessing
    text = remove_urls(text)
    text = normalize_currency(text)
    return text
```

### Ensemble Methods
```python
# Combine multiple models
def ensemble_predict(text):
    lr_pred = lr_model.predict_proba(text)[0][1]
    svm_pred = svm_model.predict_proba(text)[0][1]
    bert_pred = bert_model.predict(text)['probabilities']['fraud']
    
    # Weighted average
    final_score = 0.3 * lr_pred + 0.3 * svm_pred + 0.4 * bert_pred
    return 'fraud' if final_score > 0.5 else 'normal'
```

## 🌐 Deployment Options

### Flask Web App
```python
from flask import Flask, request, jsonify

app = Flask(__name__)
detector = FraudDetectionBaseline()

@app.route('/predict', methods=['POST'])
def predict():
    text = request.json['text']
    result = detector.predict_new_message(text)
    return jsonify(result)
```

### Streamlit Dashboard
```python
import streamlit as st

st.title("Fraud Detection System")
text_input = st.text_area("Enter message to analyze:")

if st.button("Analyze"):
    result = detector.predict_new_message(text_input)
    st.write(f"Prediction: {result['prediction']}")
    st.write(f"Confidence: {result['confidence']:.2f}")
```

## 🔍 Evaluation Metrics

The models are evaluated using:
- **Accuracy**: Overall correctness
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Area under the ROC curve

For fraud detection, **Recall** is often most important (don't miss actual fraud).

## 🚧 Known Limitations

1. **Sample Data**: Currently uses synthetic data; real datasets needed for production
2. **Class Imbalance**: Real fraud data is typically very imbalanced
3. **Context**: Simple models may miss contextual nuances
4. **Adversarial Examples**: Sophisticated scammers may evade detection
5. **Language Support**: Currently English-only

## 🔮 Next Steps

1. **Data Collection**: Gather real fraud/scam datasets
2. **Feature Engineering**: Add metadata features (sender, timestamp, etc.)
3. **Advanced Models**: Experiment with RoBERTa, DistilBERT, or domain-specific models
4. **Active Learning**: Implement feedback loop for continuous improvement
5. **Multi-modal**: Combine text with image analysis for comprehensive detection
6. **Real-time Processing**: Optimize for low-latency inference
7. **A/B Testing**: Compare model performance in production

## 📚 References

Based on analysis of existing projects including:
- BERT Mail Classification
- Fine-Tuning BERT for Phishing URL Detection
- Spam-T5: Benchmarking LLMs for Email Spam Detection
- Various Kaggle fraud detection competitions

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add your improvements
4. Submit a pull request

## 📄 License

This project is provided as-is for educational and research purposes.

---

**Note**: This is a baseline implementation. For production use, consider:
- Larger, diverse datasets
- More sophisticated preprocessing
- Ensemble methods
- Regular model retraining
- Bias and fairness considerations