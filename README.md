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
NLP Detection/
├── README.md                           # This comprehensive documentation
├── requirements.txt                    # Python dependencies
├── baseline_fraud_detection.py        # Traditional ML baseline models
├── bert_fraud_detection.py            # BERT-based classifier
├── fraud_detection_baseline.ipynb     # Interactive Jupyter notebook
├── kaggle_fraud_detection.ipynb       # Kaggle-optimized training notebook
├── fraud-detection-kaggle-training.ipynb # Kaggle training notebook (multiclass)
├── final_fraud_detection_dataset.csv  # Training dataset (Git LFS)
├── models/                            # Saved trained models
│   ├── bert_model/                    # Trained BERT model files
│   └── bert_tokenizer/               # BERT tokenizer files
├── fraud_detection_demo.py           # Full-featured demo script
├── fraud_detection_demo.ipynb        # Interactive demo notebook
├── quick_demo.py                     # Quick verification script
├── .gitattributes                    # Git LFS configuration
└── .git/                             # Git repository
```

## 🚀 Quick Start

### Option 1: Use Pre-trained Model (Recommended)

If you have already trained a model on Kaggle:

1. **Install Dependencies**
   ```bash
   pip install torch transformers pandas numpy matplotlib seaborn jupyter
   ```

2. **Quick Test Your Model**
   ```bash
   python quick_demo.py
   ```

3. **Interactive Demo Notebook** ⭐
   ```bash
   jupyter notebook fraud_detection_demo.ipynb
   ```

4. **Full Demo Script**
   ```bash
   python fraud_detection_demo.py
   ```

### Option 2: Train from Scratch

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

### Option 3: Kaggle Training (Recommended for GPU access)

1. Upload `final_fraud_detection_dataset.csv` to Kaggle
2. Create a new notebook and copy the code from `kaggle_fraud_detection.ipynb`
3. Enable GPU accelerator for fast BERT training
4. Download the trained models from Kaggle output
5. Use the demo scripts to test your trained model

**Note**: The dataset is stored with Git LFS due to its size (~158MB). Clone with `git lfs pull` to download the full dataset.

## 📊 Models Implemented

### 1. Traditional ML Baselines (`baseline_fraud_detection.py`)
- **TF-IDF + Logistic Regression**
- **TF-IDF + Support Vector Machine (SVM)**
- **Features**: 
  - Text preprocessing (stopword removal, lemmatization)
  - TF-IDF vectorization (5000 features)
  - Cross-validation evaluation
  - Feature importance analysis

### 2. BERT-Based Classifier (`bert_fraud_detection.py`)
- **Model**: BERT-base-uncased fine-tuned for classification
- **Features**:
  - Contextual understanding
  - Class imbalance handling (weighted loss)
  - Pre-trained language model knowledge
  - Transfer learning capabilities

### 3. Kaggle Training Notebook (`kaggle_fraud_detection.ipynb`)
- **GPU-accelerated training** on Kaggle's free infrastructure
- **Complete pipeline**: Data loading, preprocessing, training, evaluation
- **Model export**: Saves trained models for download

## 🎮 Demo and Testing Tools

Once you have a trained model, use these tools to test and demonstrate fraud detection capabilities:

### 1. **fraud_detection_demo.ipynb** ⭐ (Recommended)
- **Type**: Interactive Jupyter Notebook
- **Best for**: Exploratory testing, visualizations, learning
- **Features**:
  - Step-by-step model loading
  - Interactive prediction cells
  - Sample test cases for all fraud types
  - Visualizations and analysis
  - Batch prediction capabilities
  - Model information display

### 2. **fraud_detection_demo.py**
- **Type**: Comprehensive Python script
- **Best for**: Integration into applications, command-line use
- **Features**:
  - Full-featured demo class
  - Interactive terminal interface
  - Sample test runner
  - Single and batch predictions
  - Production-ready code structure

### 3. **quick_demo.py**
- **Type**: Simple test script
- **Best for**: Quick verification that your model works
- **Features**:
  - Fast model loading test
  - 5 sample predictions
  - Basic accuracy check
  - Minimal dependencies

## 🎯 Fraud Types Detected

Your trained model can detect these 9 classes:

1. **legitimate** - Normal, safe messages
2. **phishing** - Attempts to steal credentials/personal info
3. **tech_support_scam** - Fake technical support
4. **reward_scam** - Fake prizes/lottery winnings
5. **job_scam** - Fraudulent employment opportunities
6. **sms_spam** - Unwanted promotional messages
7. **popup_scam** - Fake security alerts
8. **refund_scam** - Fake refund/billing notifications
9. **ssn_scam** - Social Security number theft attempts

## � Demo Usage Examples

### Single Prediction
```python
from fraud_detection_demo import FraudDetectionDemo

demo = FraudDetectionDemo()
result = demo.predict_single("Your account has been compromised! Click here now!")
print(f"Prediction: {result['predicted_class']}")
print(f"Confidence: {result['confidence']:.4f}")
print(f"Is Fraud: {result['is_fraud']}")
```

### Batch Prediction
```python
texts = [
    "Meeting at 3 PM tomorrow",
    "URGENT: Verify your SSN now!",
    "You won $10,000! Send fee to claim"
]

results = demo.predict_batch(texts)
for result in results:
    print(f"{result['predicted_class']}: {result['text']}")
```

### Interactive Jupyter Demo
```python
# In fraud_detection_demo.ipynb
your_text = "Your Netflix subscription has expired. Update your payment method to continue watching."
result = predict_fraud(your_text)
display_prediction(result)
```

## �📈 Expected Performance

Based on similar projects and baseline implementations:

| Model Type | Expected Accuracy | F1-Score | Notes |
|------------|------------------|----------|-------|
| Simple Rule-Based | 60-70% | 0.6-0.7 | Quick prototype |
| TF-IDF + LogReg | 80-90% | 0.8-0.9 | Good baseline |
| TF-IDF + SVM | 80-90% | 0.8-0.9 | Robust to noise |
| BERT Fine-tuned | 90-95% | 0.9-0.95 | Best performance |

## �️ Demo Troubleshooting

### Model Not Loading
- ✅ Check that `models/bert_model/` and `models/bert_tokenizer/` exist
- ✅ Verify you downloaded the complete model from Kaggle
- ✅ Ensure all required packages are installed: `pip install torch transformers pandas numpy matplotlib seaborn`

### Low Performance
- 🎯 Check if your test data matches the training distribution
- 🎯 Consider retraining with more diverse examples
- 🎯 Adjust confidence thresholds for your specific needs

### Memory Issues
- 💾 Reduce batch size in `predict_batch()`
- 💾 Use CPU instead of GPU if memory is limited
- 💾 Process smaller chunks of data at a time

### Customization Tips
- **Confidence thresholds**: Consider predictions with confidence < 0.5 as uncertain
- **Adding custom test cases**: Edit the sample test cases in the demo files
- **Integration**: Use the `FraudDetectionDemo` class as a starting point for applications

## �🔧 Configuration

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

### Kaggle Training Configuration
```python
# In kaggle_fraud_detection.ipynb
batch_size = 16      # Adjust based on GPU memory
max_length = 128     # Maximum sequence length
epochs = 3          # Training epochs
learning_rate = 2e-5 # BERT learning rate
```

## 📊 Sample Results

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

### BERT Output:
```
BERT Evaluation Results:
              precision    recall  f1-score   support

      normal       0.92      0.92      0.92        38
       fraud       0.92      0.92      0.92        37

    accuracy                           0.92        75
   macro avg       0.92      0.92      0.92        75
weighted avg       0.92      0.92      0.92        75
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

### Using the Demo Framework
```python
# Production-ready integration using the demo class
from fraud_detection_demo import FraudDetectionDemo

detector = FraudDetectionDemo()

# Single prediction
result = detector.predict_single(user_message)
if result['is_fraud'] and result['confidence'] > 0.8:
    alert_user(result['predicted_class'])
```

### Flask Web App
```python
from flask import Flask, request, jsonify
from fraud_detection_demo import FraudDetectionDemo

app = Flask(__name__)
detector = FraudDetectionDemo()

@app.route('/predict', methods=['POST'])
def predict():
    text = request.json['text']
    result = detector.predict_single(text)
    return jsonify({
        'prediction': result['predicted_class'],
        'is_fraud': result['is_fraud'],
        'confidence': result['confidence']
    })
```

### Streamlit Dashboard
```python
import streamlit as st
from fraud_detection_demo import FraudDetectionDemo

st.title("🛡️ Fraud Detection System")
detector = FraudDetectionDemo()

text_input = st.text_area("Enter message to analyze:")

if st.button("Analyze"):
    result = detector.predict_single(text_input)
    
    if result['is_fraud']:
        st.error(f"🚨 FRAUD DETECTED: {result['predicted_class']}")
    else:
        st.success("✅ Message appears legitimate")
    
    st.write(f"Confidence: {result['confidence']:.2%}")
    
    # Show probability distribution
    st.bar_chart(result['all_probabilities'])
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

### For Model Development
1. **Data Collection**: Gather real fraud/scam datasets
2. **Feature Engineering**: Add metadata features (sender, timestamp, etc.)
3. **Advanced Models**: Experiment with RoBERTa, DistilBERT, or domain-specific models
4. **Active Learning**: Implement feedback loop for continuous improvement
5. **Multi-modal**: Combine text with image analysis for comprehensive detection

### For Production Deployment
1. **Performance Optimization**: Optimize for low-latency inference using the demo framework
2. **A/B Testing**: Compare model performance in production using demo tools
3. **Real-time Processing**: Integrate demo classes into streaming systems
4. **Monitoring**: Use demo tools to validate model performance over time
5. **User Interface**: Build on the Streamlit demo for user-facing applications

### For Demo Enhancement
1. **Interactive Web App**: Extend the Streamlit demo with more features
2. **API Development**: Use the demo classes to build REST APIs
3. **Batch Processing**: Implement large-scale batch prediction capabilities
4. **Model Comparison**: Add functionality to compare multiple model versions
5. **Feedback Collection**: Integrate user feedback mechanisms for continuous learning

## 📚 References

Based on analysis of existing projects including:
- BERT Mail Classification
- Fine-Tuning BERT for Phishing URL Detection
- Spam-T5: Benchmarking LLMs for Email Spam Detection
- Various Kaggle fraud detection competitions

## 🔗 Quick Reference

### Training Files
- `baseline_fraud_detection.py` - Traditional ML models
- `bert_fraud_detection.py` - BERT training script
- `kaggle_fraud_detection.ipynb` - Kaggle training notebook

### Demo Files
- `fraud_detection_demo.ipynb` - Interactive demo notebook ⭐
- `fraud_detection_demo.py` - Full demo script
- `quick_demo.py` - Quick verification

### Model Files (after training)
- `models/bert_model/` - Trained BERT model
- `models/bert_tokenizer/` - BERT tokenizer

### Commands
```bash
# Quick test
python quick_demo.py

# Interactive demo
jupyter notebook fraud_detection_demo.ipynb

# Full demo
python fraud_detection_demo.py

# Train from scratch
python baseline_fraud_detection.py
```

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