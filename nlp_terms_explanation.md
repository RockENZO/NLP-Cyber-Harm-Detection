# NLP Key Terms Explanation: TF-IDF and BERT Fine-Tuning

This document provides detailed explanations of two key concepts used in the NLP Fraud Detection Baseline project: TF-IDF and BERT-based fine-tuning.

## TF-IDF (Term Frequency-Inverse Document Frequency)

### What it is
TF-IDF is a statistical measure used to evaluate the importance of a word in a document relative to a collection of documents (corpus). It combines two components:

- **Term Frequency (TF)**: How often a word appears in a specific document.
- **Inverse Document Frequency (IDF)**: How rare or common a word is across the entire corpus (logarithmically scaled to reduce the weight of common words).

The formula is: `TF-IDF = TF * IDF`. This helps highlight unique words that are important for distinguishing documents, reducing the impact of common words like "the" or "is".

### Why used in NLP
It's a simple, effective way to convert text into numerical vectors for machine learning models, capturing semantic importance without deep learning.

### Implementation in your project
- Uses `TfidfVectorizer` from scikit-learn.
- Configured with parameters like `max_features=5000` (limits vocabulary), `ngram_range=(1, 2)` (includes unigrams and bigrams), and `stop_words='english'` (removes common words).
- Applied to preprocessed text: `X_tfidf = tfidf_vectorizer.fit_transform(df['cleaned_message'])`.
- Output is a sparse matrix of TF-IDF scores, which is then fed into models like Logistic Regression or SVM for classification.

#### Example code snippet:
```python
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))
X_tfidf = vectorizer.fit_transform(cleaned_texts)
```

## BERT-Based Fine-Tune Structure

### What it is
BERT (Bidirectional Encoder Representations from Transformers) is a pre-trained transformer model that understands context in text by processing words bidirectionally. Fine-tuning adapts this pre-trained model for a specific task (e.g., binary classification for fraud detection) by training it on labeled data with a small learning rate.

### Structure
- **Pre-trained BERT**: Trained on massive text corpora (e.g., Wikipedia) to learn general language patterns.
- **Fine-tuning process**: Add a classification head (e.g., a linear layer) on top of BERT's output. Freeze most BERT layers and train only the head plus a few top layers on your dataset.
- **Key components**: Input text is tokenized, passed through BERT layers, and the [CLS] token's output is used for classification. Uses techniques like attention mechanisms for contextual understanding.

### Why used
Excels at capturing nuanced meanings, sarcasm, or context that simpler models miss, leading to higher accuracy in tasks like fraud detection.

### Implementation
- Uses `BertForSequenceClassification` from the `transformers` library.
- Tokenizes input with `BertTokenizer`, handling max length (e.g., 128 tokens).
- Fine-tunes with `AdamW` optimizer, linear scheduler, and cross-entropy loss (weighted for class imbalance).
- Training loop: Forward pass through BERT, compute loss, backpropagate, and update weights over epochs (e.g., 2-3 epochs).
- Handles imbalanced data with class weights and evaluates with metrics like F1-score and AUC.

#### Example code snippet:
```python
from transformers import BertTokenizer, BertForSequenceClassification, AdamW

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Tokenize and prepare data
inputs = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors='pt')

# Fine-tune loop (simplified)
optimizer = AdamW(model.parameters(), lr=2e-5)
for epoch in range(3):
    outputs = model(**inputs, labels=labels)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
```

