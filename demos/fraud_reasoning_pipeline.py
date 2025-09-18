#!/usr/bin/env python3
"""
Fraud Detection with Reasoning Pipeline
=======================================

This demo combines DistilBERT for fast fraud classification with GPT2-medium 
for detailed reasoning explanations, using the main fraud detection dataset.

Dataset: final_fraud_detection_dataset.csv (main Kaggle dataset)
Based on LLM testing results:
- GPT2-medium: Best reasoning quality (4.9%), moderate speed (0.26s)
- DistilBERT: Fast classification with good accuracy

Usage:
    python fraud_reasoning_pipeline.py

Features:
- Fast fraud classification using trained DistilBERT
- Detailed reasoning explanations using GPT2-medium
- Real dataset examples from main fraud detection dataset
- Confidence scores and fraud indicators
- Interactive demo with actual dataset samples
"""

import torch
import numpy as np
import pandas as pd
from transformers import (
    DistilBertTokenizer, 
    DistilBertForSequenceClassification,
    GPT2LMHeadModel, 
    GPT2Tokenizer
)
import warnings
warnings.filterwarnings('ignore')

class FraudReasoningPipeline:
    """
    Complete fraud detection pipeline with reasoning explanations
    """
    
    def __init__(self, 
                 distilbert_model_path='../models/distilbert_model',
                 distilbert_tokenizer_path='../models/distilbert_tokenizer'):
        """
        Initialize the fraud reasoning pipeline
        
        Args:
            distilbert_model_path: Path to trained DistilBERT model
            distilbert_tokenizer_path: Path to DistilBERT tokenizer
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Class labels (alphabetical order from training)
        self.class_labels = [
            'job_scam',        # 0
            'legitimate',      # 1  
            'phishing',        # 2
            'popup_scam',      # 3
            'refund_scam',     # 4
            'reward_scam',     # 5
            'sms_spam',        # 6
            'ssn_scam',        # 7
            'tech_support_scam' # 8
        ]
        
        # Fraud indicators for reasoning
        self.fraud_indicators = {
            'urgency_words': ['urgent', 'immediate', 'expires', 'limited time', 'act now', 'hurry'],
            'financial_words': ['money', 'payment', 'bank', 'account', 'credit', 'cash', 'loan', 'fee'],
            'action_words': ['click', 'call', 'send', 'verify', 'confirm', 'update', 'download'],
            'threat_words': ['suspended', 'blocked', 'expired', 'compromised', 'security', 'alert'],
            'reward_words': ['winner', 'congratulations', 'free', 'prize', 'gift', 'selected']
        }
        
        print(f"🚀 Initializing Fraud Reasoning Pipeline...")
        print(f"Device: {self.device}")
        print(f"📊 Classification: DistilBERT (fast inference)")
        print(f"🧠 Reasoning: GPT2-medium (best LLM from analysis)")
        
        # Load models
        self._load_classification_model(distilbert_model_path, distilbert_tokenizer_path)
        self._load_reasoning_model()
    
    def _load_classification_model(self, model_path, tokenizer_path):
        """Load DistilBERT for fast classification"""
        try:
            print("📥 Loading DistilBERT tokenizer...")
            self.classifier_tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_path)
            
            print("📥 Loading DistilBERT model...")
            self.classifier_model = DistilBertForSequenceClassification.from_pretrained(model_path)
            self.classifier_model.to(self.device)
            self.classifier_model.eval()
            
            print("✅ DistilBERT loaded successfully!")
            
        except Exception as e:
            print(f"❌ Error loading DistilBERT: {e}")
            raise
    
    def _load_reasoning_model(self):
        """Load GPT2-medium for reasoning explanations"""
        try:
            print("📥 Loading GPT2-medium for reasoning...")
            
            # Use pre-trained GPT2-medium (best from LLM analysis)
            self.reasoning_tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
            self.reasoning_model = GPT2LMHeadModel.from_pretrained('gpt2-medium')
            
            # Set pad token
            self.reasoning_tokenizer.pad_token = self.reasoning_tokenizer.eos_token
            
            self.reasoning_model.to(self.device)
            self.reasoning_model.eval()
            
            print("✅ GPT2-medium loaded successfully!")
            
        except Exception as e:
            print(f"❌ Error loading GPT2-medium: {e}")
            print("⚠️  Will continue without reasoning explanations")
            self.reasoning_model = None
            self.reasoning_tokenizer = None
    
    def classify_text(self, text, max_length=128):
        """
        Classify text using DistilBERT
        
        Args:
            text: Input text to classify
            max_length: Maximum sequence length
            
        Returns:
            dict: Classification results
        """
        # Tokenize input
        inputs = self.classifier_tokenizer(
            text,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Move to device
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        
        # Get prediction
        with torch.no_grad():
            outputs = self.classifier_model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)
            predicted_class_id = torch.argmax(logits, dim=1).item()
            confidence = probabilities[0][predicted_class_id].item()
        
        # Format results
        predicted_class = self.class_labels[predicted_class_id]
        all_probabilities = {
            self.class_labels[i]: prob.item() 
            for i, prob in enumerate(probabilities[0])
        }
        
        return {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'is_fraud': predicted_class != 'legitimate',
            'all_probabilities': all_probabilities
        }
    
    def analyze_fraud_indicators(self, text):
        """
        Analyze text for fraud indicators
        
        Args:
            text: Input text to analyze
            
        Returns:
            dict: Fraud indicators found
        """
        text_lower = text.lower()
        found_indicators = {}
        
        for category, words in self.fraud_indicators.items():
            found_words = [word for word in words if word in text_lower]
            if found_words:
                found_indicators[category] = found_words
        
        return found_indicators
    
    def generate_reasoning(self, text, classification_result, fraud_indicators, max_length=200):
        """
        Generate reasoning explanation using GPT2-medium
        
        Args:
            text: Original text
            classification_result: Classification results
            fraud_indicators: Found fraud indicators
            max_length: Maximum generation length
            
        Returns:
            str: Generated reasoning explanation
        """
        predicted_class = classification_result['predicted_class']
        confidence = classification_result['confidence']
        is_fraud = classification_result['is_fraud']
        
        # Skip detailed reasoning for legitimate messages
        if predicted_class == 'legitimate':
            return "Message classified as legitimate - no detailed reasoning required."
        
        if not self.reasoning_model:
            return "Reasoning model not available."
        
        # Build indicator summary
        indicator_summary = ""
        if fraud_indicators:
            categories = list(fraud_indicators.keys())
            indicator_summary = f" Contains {', '.join(categories)} indicators."
        
        # Create prompt for reasoning (only for fraud)
        fraud_status = "fraudulent"
        prompt = f"Text analysis: This message is classified as {fraud_status} ({predicted_class}) with {confidence:.1%} confidence.{indicator_summary} Explanation:"
        
        try:
            # Generate reasoning
            inputs = self.reasoning_tokenizer.encode(prompt, return_tensors='pt', max_length=512, truncation=True)
            inputs = inputs.to(self.device)
            
            with torch.no_grad():
                outputs = self.reasoning_model.generate(
                    inputs,
                    max_length=inputs.shape[1] + max_length,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.reasoning_tokenizer.eos_token_id,
                    no_repeat_ngram_size=2
                )
            
            # Decode and clean
            generated_text = self.reasoning_tokenizer.decode(outputs[0], skip_special_tokens=True)
            reasoning = generated_text[len(prompt):].strip()
            
            # Clean up the reasoning
            if reasoning:
                # Remove incomplete sentences at the end
                sentences = reasoning.split('.')
                if len(sentences) > 1 and len(sentences[-1].strip()) < 10:
                    reasoning = '.'.join(sentences[:-1]) + '.'
                return reasoning[:300]  # Limit length
            else:
                return "The classification confidence suggests a clear pattern match."
                
        except Exception as e:
            print(f"⚠️ Error generating reasoning: {e}")
            return "Unable to generate detailed reasoning at this time."
    
    def comprehensive_analysis(self, text):
        """
        Perform comprehensive fraud analysis with reasoning
        
        Args:
            text: Input text to analyze
            
        Returns:
            dict: Complete analysis results
        """
        # Classify text
        classification = self.classify_text(text)
        
        # Analyze fraud indicators
        indicators = self.analyze_fraud_indicators(text)
        
        # Generate reasoning
        reasoning = self.generate_reasoning(text, classification, indicators)
        
        return {
            'text': text,
            'classification': classification,
            'fraud_indicators': indicators,
            'reasoning': reasoning,
            'risk_score': self._calculate_risk_score(classification, indicators)
        }
    
    def _calculate_risk_score(self, classification, indicators):
        """Calculate overall risk score"""
        base_score = 0 if classification['predicted_class'] == 'legitimate' else 50
        confidence_score = classification['confidence'] * 50
        indicator_score = len(indicators) * 5
        
        risk_score = min(100, base_score + confidence_score + indicator_score)
        return risk_score
    
    def print_analysis(self, result):
        """Pretty print analysis results"""
        print("\n" + "="*80)
        print("🔍 COMPREHENSIVE FRAUD ANALYSIS")
        print("="*80)
        
        print(f"📝 Text: {result['text'][:100]}{'...' if len(result['text']) > 100 else ''}")
        
        # Classification results
        classification = result['classification']
        print(f"\n🎯 CLASSIFICATION RESULTS:")
        print(f"   Prediction: {classification['predicted_class'].upper()}")
        print(f"   Confidence: {classification['confidence']:.4f}")
        print(f"   Is Fraud: {'YES' if classification['is_fraud'] else 'NO'}")
        print(f"   Risk Score: {result['risk_score']:.1f}/100")
        
        # Fraud indicators
        if result['fraud_indicators']:
            print(f"\n🚨 FRAUD INDICATORS DETECTED:")
            for category, words in result['fraud_indicators'].items():
                print(f"   {category.title()}: {', '.join(words)}")
        else:
            print(f"\n✅ NO FRAUD INDICATORS DETECTED")
        
        # Top probabilities
        print(f"\n📊 TOP PROBABILITIES:")
        sorted_probs = sorted(classification['all_probabilities'].items(), 
                            key=lambda x: x[1], reverse=True)[:3]
        for class_name, prob in sorted_probs:
            bar = "█" * int(prob * 20)
            print(f"   {class_name:20s} {prob:.4f} {bar}")
        
        # Reasoning
        print(f"\n🧠 REASONING EXPLANATION:")
        print(f"   {result['reasoning']}")
    
    def get_sample_cases(self):
        """Get real dataset test cases for demonstration"""
        try:
            import pandas as pd
            
            # Load the main fraud detection dataset
            print("📊 Loading main fraud detection dataset...")
            df = pd.read_csv('../final_fraud_detection_dataset.csv')
            
            print(f"✅ Loaded dataset with {len(df)} total records")
            print(f"📋 Available categories: {df['detailed_category'].unique()}")
            
            real_cases = []
            
            # Get samples from each fraud category
            categories = df['detailed_category'].unique()
            
            for category in categories[:8]:  # Limit to 8 categories for demo
                category_samples = df[df['detailed_category'] == category].head(1)  # 1 sample per category
                
                for _, row in category_samples.iterrows():
                    if pd.notna(row['text']) and len(row['text'].strip()) > 10:
                        # Truncate very long texts for demo
                        text = row['text'][:250] + ('...' if len(row['text']) > 250 else '')
                        real_cases.append(text)
            
            if real_cases:
                print(f"📈 Extracted {len(real_cases)} real cases from main dataset")
                return real_cases
                
        except Exception as e:
            print(f"⚠️ Could not load main dataset: {e}")
            print("📋 Trying analysis results as fallback...")
            
            try:
                # Fallback to analysis results
                df1 = pd.read_csv('../runs/fraud_analysis_results_20250916_155231.csv')
                df2 = pd.read_csv('../runs/gpt2_fraud_analysis_20250917_034015.csv')
                
                real_cases = []
                
                # Extract cases from analysis results
                for _, row in df1.head(5).iterrows():
                    if pd.notna(row['text']):
                        real_cases.append(row['text'])
                
                for _, row in df2.head(5).iterrows():
                    if pd.notna(row['text']):
                        real_cases.append(row['text'])
                
                if real_cases:
                    print("📊 Using analysis results as fallback")
                    return real_cases[:10]
                    
            except Exception as e2:
                print(f"⚠️ Could not load analysis results either: {e2}")
                print("📋 Using curated real examples from dataset...")
        
        # Final fallback with manually extracted real examples from the actual dataset
        return [
            # Real examples from the main fraud detection dataset
            "Position Summary The Asset Manager will plan, develop and execute on a wide variety of projects...",
            "We offer interns that can develop web sites references as well as profit sharing. Great chance to get your skills shown to the world...",
            "We are a Health Benefits company. Helping people save money when they cant afford insurance. 2 positions available...",
            "Your account will be closed unless you verify immediately!",
            "Hi John, thanks for the great meeting today. Let's follow up next week.",
            "You've won a free vacation! Call now to claim your prize!",
            "Your Netflix account will be suspended unless you update payment info immediately",
            "WARNING: 5 viruses found on your device! Download our antivirus now",
            "Congratulations! You've won $50,000 in our weekly lottery draw!",
            "URGENT: Your social security number has been compromised!"
        ]
    
    def run_sample_demo(self):
        """Run demonstration with real dataset cases"""
        print("\n🧪 RUNNING REAL DATASET FRAUD ANALYSIS")
        print("="*80)
        print("📊 Using actual fraud cases from previous analysis results")
        
        sample_cases = self.get_sample_cases()
        
        for i, text in enumerate(sample_cases, 1):
            print(f"\n📋 Real Case {i}/{len(sample_cases)}")
            print("-" * 40)
            
            result = self.comprehensive_analysis(text)
            self.print_analysis(result)
            
            # Brief summary
            classification = result['classification']
            fraud_status = "FRAUD" if classification['is_fraud'] else "LEGITIMATE"
            print(f"\n✅ SUMMARY: {fraud_status} | Confidence: {classification['confidence']:.1%} | Risk: {result['risk_score']:.0f}/100")
    
    def interactive_demo(self):
        """Run interactive demo"""
        print("\n🎮 INTERACTIVE FRAUD DETECTION WITH REASONING")
        print("="*80)
        print("Enter text messages to analyze (type 'quit' to exit)")
        print("Type 'samples' to run real dataset demonstrations")
        print("-" * 80)
        
        while True:
            user_input = input("\n📝 Enter text to analyze: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Thanks for using the fraud reasoning pipeline!")
                break
                
            elif user_input.lower() == 'samples':
                self.run_sample_demo()
                continue
                
            elif not user_input:
                print("⚠️ Please enter some text to analyze")
                continue
            
            # Perform comprehensive analysis
            result = self.comprehensive_analysis(user_input)
            self.print_analysis(result)


def main():
    """Main function to run the demo"""
    print("🚀 Fraud Detection with Reasoning Pipeline")
    print("=" * 50)
    print("📊 Using REAL dataset from fraud analysis results")
    print("Based on LLM Analysis Results:")
    print("• Classification: DistilBERT (fast, accurate)")
    print("• Reasoning: GPT2-medium (best quality: 4.9%)")
    print("=" * 50)
    
    try:
        # Initialize pipeline
        pipeline = FraudReasoningPipeline()
        
        # Show menu
        print("\n📋 Available Options:")
        print("1. Interactive demo (analyze your own text)")
        print("2. Run real dataset demonstrations")
        print("3. Both real dataset and interactive")
        
        choice = input("\nSelect option (1-3): ").strip()
        
        if choice == '1':
            pipeline.interactive_demo()
        elif choice == '2':
            pipeline.run_sample_demo()
        elif choice == '3':
            pipeline.run_sample_demo()
            pipeline.interactive_demo()
        else:
            print("Invalid choice. Running interactive demo...")
            pipeline.interactive_demo()
            
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure DistilBERT model files are in '../models/distilbert_model/'")
        print("2. Make sure DistilBERT tokenizer files are in '../models/distilbert_tokenizer/'")
        print("3. Install required packages:")
        print("   pip install torch transformers")
        print("4. Check internet connection for GPT2-medium download")


if __name__ == "__main__":
    main()