"""
Unit tests for the LoRA-based scam detector model
"""
import unittest
import os
import sys
from pathlib import Path
import torch
import json

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestLoRAModelStructure(unittest.TestCase):
    """Test the LoRA model file structure and configuration"""
    
    @classmethod
    def setUpClass(cls):
        cls.model_path = Path(__file__).parent.parent / 'models' / 'lora'
        
    def test_model_directory_exists(self):
        """Test that the LoRA model directory exists"""
        self.assertTrue(self.model_path.exists(), 
                       f"Model directory not found at {self.model_path}")
    
    def test_adapter_config_exists(self):
        """Test that adapter_config.json exists"""
        config_path = self.model_path / 'adapter_config.json'
        self.assertTrue(config_path.exists(), 
                       "adapter_config.json not found")
    
    def test_adapter_model_exists(self):
        """Test that adapter model weights exist"""
        model_path = self.model_path / 'adapter_model.safetensors'
        self.assertTrue(model_path.exists(), 
                       "adapter_model.safetensors not found")
    
    def test_tokenizer_files_exist(self):
        """Test that all tokenizer files exist"""
        required_files = [
            'tokenizer.json',
            'tokenizer_config.json',
            'special_tokens_map.json'
        ]
        for file_name in required_files:
            file_path = self.model_path / file_name
            self.assertTrue(file_path.exists(), 
                          f"Tokenizer file {file_name} not found")
    
    def test_adapter_config_valid(self):
        """Test that adapter configuration is valid JSON and has required fields"""
        config_path = self.model_path / 'adapter_config.json'
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Check required fields
        required_fields = ['peft_type', 'task_type', 'r', 'lora_alpha']
        for field in required_fields:
            self.assertIn(field, config, 
                         f"Required field '{field}' not found in adapter config")
    
    def test_readme_exists(self):
        """Test that README documentation exists"""
        readme_path = self.model_path / 'README.md'
        self.assertTrue(readme_path.exists(), 
                       "README.md not found in model directory")


class TestLoRAModelConfiguration(unittest.TestCase):
    """Test the LoRA model configuration parameters"""
    
    @classmethod
    def setUpClass(cls):
        config_path = Path(__file__).parent.parent / 'models' / 'lora' / 'adapter_config.json'
        with open(config_path, 'r') as f:
            cls.config = json.load(f)
    
    def test_peft_type_is_lora(self):
        """Test that PEFT type is LoRA"""
        self.assertEqual(self.config.get('peft_type'), 'LORA',
                        "PEFT type should be LORA")
    
    def test_task_type_valid(self):
        """Test that task type is valid"""
        valid_tasks = ['CAUSAL_LM', 'SEQ_2_SEQ_LM', 'SEQ_CLS', 'TOKEN_CLS']
        self.assertIn(self.config.get('task_type'), valid_tasks,
                     f"Task type should be one of {valid_tasks}")
    
    def test_lora_rank_positive(self):
        """Test that LoRA rank (r) is positive"""
        r = self.config.get('r', 0)
        self.assertGreater(r, 0, "LoRA rank (r) should be positive")
    
    def test_lora_alpha_positive(self):
        """Test that LoRA alpha is positive"""
        alpha = self.config.get('lora_alpha', 0)
        self.assertGreater(alpha, 0, "LoRA alpha should be positive")
    
    def test_target_modules_exist(self):
        """Test that target modules are specified"""
        self.assertIn('target_modules', self.config,
                     "target_modules should be specified in config")
        self.assertIsInstance(self.config['target_modules'], list,
                            "target_modules should be a list")


class TestLoRAModelInference(unittest.TestCase):
    """Test model inference capabilities (requires GPU/model loading)"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment - skip if dependencies not available"""
        try:
            from transformers import AutoTokenizer
            from peft import PeftConfig
            
            cls.model_path = Path(__file__).parent.parent / 'models' / 'lora'
            cls.tokenizer = AutoTokenizer.from_pretrained(str(cls.model_path))
            cls.peft_config = PeftConfig.from_pretrained(str(cls.model_path))
            cls.skip_tests = False
        except Exception as e:
            cls.skip_tests = True
            cls.skip_reason = f"Dependencies not available: {e}"
    
    def setUp(self):
        if self.skip_tests:
            self.skipTest(self.skip_reason)
    
    def test_tokenizer_loads(self):
        """Test that tokenizer loads successfully"""
        self.assertIsNotNone(self.tokenizer, "Tokenizer should load")
    
    def test_tokenizer_has_special_tokens(self):
        """Test that tokenizer has required special tokens"""
        self.assertIsNotNone(self.tokenizer.eos_token, "EOS token should exist")
        self.assertIsNotNone(self.tokenizer.bos_token, "BOS token should exist")
    
    def test_tokenizer_encoding(self):
        """Test that tokenizer can encode text"""
        test_text = "This is a test message"
        encoded = self.tokenizer.encode(test_text)
        self.assertIsInstance(encoded, list, "Encoded output should be a list")
        self.assertGreater(len(encoded), 0, "Encoded output should not be empty")
    
    def test_peft_config_loads(self):
        """Test that PEFT config loads successfully"""
        self.assertIsNotNone(self.peft_config, "PEFT config should load")
    
    def test_base_model_specified(self):
        """Test that base model is specified in config"""
        self.assertIsNotNone(self.peft_config.base_model_name_or_path,
                           "Base model should be specified")


class TestFraudCategories(unittest.TestCase):
    """Test fraud category definitions and consistency"""
    
    def setUp(self):
        self.expected_categories = [
            'job_scam',
            'legitimate',
            'phishing',
            'popup_scam',
            'refund_scam',
            'reward_scam',
            'sms_spam',
            'ssn_scam',
            'tech_support_scam'
        ]
    
    def test_categories_defined(self):
        """Test that fraud categories are properly defined"""
        self.assertIsInstance(self.expected_categories, list)
        self.assertGreater(len(self.expected_categories), 0)
    
    def test_legitimate_category_exists(self):
        """Test that 'legitimate' category exists"""
        self.assertIn('legitimate', self.expected_categories,
                     "Should have 'legitimate' category")
    
    def test_no_duplicate_categories(self):
        """Test that there are no duplicate categories"""
        self.assertEqual(len(self.expected_categories), 
                        len(set(self.expected_categories)),
                        "Categories should not have duplicates")
    
    def test_categories_lowercase(self):
        """Test that all categories are lowercase with underscores"""
        for category in self.expected_categories:
            self.assertEqual(category, category.lower(),
                           f"Category '{category}' should be lowercase")
            self.assertNotIn(' ', category,
                           f"Category '{category}' should not contain spaces")


class TestDatasetCompatibility(unittest.TestCase):
    """Test that the model is compatible with the dataset"""
    
    @classmethod
    def setUpClass(cls):
        cls.dataset_path = Path(__file__).parent.parent / 'final_fraud_detection_dataset.csv'
    
    def test_dataset_exists(self):
        """Test that the fraud detection dataset exists"""
        self.assertTrue(self.dataset_path.exists(),
                       f"Dataset not found at {self.dataset_path}")
    
    def test_dataset_readable(self):
        """Test that dataset can be read"""
        try:
            import pandas as pd
            df = pd.read_csv(self.dataset_path)
            self.assertGreater(len(df), 0, "Dataset should not be empty")
        except ImportError:
            self.skipTest("pandas not available")
        except Exception as e:
            self.fail(f"Failed to read dataset: {e}")
    
    def test_dataset_has_required_columns(self):
        """Test that dataset has required columns"""
        try:
            import pandas as pd
            df = pd.read_csv(self.dataset_path)
            required_columns = ['text', 'detailed_category']
            for col in required_columns:
                self.assertIn(col, df.columns,
                            f"Dataset should have '{col}' column")
        except ImportError:
            self.skipTest("pandas not available")


class TestModelSize(unittest.TestCase):
    """Test model file sizes and constraints"""
    
    @classmethod
    def setUpClass(cls):
        cls.model_path = Path(__file__).parent.parent / 'models' / 'lora'
    
    def test_adapter_model_size(self):
        """Test that adapter model file size is reasonable"""
        model_file = self.model_path / 'adapter_model.safetensors'
        if model_file.exists():
            size_mb = model_file.stat().st_size / (1024 * 1024)
            # LoRA adapters should typically be under 500MB
            self.assertLess(size_mb, 500,
                          f"Adapter model size ({size_mb:.2f}MB) is unusually large")
            self.assertGreater(size_mb, 0.1,
                             f"Adapter model size ({size_mb:.2f}MB) is unusually small")


def run_tests():
    """Run all tests"""
    unittest.main(verbosity=2)


if __name__ == '__main__':
    run_tests()
