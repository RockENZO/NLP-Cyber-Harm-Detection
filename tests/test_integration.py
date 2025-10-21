"""
Integration tests for the fraud detection system
"""
import unittest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestModelIntegration(unittest.TestCase):
    """Test integration between different components"""
    
    def test_models_directory_structure(self):
        """Test that models directory has expected structure"""
        models_dir = Path(__file__).parent.parent / 'models'
        self.assertTrue(models_dir.exists(), "models directory should exist")
        
        # Check for LoRA model
        lora_dir = models_dir / 'lora'
        self.assertTrue(lora_dir.exists(), "LoRA model directory should exist")
    
    def test_demos_directory_exists(self):
        """Test that demos directory exists"""
        demos_dir = Path(__file__).parent.parent / 'demos'
        self.assertTrue(demos_dir.exists(), "demos directory should exist")
    
    def test_lora_demo_exists(self):
        """Test that LoRA demo notebook exists"""
        demo_path = Path(__file__).parent.parent / 'demos' / 'lora-kaggle-test.ipynb'
        self.assertTrue(demo_path.exists(), 
                       "lora-kaggle-test.ipynb should exist in demos")


class TestProjectStructure(unittest.TestCase):
    """Test overall project structure"""
    
    @classmethod
    def setUpClass(cls):
        cls.base_dir = Path(__file__).parent.parent
    
    def test_readme_exists(self):
        """Test that main README exists"""
        readme = self.base_dir / 'README.md'
        self.assertTrue(readme.exists(), "Main README.md should exist")
    
    def test_requirements_exists(self):
        """Test that requirements.txt exists"""
        requirements = self.base_dir / 'requirements.txt'
        self.assertTrue(requirements.exists(), "requirements.txt should exist")
    
    def test_gitignore_exists(self):
        """Test that .gitignore exists"""
        gitignore = self.base_dir / '.gitignore'
        self.assertTrue(gitignore.exists(), ".gitignore should exist")
    
    def test_required_directories_exist(self):
        """Test that all required directories exist"""
        required_dirs = ['models', 'demos', 'tests', 'training', 'docs']
        for dir_name in required_dirs:
            dir_path = self.base_dir / dir_name
            self.assertTrue(dir_path.exists(), 
                          f"{dir_name} directory should exist")


class TestModelDocumentation(unittest.TestCase):
    """Test that models have proper documentation"""
    
    def test_lora_readme_exists(self):
        """Test that LoRA model has README"""
        readme_path = Path(__file__).parent.parent / 'models' / 'lora' / 'README.md'
        self.assertTrue(readme_path.exists(), 
                       "LoRA model should have README.md")
    
    def test_lora_readme_not_empty(self):
        """Test that LoRA README has content"""
        readme_path = Path(__file__).parent.parent / 'models' / 'lora' / 'README.md'
        if readme_path.exists():
            content = readme_path.read_text()
            self.assertGreater(len(content), 100, 
                             "README should have substantial content")


if __name__ == '__main__':
    unittest.main(verbosity=2)
