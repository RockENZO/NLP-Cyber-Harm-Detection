# Test Suite for Fraud Detection Models

This directory contains unit tests and integration tests for the fraud detection system, including the LoRA-based scam detector model.

## Test Files

### `test_lora_model.py`
Comprehensive tests for the LoRA model including:
- Model structure validation
- Configuration verification
- Tokenizer functionality
- Fraud category definitions
- Dataset compatibility
- Model file size checks

### `test_integration.py`
Integration tests for:
- Project structure validation
- Model-demo compatibility
- Documentation completeness
- Directory structure

## Running Tests

### Run all tests
```bash
python -m pytest tests/
```

### Run specific test file
```bash
python -m pytest tests/test_lora_model.py -v
```

### Run with unittest
```bash
cd /Users/admin/Desktop/Workbench/Baseline\ Demo
python -m unittest discover -s tests -p "test_*.py" -v
```

### Run a specific test class
```bash
python -m unittest tests.test_lora_model.TestLoRAModelStructure -v
```

## Test Requirements

Install test dependencies:
```bash
pip install pytest pytest-cov
```

Optional dependencies for full test suite:
```bash
pip install transformers peft torch pandas
```

## Test Coverage

To run tests with coverage:
```bash
pytest tests/ --cov=models --cov-report=html
```

## Continuous Integration

These tests are designed to run in CI/CD pipelines. Some tests that require GPU or large model downloads are skipped automatically if dependencies are not available.

## Test Categories

- **Structure Tests**: Verify file existence and organization
- **Configuration Tests**: Validate model configurations
- **Inference Tests**: Test model loading and basic inference (optional)
- **Integration Tests**: Verify component compatibility
- **Dataset Tests**: Ensure dataset compatibility

## Adding New Tests

When adding new models or features:
1. Add corresponding tests in `test_*.py` files
2. Ensure tests can run without GPU when possible
3. Document any special requirements
4. Update this README
