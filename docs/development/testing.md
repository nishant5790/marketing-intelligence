# Testing Guide

Comprehensive guide for testing the Marketing Data Intelligence system.

## Test Structure

```
tests/
├── __init__.py
├── conftest.py              # Shared fixtures
├── unit/                    # Unit tests
│   ├── test_data_loader.py
│   ├── test_preprocessor.py
│   ├── test_predictor.py
│   ├── test_embedder.py
│   ├── test_eda.py
│   ├── test_schemas.py
│   └── test_safety.py
├── integration/             # Integration tests
│   └── test_api.py
└── load/                    # Load tests
    └── locustfile.py
```

## Running Tests

### Quick Commands

```bash
# Run all tests
uv run pytest

# Run with verbose output
uv run pytest -v

# Run specific test file
uv run pytest tests/unit/test_predictor.py

# Run specific test
uv run pytest tests/unit/test_predictor.py::test_prediction_range

# Run with coverage
uv run pytest --cov=src --cov-report=html
```

### Test Categories

```bash
# Unit tests only
uv run pytest tests/unit -v

# Integration tests only
uv run pytest tests/integration -v

# Skip slow tests
uv run pytest -m "not slow"
```

## Test Configuration

### pytest.ini (via pyproject.toml)

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
markers = [
    "slow: marks tests as slow",
    "integration: marks integration tests",
]
```

### conftest.py Fixtures

```python
# tests/conftest.py

import pytest
import pandas as pd
import numpy as np

@pytest.fixture
def sample_dataframe():
    """Generate sample DataFrame for testing."""
    return pd.DataFrame({
        "product_id": ["P001", "P002"],
        "product_name": ["Product A", "Product B"],
        "main_category": ["Electronics", "Clothing"],
        "actual_price": [100.0, 50.0],
        "rating": [4.5, 3.8],
        "rating_count": [100, 50],
        "discount_percentage": [20.0, 15.0],
    })

@pytest.fixture
def mock_predictor():
    """Create mock predictor for testing."""
    # Implementation
```

## Unit Tests

### Testing Data Loader

```python
# tests/unit/test_data_loader.py

import pytest
from src.data.loader import clean_price, clean_rating, load_amazon_dataset

class TestCleanPrice:
    def test_clean_rupee_price(self):
        assert clean_price("₹1,299") == 1299.0
    
    def test_clean_dollar_price(self):
        assert clean_price("$99.99") == 99.99
    
    def test_clean_invalid_price(self):
        assert clean_price("invalid") is None
    
    def test_clean_none_price(self):
        assert clean_price(None) is None

class TestCleanRating:
    def test_clean_simple_rating(self):
        assert clean_rating("4.5") == 4.5
    
    def test_clean_verbose_rating(self):
        assert clean_rating("4.5 out of 5 stars") == 4.5
```

### Testing Predictor

```python
# tests/unit/test_predictor.py

import pytest
import numpy as np
from src.ml.predictor import DiscountPredictor

class TestDiscountPredictor:
    def test_prediction_range(self, trained_predictor):
        """Predictions should be in valid range [0, 100]."""
        X = np.random.rand(10, 11)
        predictions = trained_predictor.predict(X)
        
        assert all(predictions >= 0)
        assert all(predictions <= 100)
    
    def test_model_not_loaded_error(self):
        """Should raise error when model not loaded."""
        predictor = DiscountPredictor()
        
        with pytest.raises(ValueError, match="Model not trained"):
            predictor.predict(np.array([[1, 2, 3]]))
    
    def test_confidence_calculation(self, trained_predictor):
        """Confidence should be between 0.3 and 1.0."""
        confidence = trained_predictor._calculate_confidence(
            actual_price=500,
            rating=4.0,
            rating_count=100
        )
        
        assert 0.3 <= confidence <= 1.0
```

### Testing Schemas

```python
# tests/unit/test_schemas.py

import pytest
from pydantic import ValidationError
from src.api.schemas import PredictDiscountRequest

class TestPredictDiscountRequest:
    def test_valid_request(self):
        request = PredictDiscountRequest(
            category="Electronics",
            actual_price=999.99,
            rating=4.5,
            rating_count=100
        )
        assert request.category == "Electronics"
    
    def test_invalid_price(self):
        with pytest.raises(ValidationError):
            PredictDiscountRequest(
                category="Electronics",
                actual_price=-10,  # Invalid
                rating=4.5,
                rating_count=100
            )
    
    def test_invalid_rating(self):
        with pytest.raises(ValidationError):
            PredictDiscountRequest(
                category="Electronics",
                actual_price=999.99,
                rating=6.0,  # Invalid: > 5
                rating_count=100
            )
```

## Integration Tests

### Testing API Endpoints

```python
# tests/integration/test_api.py

import pytest
from httpx import AsyncClient, ASGITransport
from src.main import app

@pytest.fixture
async def client():
    """Create async test client."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client

class TestHealthEndpoint:
    @pytest.mark.asyncio
    async def test_health_check(self, client):
        response = await client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "components" in data

class TestPredictEndpoint:
    @pytest.mark.asyncio
    async def test_predict_discount(self, client):
        response = await client.post(
            "/predict_discount",
            json={
                "category": "Electronics",
                "actual_price": 999.99,
                "rating": 4.5,
                "rating_count": 100
            }
        )
        
        # May be 503 if model not loaded
        assert response.status_code in [200, 503]
    
    @pytest.mark.asyncio
    async def test_predict_invalid_input(self, client):
        response = await client.post(
            "/predict_discount",
            json={
                "category": "Electronics",
                "actual_price": -10,  # Invalid
                "rating": 4.5,
                "rating_count": 100
            }
        )
        
        assert response.status_code == 422

class TestQAEndpoint:
    @pytest.mark.asyncio
    async def test_answer_question(self, client):
        response = await client.post(
            "/answer_question",
            json={
                "question": "What are the best products?"
            }
        )
        
        # May fail if Gemini not configured
        assert response.status_code in [200, 503]
```

## Load Tests

### Locust Configuration

```python
# tests/load/locustfile.py

from locust import HttpUser, task, between

class MarketingAPIUser(HttpUser):
    wait_time = between(1, 3)
    
    @task(3)
    def predict_discount(self):
        self.client.post(
            "/predict_discount",
            json={
                "category": "Electronics",
                "actual_price": 999.99,
                "rating": 4.5,
                "rating_count": 100
            }
        )
    
    @task(1)
    def answer_question(self):
        self.client.post(
            "/answer_question",
            json={
                "question": "What are the best deals?"
            }
        )
    
    @task(5)
    def health_check(self):
        self.client.get("/health")
```

### Running Load Tests

```bash
# Start locust web UI
uv run locust -f tests/load/locustfile.py --host=http://localhost:8000

# Open browser
open http://localhost:8089

# Headless mode
uv run locust -f tests/load/locustfile.py \
  --host=http://localhost:8000 \
  --headless \
  --users=50 \
  --spawn-rate=10 \
  --run-time=1m
```

## Test Coverage

### Running Coverage

```bash
# Generate coverage report
uv run pytest --cov=src --cov-report=html --cov-report=term-missing

# View HTML report
open htmlcov/index.html
```

### Coverage Configuration

```toml
# pyproject.toml
[tool.coverage.run]
source = ["src"]
omit = ["*/__init__.py", "*/conftest.py"]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "if TYPE_CHECKING:",
    "if __name__ == .__main__.:",
]
```

## Mocking

### Mocking External Services

```python
from unittest.mock import Mock, patch

@pytest.fixture
def mock_gemini():
    """Mock Gemini client."""
    with patch("src.llm.gemini_client.GeminiClient") as mock:
        mock_instance = Mock()
        mock_instance.generate.return_value = "Test response"
        mock.return_value = mock_instance
        yield mock_instance

def test_with_mock_gemini(mock_gemini):
    # Test code that uses Gemini
    result = mock_gemini.generate("test")
    assert result == "Test response"
```

### Mocking Qdrant

```python
@pytest.fixture
def mock_qdrant():
    """Mock Qdrant client."""
    with patch("src.rag.retriever.QdrantClient") as mock:
        mock_instance = Mock()
        mock_instance.search.return_value = []
        mock.return_value = mock_instance
        yield mock_instance
```

## Test Best Practices

### Naming Conventions

```python
# test_{module}_{functionality}
def test_predictor_returns_valid_range():
    pass

def test_loader_handles_missing_file():
    pass

def test_schema_validates_rating_bounds():
    pass
```

### Test Organization

```python
class TestClassName:
    """Group related tests."""
    
    def test_normal_case(self):
        """Test expected behavior."""
        pass
    
    def test_edge_case(self):
        """Test boundary conditions."""
        pass
    
    def test_error_case(self):
        """Test error handling."""
        pass
```

### Async Testing

```python
@pytest.mark.asyncio
async def test_async_endpoint():
    """Use asyncio marker for async tests."""
    result = await some_async_function()
    assert result is not None
```

## Continuous Integration

### GitHub Actions Example

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    services:
      qdrant:
        image: qdrant/qdrant
        ports:
          - 6333:6333
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      
      - name: Install uv
        run: pip install uv
      
      - name: Install dependencies
        run: uv sync --all-extras
      
      - name: Run tests
        env:
          QDRANT_HOST: localhost
        run: uv run pytest --cov=src --cov-report=xml
      
      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

## Debugging Tests

### Running Single Test with Debug

```bash
# Verbose output
uv run pytest tests/unit/test_predictor.py -v -s

# With debugger
uv run pytest tests/unit/test_predictor.py --pdb

# Stop on first failure
uv run pytest tests/unit/ -x
```

### Inspecting Test Failures

```python
# Add detailed assertions
def test_example():
    result = function_under_test()
    
    # Better error message
    assert result == expected, f"Got {result}, expected {expected}"
    
    # Or use pytest.approx for floats
    assert result == pytest.approx(expected, rel=0.01)
```

## Related Documentation

- [Development Setup](./setup.md) - Environment setup
- [API Reference](../api/endpoints.md) - API documentation
- [Architecture](../architecture/overview.md) - System design
