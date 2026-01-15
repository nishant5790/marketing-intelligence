"""Integration tests for API endpoints."""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# Set test environment before imports
os.environ["APP_ENV"] = "development"
os.environ["GEMINI_API_KEY"] = "test_api_key"


@pytest.fixture
def client():
    """Create test client."""
    from src.main import app
    return TestClient(app)


@pytest.fixture
def mock_predictor():
    """Mock the predictor for testing."""
    with patch("src.api.routes.predict.get_predictor") as mock:
        predictor = MagicMock()
        predictor.is_loaded = True
        predictor.model_path = Path("models/test.joblib")
        predictor.metrics = {"rmse": 5.0, "mae": 3.5, "r2": 0.85}
        predictor.predict_from_features.return_value = {
            "predicted_discount": 25.5,
            "confidence": 0.85,
            "features_used": ["category", "actual_price", "rating", "rating_count"],
        }
        predictor.get_feature_importance.return_value = {
            "category": 0.35,
            "actual_price": 0.25,
            "rating": 0.22,
            "rating_count": 0.18,
        }
        mock.return_value = predictor
        yield predictor


@pytest.fixture
def mock_retriever():
    """Mock the retriever for testing."""
    with patch("src.api.routes.qa.get_retriever") as mock:
        retriever = MagicMock()
        retriever.health_check.return_value = True
        retriever.search.return_value = [
            {
                "id": "prod1",
                "score": 0.92,
                "text": "Sony Headphones",
                "product_name": "Sony WH-1000",
                "category": "Electronics",
                "actual_price": 349.99,
                "rating": 4.8,
            }
        ]
        retriever.get_context_for_query.return_value = "1. Sony WH-1000\n   Price: $349.99"
        retriever.get_sources.return_value = [
            {"product": "Sony WH-1000", "relevance": 0.92, "id": "prod1"}
        ]
        mock.return_value = retriever
        yield retriever


@pytest.fixture
def mock_gemini():
    """Mock the Gemini client for testing."""
    with patch("src.api.routes.qa.get_gemini_client") as mock:
        client = MagicMock()
        client.answer_question.return_value = {
            "answer": "Based on our catalog, the Sony WH-1000 headphones are highly recommended...",
            "sources": [{"product": "Sony WH-1000", "relevance": 0.92, "id": "prod1"}],
            "grounded": True,
            "question": "What are the best headphones?",
        }
        client.health_check.return_value = {
            "status": "healthy",
            "configured": True,
            "available_models": ["gemini-2.0-flash"],
        }
        mock.return_value = client
        yield client


class TestRootEndpoint:
    """Tests for root endpoint."""

    def test_root_returns_info(self, client):
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data
        assert "endpoints" in data


class TestHealthEndpoint:
    """Tests for health check endpoint."""

    def test_health_check(self, client):
        """Test health check with mocked dependencies."""
        mock_predictor = MagicMock()
        mock_predictor.is_loaded = True
        mock_predictor.model_path = Path("models/test.joblib")
        
        mock_retriever = MagicMock()
        mock_retriever.health_check.return_value = True
        
        mock_gemini = MagicMock()
        mock_gemini.health_check.return_value = {
            "status": "healthy",
            "configured": True,
            "available_models": ["gemini-2.0-flash"],
        }
        
        # Patch at the import locations used inside the health_check function
        with patch("src.ml.predictor.get_predictor", return_value=mock_predictor):
            with patch("src.rag.retriever.get_retriever", return_value=mock_retriever):
                with patch("src.llm.gemini_client.get_gemini_client", return_value=mock_gemini):
                    response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "components" in data
        assert "version" in data


class TestPredictDiscountEndpoint:
    """Tests for /predict_discount endpoint."""

    def test_predict_discount_success(self, client, mock_predictor):
        request_data = {
            "category": "Electronics",
            "actual_price": 999.99,
            "rating": 4.5,
            "rating_count": 500,
        }
        
        response = client.post("/predict_discount", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert "predicted_discount" in data
        assert "confidence" in data
        assert "explanation" in data
        assert 0 <= data["predicted_discount"] <= 100

    def test_predict_discount_invalid_price(self, client):
        request_data = {
            "category": "Electronics",
            "actual_price": -10,  # Invalid
            "rating": 4.5,
            "rating_count": 500,
        }
        
        response = client.post("/predict_discount", json=request_data)
        
        assert response.status_code == 422  # Validation error

    def test_predict_discount_invalid_rating(self, client):
        request_data = {
            "category": "Electronics",
            "actual_price": 100,
            "rating": 6.0,  # Invalid, max is 5
            "rating_count": 500,
        }
        
        response = client.post("/predict_discount", json=request_data)
        
        assert response.status_code == 422

    def test_predict_discount_missing_field(self, client):
        request_data = {
            "category": "Electronics",
            "actual_price": 100,
            # Missing rating and rating_count
        }
        
        response = client.post("/predict_discount", json=request_data)
        
        assert response.status_code == 422


class TestAnswerQuestionEndpoint:
    """Tests for /answer_question endpoint."""

    def test_answer_question_success(self, client, mock_retriever, mock_gemini):
        request_data = {
            "question": "What are the best headphones?",
        }
        
        response = client.post("/answer_question", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert "grounded" in data
        assert "question" in data

    def test_answer_question_with_filters(self, client, mock_retriever, mock_gemini):
        request_data = {
            "question": "What are the best headphones under $100?",
            "filter_category": "Electronics",
            "filter_max_price": 100,
            "filter_min_rating": 4.0,
        }
        
        response = client.post("/answer_question", json=request_data)
        
        assert response.status_code == 200

    def test_answer_question_too_short(self, client):
        request_data = {
            "question": "Hi",  # Too short (min 3 chars)
        }
        
        response = client.post("/answer_question", json=request_data)
        
        assert response.status_code == 422

    def test_answer_question_too_long(self, client):
        request_data = {
            "question": "x" * 1001,  # Too long
        }
        
        response = client.post("/answer_question", json=request_data)
        
        assert response.status_code == 422


class TestModelStatusEndpoint:
    """Tests for model status endpoint."""

    def test_model_status(self, client, mock_predictor):
        response = client.get("/predict/status")
        
        assert response.status_code == 200
        data = response.json()
        assert "loaded" in data
        assert "model_path" in data


class TestRAGHealthEndpoint:
    """Tests for RAG health endpoint."""

    def test_rag_health(self, client, mock_retriever, mock_gemini):
        response = client.get("/qa/health")
        
        assert response.status_code == 200
        data = response.json()
        assert "qdrant" in data
        assert "gemini" in data


class TestMetricsEndpoint:
    """Tests for Prometheus metrics endpoint."""

    def test_metrics_endpoint(self, client):
        response = client.get("/metrics")
        
        assert response.status_code == 200
        assert "text/plain" in response.headers["content-type"]
        # Should contain some prometheus metrics
        assert "http_requests_total" in response.text or "python" in response.text


class TestAnalysisEndpoints:
    """Tests for analysis endpoints."""

    def test_dataset_info(self, client):
        """Test dataset info endpoint."""
        response = client.get("/analysis/dataset-info")
        
        assert response.status_code == 200
        data = response.json()
        assert "total_products" in data or "error" not in data

    def test_analysis_summary(self, client):
        """Test analysis summary endpoint."""
        response = client.get("/analysis/summary")
        
        assert response.status_code == 200
        data = response.json()
        assert "dataset_info" in data

    def test_categories_endpoint(self, client):
        """Test categories analysis endpoint."""
        response = client.get("/analysis/categories")
        
        assert response.status_code == 200

    def test_prices_endpoint(self, client):
        """Test prices analysis endpoint."""
        response = client.get("/analysis/prices")
        
        assert response.status_code == 200

    def test_ratings_endpoint(self, client):
        """Test ratings analysis endpoint."""
        response = client.get("/analysis/ratings")
        
        assert response.status_code == 200

    def test_discounts_endpoint(self, client):
        """Test discounts analysis endpoint."""
        response = client.get("/analysis/discounts")
        
        assert response.status_code == 200

    def test_top_products_endpoint(self, client):
        """Test top products endpoint."""
        response = client.get("/analysis/top-products?by=rating&n=5")
        
        assert response.status_code == 200
        data = response.json()
        assert "products" in data
        assert "metric" in data

    def test_top_products_invalid_metric(self, client):
        """Test top products with invalid metric."""
        response = client.get("/analysis/top-products?by=invalid_metric")
        
        assert response.status_code == 400

    def test_correlations_endpoint(self, client):
        """Test correlations endpoint."""
        response = client.get("/analysis/correlations")
        
        assert response.status_code == 200
