"""Unit tests for API schemas."""

import pytest
from pydantic import ValidationError

from src.api.schemas import (
    AnswerQuestionRequest,
    AnswerQuestionResponse,
    FeatureExplanation,
    PredictDiscountRequest,
    PredictDiscountResponse,
    SourceReference,
)


class TestPredictDiscountRequest:
    """Tests for PredictDiscountRequest schema."""

    def test_valid_request(self):
        request = PredictDiscountRequest(
            category="Electronics",
            actual_price=999.99,
            rating=4.5,
            rating_count=100,
        )
        
        assert request.category == "Electronics"
        assert request.actual_price == 999.99
        assert request.rating == 4.5
        assert request.rating_count == 100

    def test_invalid_price_zero(self):
        with pytest.raises(ValidationError):
            PredictDiscountRequest(
                category="Electronics",
                actual_price=0,  # Must be > 0
                rating=4.5,
                rating_count=100,
            )

    def test_invalid_price_negative(self):
        with pytest.raises(ValidationError):
            PredictDiscountRequest(
                category="Electronics",
                actual_price=-10,
                rating=4.5,
                rating_count=100,
            )

    def test_invalid_rating_too_high(self):
        with pytest.raises(ValidationError):
            PredictDiscountRequest(
                category="Electronics",
                actual_price=100,
                rating=6.0,  # Max is 5
                rating_count=100,
            )

    def test_invalid_rating_negative(self):
        with pytest.raises(ValidationError):
            PredictDiscountRequest(
                category="Electronics",
                actual_price=100,
                rating=-1.0,
                rating_count=100,
            )

    def test_invalid_rating_count_negative(self):
        with pytest.raises(ValidationError):
            PredictDiscountRequest(
                category="Electronics",
                actual_price=100,
                rating=4.0,
                rating_count=-5,
            )


class TestPredictDiscountResponse:
    """Tests for PredictDiscountResponse schema."""

    def test_valid_response(self):
        response = PredictDiscountResponse(
            predicted_discount=25.5,
            confidence=0.85,
            explanation=FeatureExplanation(
                top_features=["category", "price"],
                importance_scores={"category": 0.4, "price": 0.3},
            ),
        )
        
        assert response.predicted_discount == 25.5
        assert response.confidence == 0.85
        assert len(response.explanation.top_features) == 2

    def test_default_explanation(self):
        response = PredictDiscountResponse(
            predicted_discount=20.0,
            confidence=0.9,
        )
        
        assert response.explanation.top_features == []
        assert response.explanation.importance_scores == {}

    def test_invalid_confidence(self):
        with pytest.raises(ValidationError):
            PredictDiscountResponse(
                predicted_discount=20.0,
                confidence=1.5,  # Must be <= 1
            )


class TestAnswerQuestionRequest:
    """Tests for AnswerQuestionRequest schema."""

    def test_valid_request(self):
        request = AnswerQuestionRequest(
            question="What are the best laptops?",
        )
        
        assert request.question == "What are the best laptops?"
        assert request.filter_category is None
        assert request.top_k is None

    def test_with_filters(self):
        request = AnswerQuestionRequest(
            question="What are the best laptops?",
            filter_category="Electronics",
            filter_max_price=1000,
            filter_min_rating=4.0,
            top_k=10,
        )
        
        assert request.filter_category == "Electronics"
        assert request.filter_max_price == 1000
        assert request.filter_min_rating == 4.0
        assert request.top_k == 10

    def test_question_too_short(self):
        with pytest.raises(ValidationError):
            AnswerQuestionRequest(question="Hi")  # Min 3 chars

    def test_question_too_long(self):
        with pytest.raises(ValidationError):
            AnswerQuestionRequest(question="x" * 1001)  # Max 1000 chars

    def test_invalid_top_k(self):
        with pytest.raises(ValidationError):
            AnswerQuestionRequest(
                question="Test question",
                top_k=25,  # Max is 20
            )


class TestAnswerQuestionResponse:
    """Tests for AnswerQuestionResponse schema."""

    def test_valid_response(self):
        response = AnswerQuestionResponse(
            answer="Here are the best laptops...",
            sources=[
                SourceReference(product="MacBook Pro", relevance=0.95),
                SourceReference(product="Dell XPS", relevance=0.88),
            ],
            grounded=True,
            question="What are the best laptops?",
        )
        
        assert response.grounded is True
        assert len(response.sources) == 2
        assert response.sources[0].product == "MacBook Pro"

    def test_empty_sources(self):
        response = AnswerQuestionResponse(
            answer="I don't have information about that.",
            sources=[],
            grounded=False,
            question="Unknown question",
        )
        
        assert response.grounded is False
        assert response.sources == []


class TestSourceReference:
    """Tests for SourceReference schema."""

    def test_valid_source(self):
        source = SourceReference(
            product="Test Product",
            relevance=0.85,
            id="PROD001",
        )
        
        assert source.product == "Test Product"
        assert source.relevance == 0.85
        assert source.id == "PROD001"

    def test_optional_id(self):
        source = SourceReference(
            product="Test Product",
            relevance=0.85,
        )
        
        assert source.id is None

    def test_invalid_relevance(self):
        with pytest.raises(ValidationError):
            SourceReference(
                product="Test",
                relevance=1.5,  # Max is 1
            )
