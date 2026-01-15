"""Pydantic schemas for API request/response models."""

from typing import Any, List, Optional

from pydantic import BaseModel, Field


# ============================================================================
# Discount Prediction Schemas
# ============================================================================


class PredictDiscountRequest(BaseModel):
    """Request model for discount prediction."""

    category: str = Field(..., description="Product category (e.g., 'Electronics')")
    actual_price: float = Field(..., gt=0, description="Original product price")
    rating: float = Field(..., ge=0, le=5, description="Product rating (0-5)")
    rating_count: int = Field(..., ge=0, description="Number of ratings/reviews")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "category": "Electronics",
                    "actual_price": 1499.99,
                    "rating": 4.2,
                    "rating_count": 1250,
                }
            ]
        }
    }


class FeatureExplanation(BaseModel):
    """Feature importance explanation."""

    top_features: List[str] = Field(default_factory=list)
    importance_scores: dict[str, float] = Field(default_factory=dict)


class PredictDiscountResponse(BaseModel):
    """Response model for discount prediction."""

    predicted_discount: float = Field(..., description="Predicted discount percentage")
    confidence: float = Field(..., ge=0, le=1, description="Prediction confidence score")
    explanation: FeatureExplanation = Field(default_factory=FeatureExplanation)

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "predicted_discount": 23.5,
                    "confidence": 0.85,
                    "explanation": {
                        "top_features": ["category", "rating_count"],
                        "importance_scores": {"category": 0.35, "rating_count": 0.28},
                    },
                }
            ]
        }
    }


# ============================================================================
# Q&A (RAG) Schemas
# ============================================================================


class AnswerQuestionRequest(BaseModel):
    """Request model for question answering."""

    question: str = Field(..., min_length=3, max_length=1000, description="User question")
    filter_category: Optional[str] = Field(None, description="Filter results by category")
    filter_min_price: Optional[float] = Field(None, ge=0, description="Minimum price filter")
    filter_max_price: Optional[float] = Field(None, ge=0, description="Maximum price filter")
    filter_min_rating: Optional[float] = Field(None, ge=0, le=5, description="Minimum rating filter")
    top_k: Optional[int] = Field(None, ge=1, le=20, description="Number of products to consider")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "question": "What are the best rated headphones under $100?",
                    "filter_max_price": 100,
                    "filter_min_rating": 4.0,
                }
            ]
        }
    }


class SourceReference(BaseModel):
    """Source reference for grounded responses."""

    product: str = Field(..., description="Product name")
    relevance: float = Field(..., ge=0, le=1, description="Relevance score")
    id: Optional[str] = Field(None, description="Product ID")


class AnswerQuestionResponse(BaseModel):
    """Response model for question answering."""

    answer: str = Field(..., description="Generated answer")
    sources: List[SourceReference] = Field(default_factory=list, description="Source products")
    grounded: bool = Field(..., description="Whether answer is grounded in retrieved data")
    question: str = Field(..., description="Original question")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "answer": "Based on our catalog, the top-rated headphones under $100 are...",
                    "sources": [
                        {"product": "Sony WH-1000", "relevance": 0.92, "id": "PROD001"}
                    ],
                    "grounded": True,
                    "question": "What are the best rated headphones under $100?",
                }
            ]
        }
    }


# ============================================================================
# Health & Status Schemas
# ============================================================================


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Overall system status")
    components: dict[str, Any] = Field(default_factory=dict, description="Component statuses")
    version: str = Field(..., description="API version")


class ModelStatusResponse(BaseModel):
    """ML model status response."""

    loaded: bool = Field(..., description="Whether model is loaded")
    model_path: str = Field(..., description="Path to model file")
    metrics: dict[str, float] = Field(default_factory=dict, description="Model metrics")
    feature_importance: dict[str, float] = Field(
        default_factory=dict, description="Feature importance scores"
    )


# ============================================================================
# Explainability Schemas
# ============================================================================


class ExplainRequest(BaseModel):
    """Request model for prediction explanation."""

    category: str = Field(..., description="Product category")
    actual_price: float = Field(..., gt=0, description="Original product price")
    rating: float = Field(..., ge=0, le=5, description="Product rating")
    rating_count: int = Field(..., ge=0, description="Number of ratings")


class ShapValues(BaseModel):
    """SHAP values for explainability."""

    feature_name: str
    value: float
    shap_value: float
    contribution: str  # "positive" or "negative"


class ExplainResponse(BaseModel):
    """Response model for prediction explanation."""

    predicted_discount: float
    shap_values: List[ShapValues]
    base_value: float
    explanation_summary: str


# ============================================================================
# Data Management Schemas
# ============================================================================


class IndexDataRequest(BaseModel):
    """Request to index data into the RAG system."""

    recreate_collection: bool = Field(
        default=False, description="Whether to recreate the collection"
    )


class IndexDataResponse(BaseModel):
    """Response for data indexing."""

    status: str
    documents_indexed: int
    collection_info: dict[str, Any]


class TrainModelRequest(BaseModel):
    """Request to train/retrain the ML model."""

    use_sample_data: bool = Field(
        default=False, description="Use generated sample data instead of real data"
    )
    test_size: float = Field(default=0.2, ge=0.1, le=0.5, description="Test split ratio")


class TrainModelResponse(BaseModel):
    """Response for model training."""

    status: str
    metrics: dict[str, float]
    feature_importance: dict[str, float]
    model_path: str
    training_samples: int
    test_samples: int
