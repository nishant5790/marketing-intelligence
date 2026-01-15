"""Prometheus metrics for monitoring."""

import time
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, Generator

from prometheus_client import Counter, Gauge, Histogram, Info

from src.config import settings

# ============================================================================
# Application Info
# ============================================================================

APP_INFO = Info("app", "Application information")
APP_INFO.info({
    "name": settings.app_name,
    "environment": settings.app_env,
})

# ============================================================================
# HTTP Metrics (already in main.py, but can be imported from here)
# ============================================================================

HTTP_REQUESTS_TOTAL = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"],
)

HTTP_REQUEST_DURATION = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "endpoint"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)

# ============================================================================
# Prediction Metrics
# ============================================================================

PREDICTION_REQUESTS = Counter(
    "prediction_requests_total",
    "Total prediction requests",
    ["category", "status"],
)

PREDICTION_LATENCY = Histogram(
    "prediction_latency_seconds",
    "Prediction latency in seconds",
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5],
)

PREDICTION_VALUES = Histogram(
    "prediction_discount_percentage",
    "Distribution of predicted discount percentages",
    buckets=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
)

PREDICTION_CONFIDENCE = Histogram(
    "prediction_confidence",
    "Distribution of prediction confidence scores",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

# ============================================================================
# RAG/Q&A Metrics
# ============================================================================

QA_REQUESTS = Counter(
    "qa_requests_total",
    "Total Q&A requests",
    ["grounded", "status"],
)

QA_LATENCY = Histogram(
    "qa_latency_seconds",
    "Q&A response latency in seconds",
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0],
)

RAG_RESULTS_COUNT = Histogram(
    "rag_results_count",
    "Number of RAG results returned",
    buckets=[0, 1, 2, 3, 4, 5, 7, 10, 15, 20],
)

RAG_SIMILARITY_SCORE = Histogram(
    "rag_similarity_score",
    "Distribution of RAG similarity scores",
    buckets=[0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0],
)

# ============================================================================
# Model Metrics
# ============================================================================

MODEL_LOADED = Gauge(
    "model_loaded",
    "Whether the ML model is loaded (1) or not (0)",
)

MODEL_TRAINING_DURATION = Histogram(
    "model_training_duration_seconds",
    "Model training duration in seconds",
    buckets=[1, 5, 10, 30, 60, 120, 300, 600],
)

MODEL_METRICS = Gauge(
    "model_performance_metric",
    "Model performance metrics",
    ["metric_name"],
)

# ============================================================================
# Vector Database Metrics
# ============================================================================

QDRANT_OPERATIONS = Counter(
    "qdrant_operations_total",
    "Total Qdrant operations",
    ["operation", "status"],
)

QDRANT_LATENCY = Histogram(
    "qdrant_operation_latency_seconds",
    "Qdrant operation latency",
    ["operation"],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)

QDRANT_DOCUMENTS = Gauge(
    "qdrant_documents_count",
    "Number of documents in Qdrant collection",
)

# ============================================================================
# LLM Metrics
# ============================================================================

LLM_REQUESTS = Counter(
    "llm_requests_total",
    "Total LLM API requests",
    ["model", "status"],
)

LLM_LATENCY = Histogram(
    "llm_latency_seconds",
    "LLM response latency in seconds",
    buckets=[0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 30.0],
)

LLM_TOKEN_COUNT = Histogram(
    "llm_response_tokens",
    "Number of tokens in LLM responses",
    buckets=[50, 100, 200, 300, 500, 750, 1000, 1500, 2000],
)

# ============================================================================
# Helper Functions
# ============================================================================


@contextmanager
def track_latency(histogram: Histogram, labels: dict = None) -> Generator[None, None, None]:
    """Context manager to track operation latency.
    
    Args:
        histogram: Prometheus Histogram to record to.
        labels: Optional labels for the histogram.
        
    Yields:
        None
    """
    start_time = time.time()
    try:
        yield
    finally:
        duration = time.time() - start_time
        if labels:
            histogram.labels(**labels).observe(duration)
        else:
            histogram.observe(duration)


def track_prediction(
    category: str,
    predicted_discount: float,
    confidence: float,
    success: bool = True,
) -> None:
    """Track prediction metrics.
    
    Args:
        category: Product category.
        predicted_discount: Predicted discount percentage.
        confidence: Prediction confidence score.
        success: Whether prediction succeeded.
    """
    PREDICTION_REQUESTS.labels(
        category=category,
        status="success" if success else "error",
    ).inc()
    
    if success:
        PREDICTION_VALUES.observe(predicted_discount)
        PREDICTION_CONFIDENCE.observe(confidence)


def track_qa_request(
    grounded: bool,
    num_results: int,
    top_similarity: float,
    success: bool = True,
) -> None:
    """Track Q&A request metrics.
    
    Args:
        grounded: Whether the response was grounded.
        num_results: Number of RAG results.
        top_similarity: Top similarity score.
        success: Whether request succeeded.
    """
    QA_REQUESTS.labels(
        grounded=str(grounded),
        status="success" if success else "error",
    ).inc()
    
    if success:
        RAG_RESULTS_COUNT.observe(num_results)
        if top_similarity > 0:
            RAG_SIMILARITY_SCORE.observe(top_similarity)


def update_model_metrics(metrics: dict[str, float]) -> None:
    """Update model performance metrics.
    
    Args:
        metrics: Dictionary of metric names and values.
    """
    for name, value in metrics.items():
        MODEL_METRICS.labels(metric_name=name).set(value)


def timed(histogram: Histogram):
    """Decorator to time function execution.
    
    Args:
        histogram: Prometheus Histogram to record to.
        
    Returns:
        Decorated function.
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with track_latency(histogram):
                return func(*args, **kwargs)
        return wrapper
    return decorator
