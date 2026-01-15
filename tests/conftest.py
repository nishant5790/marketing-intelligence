"""Pytest configuration and fixtures."""

import os
import tempfile
from pathlib import Path
from typing import Generator

import numpy as np
import pandas as pd
import pytest

# Set test environment
os.environ["APP_ENV"] = "development"
os.environ["GEMINI_API_KEY"] = "test_key"
os.environ["QDRANT_HOST"] = "localhost"
os.environ["QDRANT_PORT"] = "6333"


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """Create a sample DataFrame for testing with enhanced features."""
    np.random.seed(42)
    n_samples = 50
    
    main_categories = ["Electronics", "Clothing", "Home&Kitchen", "Books", "Sports"]
    sub_categories = ["Gadgets", "Apparel", "Furniture", "Fiction", "Equipment"]
    price_tiers = ["budget", "mid", "premium", "luxury"]
    
    data = {
        "product_id": [f"PROD{i:04d}" for i in range(n_samples)],
        "product_name": [f"Test Product {i} - High Quality Item" for i in range(n_samples)],
        "main_category": np.random.choice(main_categories, n_samples),
        "sub_category": np.random.choice(sub_categories, n_samples),
        "category_depth": np.random.randint(2, 6, n_samples),
        "actual_price": np.random.uniform(100, 5000, n_samples).round(2),
        "rating": np.random.uniform(2.5, 5.0, n_samples).round(1),
        "rating_count": np.random.randint(10, 10000, n_samples),
        "about_product": [f"Description for product {i}. " * 5 for i in range(n_samples)],
        "review_content": [f"Great product, highly recommend. " * 3 for i in range(n_samples)],
        "name_length": np.random.randint(20, 100, n_samples),
        "description_length": np.random.randint(100, 1000, n_samples),
        "price_tier": np.random.choice(price_tiers, n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Log-transformed rating count
    df["rating_count_log"] = np.log1p(df["rating_count"])
    
    # Generate realistic discounts based on features
    base_discount = 10 + np.random.uniform(-5, 5, n_samples)
    price_factor = (df["actual_price"] / 1000) * 5
    rating_factor = (5 - df["rating"]) * 3
    
    df["discount_percentage"] = np.clip(
        base_discount + price_factor + rating_factor + np.random.normal(0, 5, n_samples),
        0,
        70,
    ).round(2)
    
    df["discounted_price"] = (
        df["actual_price"] * (1 - df["discount_percentage"] / 100)
    ).round(2)
    
    return df


@pytest.fixture
def sample_csv_file(sample_dataframe: pd.DataFrame) -> Generator[Path, None, None]:
    """Create a temporary CSV file with sample data."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        sample_dataframe.to_csv(f, index=False)
        temp_path = Path(f.name)
    
    yield temp_path
    
    # Cleanup
    if temp_path.exists():
        temp_path.unlink()


@pytest.fixture
def temp_model_path() -> Generator[Path, None, None]:
    """Create a temporary path for model saving."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "test_model.joblib"


@pytest.fixture
def trained_predictor(sample_dataframe: pd.DataFrame, temp_model_path: Path):
    """Create a trained predictor for testing."""
    from src.data.preprocessor import prepare_training_data
    from src.ml.predictor import DiscountPredictor
    
    X_train, X_test, y_train, y_test, preprocessor = prepare_training_data(
        sample_dataframe,
        test_size=0.2,
    )
    
    predictor = DiscountPredictor(model_path=temp_model_path)
    predictor.train(
        X_train,
        y_train,
        X_val=X_test,
        y_val=y_test,
        preprocessor=preprocessor,
    )
    
    return predictor


@pytest.fixture
def sample_documents() -> list[dict]:
    """Create sample documents for RAG testing."""
    return [
        {
            "id": "doc1",
            "text": "Sony WH-1000XM5 Wireless Headphones. Category: Electronics. Premium noise cancelling.",
            "metadata": {
                "product_name": "Sony WH-1000XM5",
                "main_category": "Electronics",
                "actual_price": 349.99,
                "rating": 4.8,
            },
        },
        {
            "id": "doc2",
            "text": "Apple AirPods Pro. Category: Electronics. Active noise cancellation, transparency mode.",
            "metadata": {
                "product_name": "Apple AirPods Pro",
                "main_category": "Electronics",
                "actual_price": 249.99,
                "rating": 4.7,
            },
        },
        {
            "id": "doc3",
            "text": "Nike Running Shoes. Category: Sports. Lightweight, breathable, cushioned.",
            "metadata": {
                "product_name": "Nike Air Zoom",
                "main_category": "Sports",
                "actual_price": 129.99,
                "rating": 4.5,
            },
        },
    ]


@pytest.fixture
def legacy_sample_dataframe() -> pd.DataFrame:
    """Create a legacy format sample DataFrame (without enhanced features)."""
    np.random.seed(42)
    n_samples = 50
    
    categories = ["Electronics", "Clothing", "Home & Kitchen", "Books", "Sports"]
    
    data = {
        "product_id": [f"PROD{i:04d}" for i in range(n_samples)],
        "product_name": [f"Product {i}" for i in range(n_samples)],
        "category": np.random.choice(categories, n_samples),
        "actual_price": np.random.uniform(100, 5000, n_samples).round(2),
        "rating": np.random.uniform(2.5, 5.0, n_samples).round(1),
        "rating_count": np.random.randint(10, 10000, n_samples),
        "about_product": [f"Description for product {i}" for i in range(n_samples)],
    }
    
    df = pd.DataFrame(data)
    
    base_discount = 10 + np.random.uniform(-5, 5, n_samples)
    price_factor = (df["actual_price"] / 1000) * 5
    rating_factor = (5 - df["rating"]) * 3
    
    df["discount_percentage"] = np.clip(
        base_discount + price_factor + rating_factor + np.random.normal(0, 5, n_samples),
        0,
        70,
    ).round(2)
    
    df["discounted_price"] = (
        df["actual_price"] * (1 - df["discount_percentage"] / 100)
    ).round(2)
    
    return df
