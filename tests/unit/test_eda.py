"""Unit tests for EDA module."""

import numpy as np
import pandas as pd
import pytest

from src.analysis.eda import DatasetAnalyzer, analyze_dataset


@pytest.fixture
def sample_df():
    """Create sample DataFrame for EDA testing."""
    np.random.seed(42)
    n = 50
    
    return pd.DataFrame({
        "product_id": [f"PROD{i:04d}" for i in range(n)],
        "product_name": [f"Test Product {i}" for i in range(n)],
        "main_category": np.random.choice(["Electronics", "Clothing", "Home"], n),
        "sub_category": np.random.choice(["Gadgets", "Apparel", "Furniture"], n),
        "category_depth": np.random.randint(2, 6, n),
        "actual_price": np.random.uniform(100, 5000, n),
        "discounted_price": np.random.uniform(50, 4000, n),
        "discount_percentage": np.random.uniform(10, 70, n),
        "rating": np.random.uniform(2.5, 5.0, n),
        "rating_count": np.random.randint(10, 10000, n),
        "name_length": np.random.randint(20, 100, n),
        "description_length": np.random.randint(100, 1000, n),
        "price_tier": np.random.choice(["budget", "mid", "premium", "luxury"], n),
        "rating_count_log": np.log1p(np.random.randint(10, 10000, n)),
    })


class TestDatasetAnalyzer:
    """Tests for DatasetAnalyzer class."""

    def test_init_with_df(self, sample_df):
        analyzer = DatasetAnalyzer(sample_df)
        assert len(analyzer.df) == 50

    def test_get_summary_structure(self, sample_df):
        analyzer = DatasetAnalyzer(sample_df)
        summary = analyzer.get_summary()
        
        assert "dataset_info" in summary
        assert "categories" in summary
        assert "price_analysis" in summary
        assert "rating_analysis" in summary
        assert "discount_analysis" in summary
        assert "missing_values" in summary
        assert "correlations" in summary

    def test_dataset_info(self, sample_df):
        analyzer = DatasetAnalyzer(sample_df)
        summary = analyzer.get_summary()
        
        info = summary["dataset_info"]
        assert info["total_products"] == 50
        assert info["total_features"] == len(sample_df.columns)
        assert "features" in info

    def test_category_analysis(self, sample_df):
        analyzer = DatasetAnalyzer(sample_df)
        summary = analyzer.get_summary()
        
        categories = summary["categories"]
        assert "main_categories" in categories
        assert "sub_categories" in categories
        assert "price_tiers" in categories

    def test_price_analysis(self, sample_df):
        analyzer = DatasetAnalyzer(sample_df)
        summary = analyzer.get_summary()
        
        prices = summary["price_analysis"]
        assert "actual_price" in prices
        
        price_stats = prices["actual_price"]
        assert "min" in price_stats
        assert "max" in price_stats
        assert "mean" in price_stats
        assert "median" in price_stats
        assert "std" in price_stats

    def test_rating_analysis(self, sample_df):
        analyzer = DatasetAnalyzer(sample_df)
        summary = analyzer.get_summary()
        
        ratings = summary["rating_analysis"]
        assert "rating_stats" in ratings
        assert "rating_count_stats" in ratings

    def test_discount_analysis(self, sample_df):
        analyzer = DatasetAnalyzer(sample_df)
        summary = analyzer.get_summary()
        
        discounts = summary["discount_analysis"]
        assert "discount_stats" in discounts
        assert "discount_distribution" in discounts

    def test_missing_values(self, sample_df):
        # Add some missing values
        df = sample_df.copy()
        df.loc[0:5, "rating"] = np.nan
        
        analyzer = DatasetAnalyzer(df)
        summary = analyzer.get_summary()
        
        missing = summary["missing_values"]
        assert "counts" in missing
        assert "percentages" in missing
        assert "complete_rows" in missing

    def test_correlations(self, sample_df):
        analyzer = DatasetAnalyzer(sample_df)
        summary = analyzer.get_summary()
        
        correlations = summary["correlations"]
        assert "correlation_matrix" in correlations
        assert "correlations_with_discount" in correlations

    def test_get_top_products(self, sample_df):
        analyzer = DatasetAnalyzer(sample_df)
        
        top = analyzer.get_top_products(by="rating", n=5)
        
        assert len(top) == 5
        assert all("product_id" in p for p in top)
        assert all("rating" in p for p in top)

    def test_get_top_products_with_category_filter(self, sample_df):
        analyzer = DatasetAnalyzer(sample_df)
        
        top = analyzer.get_top_products(by="rating", n=5, category="Electronics")
        
        assert len(top) <= 5
        assert all(p["main_category"] == "Electronics" for p in top)

    def test_get_category_summary(self, sample_df):
        analyzer = DatasetAnalyzer(sample_df)
        
        summary = analyzer.get_category_summary("Electronics")
        
        assert summary["category"] == "Electronics"
        assert "product_count" in summary
        assert "avg_price" in summary
        assert "avg_discount" in summary
        assert "avg_rating" in summary

    def test_get_category_summary_not_found(self, sample_df):
        analyzer = DatasetAnalyzer(sample_df)
        
        summary = analyzer.get_category_summary("NonExistent")
        
        assert "error" in summary

    def test_summary_caching(self, sample_df):
        analyzer = DatasetAnalyzer(sample_df)
        
        summary1 = analyzer.get_summary()
        summary2 = analyzer.get_summary()  # Should use cache
        
        assert summary1 is summary2  # Same object (cached)
        
        summary3 = analyzer.get_summary(force_refresh=True)  # Force refresh
        
        assert summary1 is not summary3  # Different object


class TestAnalyzeDataset:
    """Tests for analyze_dataset convenience function."""

    def test_analyze_dataset_with_df(self, sample_df):
        summary = analyze_dataset(sample_df)
        
        assert "dataset_info" in summary
        assert summary["dataset_info"]["total_products"] == 50

    def test_analyze_dataset_returns_dict(self, sample_df):
        result = analyze_dataset(sample_df)
        assert isinstance(result, dict)
