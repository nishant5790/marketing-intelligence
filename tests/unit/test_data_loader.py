"""Unit tests for data loading utilities."""

from pathlib import Path

import pandas as pd
import pytest

from src.data.loader import (
    clean_discount,
    clean_price,
    clean_rating,
    clean_rating_count,
    get_sample_data,
    get_text_length,
    load_amazon_dataset,
    parse_hierarchical_category,
    count_reviews,
    get_dataset_summary,
)


class TestCleanPrice:
    """Tests for clean_price function."""

    def test_clean_price_with_rupees(self):
        assert clean_price("₹1,299") == 1299.0

    def test_clean_price_with_dollars(self):
        assert clean_price("$99.99") == 99.99

    def test_clean_price_with_commas(self):
        assert clean_price("1,234,567") == 1234567.0

    def test_clean_price_plain_number(self):
        assert clean_price("499.99") == 499.99

    def test_clean_price_none(self):
        assert clean_price(None) is None

    def test_clean_price_invalid(self):
        assert clean_price("not a price") is None


class TestCleanRating:
    """Tests for clean_rating function."""

    def test_clean_rating_full_format(self):
        assert clean_rating("4.5 out of 5 stars") == 4.5

    def test_clean_rating_simple(self):
        assert clean_rating("4.2") == 4.2

    def test_clean_rating_integer(self):
        assert clean_rating("5") == 5.0

    def test_clean_rating_none(self):
        assert clean_rating(None) is None


class TestCleanRatingCount:
    """Tests for clean_rating_count function."""

    def test_clean_rating_count_with_text(self):
        assert clean_rating_count("1,234 ratings") == 1234

    def test_clean_rating_count_plain(self):
        assert clean_rating_count("5678") == 5678

    def test_clean_rating_count_with_commas(self):
        assert clean_rating_count("10,000") == 10000

    def test_clean_rating_count_none(self):
        assert clean_rating_count(None) is None


class TestCleanDiscount:
    """Tests for clean_discount function."""

    def test_clean_discount_percentage(self):
        assert clean_discount("50%") == 50.0

    def test_clean_discount_plain(self):
        assert clean_discount("25") == 25.0

    def test_clean_discount_with_text(self):
        assert clean_discount("Save 30%") == 30.0

    def test_clean_discount_none(self):
        assert clean_discount(None) is None


class TestParseHierarchicalCategory:
    """Tests for parse_hierarchical_category function."""

    def test_parse_full_hierarchy(self):
        category = "Computers&Accessories|Cables|USBCables"
        main, sub, depth = parse_hierarchical_category(category)
        
        assert main == "Computers&Accessories"
        assert sub == "Cables"
        assert depth == 3

    def test_parse_single_category(self):
        category = "Electronics"
        main, sub, depth = parse_hierarchical_category(category)
        
        assert main == "Electronics"
        assert sub == "Electronics"  # Same as main when no sub
        assert depth == 1

    def test_parse_none(self):
        main, sub, depth = parse_hierarchical_category(None)
        
        assert main == "Unknown"
        assert sub == "Unknown"
        assert depth == 0

    def test_parse_empty(self):
        main, sub, depth = parse_hierarchical_category("")
        
        assert main == "Unknown"
        assert sub == "Unknown"
        assert depth == 0


class TestGetTextLength:
    """Tests for get_text_length function."""

    def test_text_length_normal(self):
        assert get_text_length("Hello World") == 11

    def test_text_length_empty(self):
        assert get_text_length("") == 0

    def test_text_length_none(self):
        assert get_text_length(None) == 0


class TestCountReviews:
    """Tests for count_reviews function."""

    def test_count_reviews_multiple(self):
        reviews = "Great product, Good quality, Highly recommended"
        assert count_reviews(reviews) == 3

    def test_count_reviews_single(self):
        assert count_reviews("Great product") == 1

    def test_count_reviews_none(self):
        assert count_reviews(None) == 0


class TestLoadAmazonDataset:
    """Tests for load_amazon_dataset function."""

    def test_load_dataset_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_amazon_dataset(Path("/nonexistent/path.csv"))

    def test_load_dataset_success(self, sample_csv_file: Path):
        df = load_amazon_dataset(sample_csv_file)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert "actual_price" in df.columns
        assert "discount_percentage" in df.columns

    def test_load_dataset_adds_enhanced_features(self, sample_csv_file: Path):
        df = load_amazon_dataset(sample_csv_file)
        
        # Check enhanced features are added
        assert "main_category" in df.columns
        assert "sub_category" in df.columns
        assert "category_depth" in df.columns
        assert "name_length" in df.columns
        assert "price_tier" in df.columns
        assert "rating_count_log" in df.columns


class TestGetSampleData:
    """Tests for get_sample_data function."""

    def test_sample_data_structure(self):
        df = get_sample_data()
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 100
        
        required_columns = [
            "product_id",
            "product_name",
            "main_category",
            "sub_category",
            "actual_price",
            "discount_percentage",
            "rating",
            "rating_count",
            "price_tier",
            "category_depth",
        ]
        
        for col in required_columns:
            assert col in df.columns

    def test_sample_data_values(self):
        df = get_sample_data()
        
        # Check value ranges
        assert (df["rating"] >= 2.5).all()
        assert (df["rating"] <= 5.0).all()
        assert (df["discount_percentage"] >= 0).all()
        assert (df["discount_percentage"] <= 70).all()
        assert (df["actual_price"] > 0).all()

    def test_sample_data_reproducibility(self):
        df1 = get_sample_data()
        df2 = get_sample_data()
        
        # Should produce identical data due to fixed seed
        pd.testing.assert_frame_equal(df1, df2)


class TestGetDatasetSummary:
    """Tests for get_dataset_summary function."""

    def test_summary_structure(self, sample_dataframe):
        summary = get_dataset_summary(sample_dataframe)
        
        assert "total_products" in summary
        assert "total_features" in summary
        assert "features" in summary

    def test_summary_numerical_stats(self, sample_dataframe):
        summary = get_dataset_summary(sample_dataframe)
        
        if "actual_price_stats" in summary:
            stats = summary["actual_price_stats"]
            assert "min" in stats
            assert "max" in stats
            assert "mean" in stats

    def test_summary_categories(self, sample_dataframe):
        summary = get_dataset_summary(sample_dataframe)
        
        if "main_categories" in summary:
            assert summary["main_categories"] > 0
