"""Unit tests for data preprocessing."""

import numpy as np
import pandas as pd
import pytest

from src.data.preprocessor import (
    FeaturePreprocessor,
    prepare_rag_documents,
    prepare_training_data,
    get_sentiment_score,
    compute_feature_correlations,
)


class TestSentimentAnalysis:
    """Tests for sentiment analysis functions."""

    def test_positive_sentiment(self):
        text = "This is a great product! I love it! Amazing quality!"
        score = get_sentiment_score(text)
        assert score > 0

    def test_negative_sentiment(self):
        text = "Terrible product. Worst purchase ever. Very bad quality."
        score = get_sentiment_score(text)
        assert score < 0

    def test_neutral_sentiment(self):
        text = "This is a product."
        score = get_sentiment_score(text)
        assert -0.5 <= score <= 0.5

    def test_empty_text(self):
        assert get_sentiment_score("") == 0.0
        assert get_sentiment_score(None) == 0.0


class TestFeaturePreprocessor:
    """Tests for FeaturePreprocessor class."""

    def test_fit_creates_encoders(self, sample_dataframe: pd.DataFrame):
        preprocessor = FeaturePreprocessor()
        preprocessor.fit(sample_dataframe)
        
        assert preprocessor._fitted
        # Check for enhanced categorical columns
        assert "main_category" in preprocessor.label_encoders or "price_tier" in preprocessor.label_encoders
        assert preprocessor.scaler is not None

    def test_transform_without_fit_raises(self, sample_dataframe: pd.DataFrame):
        preprocessor = FeaturePreprocessor()
        
        with pytest.raises(ValueError, match="must be fitted"):
            preprocessor.transform(sample_dataframe)

    def test_fit_transform(self, sample_dataframe: pd.DataFrame):
        preprocessor = FeaturePreprocessor()
        X = preprocessor.fit_transform(sample_dataframe)
        
        assert isinstance(X, np.ndarray)
        assert len(X) == len(sample_dataframe)
        assert X.ndim == 2

    def test_transform_output_has_features(self, sample_dataframe: pd.DataFrame):
        preprocessor = FeaturePreprocessor()
        X = preprocessor.fit_transform(sample_dataframe)
        
        # Should have multiple features (categorical + numerical)
        assert X.shape[1] > 3

    def test_handles_missing_values(self, sample_dataframe: pd.DataFrame):
        df = sample_dataframe.copy()
        df.loc[0, "rating"] = np.nan
        df.loc[1, "main_category"] = np.nan
        
        preprocessor = FeaturePreprocessor()
        X = preprocessor.fit_transform(df)
        
        # Should not have any NaN values
        assert not np.isnan(X).any()

    def test_handles_unseen_categories(self, sample_dataframe: pd.DataFrame):
        preprocessor = FeaturePreprocessor()
        preprocessor.fit(sample_dataframe)
        
        # Create test data with unseen category
        test_df = pd.DataFrame([{
            "main_category": "NewCategory",
            "sub_category": "NewSubCategory",
            "price_tier": "super_luxury",
            "actual_price": 100.0,
            "rating": 4.0,
            "rating_count": 100,
            "category_depth": 3,
            "name_length": 50,
            "description_length": 200,
            "rating_count_log": np.log1p(100),
            "review_sentiment": 0.5,
        }])
        
        # Should not raise error
        X = preprocessor.transform(test_df)
        assert X.shape[0] == 1

    def test_get_feature_names(self, sample_dataframe: pd.DataFrame):
        preprocessor = FeaturePreprocessor()
        preprocessor.fit(sample_dataframe)
        
        names = preprocessor.get_feature_names()
        
        # Should have both categorical and numerical features
        assert len(names) > 0

    def test_get_feature_info(self, sample_dataframe: pd.DataFrame):
        preprocessor = FeaturePreprocessor()
        preprocessor.fit(sample_dataframe)
        
        info = preprocessor.get_feature_info()
        
        assert "categorical_features" in info
        assert "numerical_features" in info
        assert "total_features" in info
        assert "encoder_classes" in info


class TestPrepareTrainingData:
    """Tests for prepare_training_data function."""

    def test_split_sizes(self, sample_dataframe: pd.DataFrame):
        X_train, X_test, y_train, y_test, _ = prepare_training_data(
            sample_dataframe,
            test_size=0.2,
        )
        
        total = len(sample_dataframe)
        assert len(y_train) == pytest.approx(total * 0.8, abs=2)
        assert len(y_test) == pytest.approx(total * 0.2, abs=2)

    def test_returns_preprocessor(self, sample_dataframe: pd.DataFrame):
        _, _, _, _, preprocessor = prepare_training_data(sample_dataframe)
        
        assert isinstance(preprocessor, FeaturePreprocessor)
        assert preprocessor._fitted

    def test_missing_target_raises(self, sample_dataframe: pd.DataFrame):
        with pytest.raises(ValueError, match="not found"):
            prepare_training_data(sample_dataframe, target_column="nonexistent")

    def test_reproducibility(self, sample_dataframe: pd.DataFrame):
        X1, _, y1, _, _ = prepare_training_data(
            sample_dataframe, random_state=42
        )
        X2, _, y2, _, _ = prepare_training_data(
            sample_dataframe, random_state=42
        )
        
        np.testing.assert_array_equal(y1, y2)


class TestPrepareRagDocuments:
    """Tests for prepare_rag_documents function."""

    def test_document_structure(self, sample_dataframe: pd.DataFrame):
        documents = prepare_rag_documents(sample_dataframe)
        
        assert len(documents) == len(sample_dataframe)
        
        for doc in documents:
            assert "id" in doc
            assert "text" in doc
            assert "metadata" in doc

    def test_document_text_content(self, sample_dataframe: pd.DataFrame):
        documents = prepare_rag_documents(sample_dataframe)
        
        doc = documents[0]
        assert "Product:" in doc["text"]

    def test_metadata_fields(self, sample_dataframe: pd.DataFrame):
        documents = prepare_rag_documents(sample_dataframe)
        
        metadata = documents[0]["metadata"]
        
        assert "product_id" in metadata
        assert "product_name" in metadata

    def test_metadata_includes_enhanced_fields(self, sample_dataframe: pd.DataFrame):
        documents = prepare_rag_documents(sample_dataframe)
        
        metadata = documents[0]["metadata"]
        
        # Check for enhanced fields if they exist
        if "main_category" in sample_dataframe.columns:
            assert "main_category" in metadata


class TestComputeFeatureCorrelations:
    """Tests for compute_feature_correlations function."""

    def test_correlation_matrix(self, sample_dataframe: pd.DataFrame):
        corr_matrix = compute_feature_correlations(sample_dataframe)
        
        assert isinstance(corr_matrix, pd.DataFrame)
        
        # Should be square matrix
        if len(corr_matrix) > 0:
            assert corr_matrix.shape[0] == corr_matrix.shape[1]

    def test_correlation_values_range(self, sample_dataframe: pd.DataFrame):
        corr_matrix = compute_feature_correlations(sample_dataframe)
        
        if len(corr_matrix) > 0:
            # Correlations should be between -1 and 1
            assert (corr_matrix >= -1).all().all()
            assert (corr_matrix <= 1).all().all()

    def test_diagonal_is_one(self, sample_dataframe: pd.DataFrame):
        corr_matrix = compute_feature_correlations(sample_dataframe)
        
        if len(corr_matrix) > 0:
            # Diagonal should be 1 (self-correlation)
            for col in corr_matrix.columns:
                assert corr_matrix.loc[col, col] == pytest.approx(1.0)
