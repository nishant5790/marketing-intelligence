"""Unit tests for discount prediction model."""

import numpy as np
import pandas as pd
import pytest

from src.ml.predictor import DiscountPredictor


class TestDiscountPredictor:
    """Tests for DiscountPredictor class."""

    def test_train_creates_model(self, sample_dataframe: pd.DataFrame, temp_model_path):
        from src.data.preprocessor import prepare_training_data
        
        X_train, X_test, y_train, y_test, preprocessor = prepare_training_data(
            sample_dataframe
        )
        
        predictor = DiscountPredictor(model_path=temp_model_path)
        predictor.train(X_train, y_train, preprocessor=preprocessor)
        
        assert predictor.model is not None
        assert predictor.is_loaded

    def test_predict_returns_array(self, trained_predictor, sample_dataframe):
        from src.data.preprocessor import prepare_training_data
        
        X_train, X_test, y_train, y_test, preprocessor = prepare_training_data(
            sample_dataframe
        )
        
        predictions = trained_predictor.predict(X_test)
        
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(y_test)

    def test_predict_clips_values(self, trained_predictor, sample_dataframe):
        from src.data.preprocessor import prepare_training_data
        
        X_train, X_test, y_train, y_test, preprocessor = prepare_training_data(
            sample_dataframe
        )
        
        predictions = trained_predictor.predict(X_test)
        
        # Predictions should be clipped to [0, 100]
        assert (predictions >= 0).all()
        assert (predictions <= 100).all()

    def test_predict_without_model_raises(self, temp_model_path):
        predictor = DiscountPredictor(model_path=temp_model_path)
        
        with pytest.raises(ValueError, match="not trained"):
            predictor.predict(np.array([[1, 2, 3, 4]]))

    def test_predict_from_features(self, trained_predictor):
        result = trained_predictor.predict_from_features(
            category="Electronics",
            actual_price=999.99,
            rating=4.5,
            rating_count=500,
        )
        
        assert "predicted_discount" in result
        assert "confidence" in result
        assert 0 <= result["predicted_discount"] <= 100
        assert 0 <= result["confidence"] <= 1

    def test_evaluate_returns_metrics(self, trained_predictor, sample_dataframe):
        from src.data.preprocessor import prepare_training_data
        
        _, X_test, _, y_test, _ = prepare_training_data(sample_dataframe)
        
        metrics = trained_predictor.evaluate(X_test, y_test)
        
        assert "rmse" in metrics
        assert "mae" in metrics
        assert "r2" in metrics
        assert metrics["rmse"] >= 0
        assert metrics["mae"] >= 0

    def test_feature_importance(self, trained_predictor):
        importance = trained_predictor.get_feature_importance()
        
        assert isinstance(importance, dict)
        assert len(importance) > 0
        assert all(v >= 0 for v in importance.values())

    def test_save_and_load(self, trained_predictor, temp_model_path):
        # Save
        save_path = trained_predictor.save(temp_model_path)
        assert save_path.exists()
        
        # Load into new instance
        new_predictor = DiscountPredictor(model_path=temp_model_path)
        new_predictor.load()
        
        assert new_predictor.is_loaded
        assert new_predictor.model is not None

    def test_load_nonexistent_raises(self, temp_model_path):
        predictor = DiscountPredictor(model_path=temp_model_path)
        
        with pytest.raises(FileNotFoundError):
            predictor.load()

    def test_confidence_calculation(self, trained_predictor):
        # Normal values should have high confidence
        result_normal = trained_predictor.predict_from_features(
            category="Electronics",
            actual_price=500,
            rating=4.0,
            rating_count=100,
        )
        
        # Extreme values should have lower confidence
        result_extreme = trained_predictor.predict_from_features(
            category="Electronics",
            actual_price=5,  # Very low price
            rating=6.0,  # Invalid rating
            rating_count=2,  # Very low count
        )
        
        assert result_normal["confidence"] >= result_extreme["confidence"]


class TestModelMetrics:
    """Tests for model evaluation metrics."""

    def test_rmse_calculation(self, trained_predictor, sample_dataframe):
        from src.data.preprocessor import prepare_training_data
        
        _, X_test, _, y_test, _ = prepare_training_data(sample_dataframe)
        metrics = trained_predictor.evaluate(X_test, y_test)
        
        # RMSE should be reasonable for discount percentage
        assert metrics["rmse"] < 50  # Assuming reasonable model

    def test_r2_range(self, trained_predictor, sample_dataframe):
        from src.data.preprocessor import prepare_training_data
        
        _, X_test, _, y_test, _ = prepare_training_data(sample_dataframe)
        metrics = trained_predictor.evaluate(X_test, y_test)
        
        # R² should be between -inf and 1
        assert metrics["r2"] <= 1
