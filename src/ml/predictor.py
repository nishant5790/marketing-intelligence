"""Discount prediction model using LightGBM."""

from pathlib import Path
from typing import Any, Optional

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import structlog
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import settings
from src.data.preprocessor import FeaturePreprocessor

logger = structlog.get_logger(__name__)


class DiscountPredictor:
    """LightGBM-based discount percentage predictor."""

    def __init__(self, model_path: Optional[Path] = None):
        """Initialize the predictor.
        
        Args:
            model_path: Path to saved model file.
        """
        self.model_path = model_path or settings.model_path
        self.model: Optional[lgb.LGBMRegressor] = None
        self.preprocessor: Optional[FeaturePreprocessor] = None
        self.feature_names: list[str] = []
        self.metrics: dict[str, float] = {}
        self._is_loaded = False

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        preprocessor: Optional[FeaturePreprocessor] = None,
        **kwargs: Any,
    ) -> "DiscountPredictor":
        """Train the LightGBM model.
        
        Args:
            X_train: Training features.
            y_train: Training targets.
            X_val: Validation features (optional).
            y_val: Validation targets (optional).
            preprocessor: Fitted preprocessor instance.
            **kwargs: Additional LightGBM parameters.
            
        Returns:
            Self for method chaining.
        """
        logger.info(
            "training_model",
            train_samples=len(y_train),
            val_samples=len(y_val) if y_val is not None else 0,
        )
        
        # Store preprocessor
        self.preprocessor = preprocessor
        if preprocessor:
            self.feature_names = preprocessor.get_feature_names()
        
        # Default hyperparameters
        params = {
            "objective": "regression",
            "metric": "rmse",
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 6,
            "num_leaves": 31,
            "min_child_samples": 20,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
            "random_state": 42,
            "verbose": -1,
        }
        params.update(kwargs)
        
        # Create and train model
        self.model = lgb.LGBMRegressor(**params)
        
        callbacks = []
        eval_set = None
        
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]
            callbacks.append(lgb.early_stopping(stopping_rounds=10, verbose=False))
        
        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            callbacks=callbacks if callbacks else None,
        )
        
        self._is_loaded = True
        logger.info("model_trained", n_estimators=self.model.n_estimators_)
        
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions.
        
        Args:
            X: Feature array.
            
        Returns:
            Predicted discount percentages.
            
        Raises:
            ValueError: If model hasn't been trained or loaded.
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        predictions = self.model.predict(X)
        
        # Clip predictions to valid range [0, 100]
        predictions = np.clip(predictions, 0, 100)
        
        return predictions

    def predict_from_features(
        self,
        category: str,
        actual_price: float,
        rating: float,
        rating_count: int,
    ) -> dict[str, Any]:
        """Make prediction from raw feature values.
        
        Args:
            category: Product category.
            actual_price: Original product price.
            rating: Product rating (1-5).
            rating_count: Number of ratings.
            
        Returns:
            Dictionary with prediction and confidence.
        """
        if self.model is None or self.preprocessor is None:
            raise ValueError("Model not trained or loaded")
        
        # Determine price tier
        if actual_price <= 500:
            price_tier = "budget"
        elif actual_price <= 2000:
            price_tier = "mid"
        elif actual_price <= 10000:
            price_tier = "premium"
        else:
            price_tier = "luxury"
        
        # Create DataFrame with all required features
        df = pd.DataFrame([{
            "main_category": category,
            "sub_category": category,  # Use same as main for API simplicity
            "price_tier": price_tier,
            "actual_price": actual_price,
            "rating": rating,
            "rating_count": rating_count,
            "category_depth": 3,  # Default depth
            "name_length": 50,  # Default name length
            "description_length": 200,  # Default description length
            "rating_count_log": np.log1p(rating_count),
            "review_sentiment": 0.5,  # Neutral sentiment default
        }])
        
        # Transform features
        X = self.preprocessor.transform(df)
        
        # Make prediction
        prediction = self.predict(X)[0]
        
        # Calculate prediction confidence based on feature importance
        # Higher confidence if features are in typical ranges
        confidence = self._calculate_confidence(actual_price, rating, rating_count)
        
        return {
            "predicted_discount": round(float(prediction), 2),
            "confidence": round(confidence, 2),
            "features_used": self.feature_names,
        }

    def _calculate_confidence(
        self,
        actual_price: float,
        rating: float,
        rating_count: int,
    ) -> float:
        """Calculate prediction confidence based on input validity.
        
        Args:
            actual_price: Original product price.
            rating: Product rating.
            rating_count: Number of ratings.
            
        Returns:
            Confidence score between 0 and 1.
        """
        confidence = 1.0
        
        # Penalize extreme prices
        if actual_price < 10 or actual_price > 100000:
            confidence *= 0.8
        
        # Penalize out-of-range ratings
        if rating < 1 or rating > 5:
            confidence *= 0.7
        
        # Penalize very low rating counts
        if rating_count < 5:
            confidence *= 0.8
        
        return max(0.3, confidence)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict[str, float]:
        """Evaluate model performance.
        
        Args:
            X_test: Test features.
            y_test: True target values.
            
        Returns:
            Dictionary of evaluation metrics.
        """
        y_pred = self.predict(X_test)
        
        self.metrics = {
            "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
            "mae": float(mean_absolute_error(y_test, y_pred)),
            "r2": float(r2_score(y_test, y_pred)),
            "mape": float(np.mean(np.abs((y_test - y_pred) / (y_test + 1e-8))) * 100),
        }
        
        logger.info("model_evaluated", **self.metrics)
        
        return self.metrics

    def get_feature_importance(self) -> dict[str, float]:
        """Get feature importance scores.
        
        Returns:
            Dictionary mapping feature names to importance scores.
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        importance = self.model.feature_importances_
        
        if self.feature_names:
            return dict(zip(self.feature_names, importance.tolist()))
        
        return {f"feature_{i}": float(imp) for i, imp in enumerate(importance)}

    def save(self, path: Optional[Path] = None) -> Path:
        """Save model and preprocessor to disk.
        
        Args:
            path: Path to save to. Defaults to model_path.
            
        Returns:
            Path where model was saved.
        """
        save_path = path or self.model_path
        save_path = Path(save_path)
        
        # Create directory if needed
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save everything together
        data = {
            "model": self.model,
            "preprocessor": self.preprocessor,
            "feature_names": self.feature_names,
            "metrics": self.metrics,
        }
        
        joblib.dump(data, save_path)
        logger.info("model_saved", path=str(save_path))
        
        return save_path

    def load(self, path: Optional[Path] = None) -> "DiscountPredictor":
        """Load model and preprocessor from disk.
        
        Args:
            path: Path to load from. Defaults to model_path.
            
        Returns:
            Self for method chaining.
            
        Raises:
            FileNotFoundError: If model file doesn't exist.
        """
        load_path = path or self.model_path
        load_path = Path(load_path)
        
        if not load_path.exists():
            raise FileNotFoundError(f"Model not found at {load_path}")
        
        data = joblib.load(load_path)
        
        self.model = data["model"]
        self.preprocessor = data["preprocessor"]
        self.feature_names = data.get("feature_names", [])
        self.metrics = data.get("metrics", {})
        self._is_loaded = True
        
        logger.info("model_loaded", path=str(load_path))
        
        return self

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded and ready for predictions."""
        return self._is_loaded and self.model is not None


# Singleton instance for API usage
_predictor_instance: Optional[DiscountPredictor] = None


def get_predictor() -> DiscountPredictor:
    """Get or create the singleton predictor instance.
    
    Returns:
        DiscountPredictor instance.
    """
    global _predictor_instance
    
    if _predictor_instance is None:
        _predictor_instance = DiscountPredictor()
        
        # Try to load existing model
        if _predictor_instance.model_path.exists():
            _predictor_instance.load()
    
    return _predictor_instance
