"""Model training pipeline."""

from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import structlog

from src.config import settings
from src.data.loader import get_sample_data, load_amazon_dataset
from src.data.preprocessor import prepare_training_data
from src.ml.predictor import DiscountPredictor

logger = structlog.get_logger(__name__)


def train_discount_model(
    data_path: Optional[Path] = None,
    model_path: Optional[Path] = None,
    test_size: float = 0.2,
    use_sample_data: bool = False,
) -> tuple[DiscountPredictor, dict]:
    """Train the discount prediction model.
    
    Args:
        data_path: Path to training data CSV.
        model_path: Path to save trained model.
        test_size: Fraction of data for testing.
        use_sample_data: If True, use generated sample data.
        
    Returns:
        Tuple of (trained predictor, training results dict).
    """
    logger.info(
        "starting_training",
        data_path=str(data_path) if data_path else "default",
        use_sample_data=use_sample_data,
    )
    
    # Load data
    if use_sample_data:
        logger.info("using_sample_data")
        df = get_sample_data()
    else:
        df = load_amazon_dataset(data_path or settings.data_path)
    
    # Prepare training data
    X_train, X_test, y_train, y_test, preprocessor = prepare_training_data(
        df,
        target_column="discount_percentage",
        test_size=test_size,
    )
    
    # Create and train model
    predictor = DiscountPredictor(model_path or settings.model_path)
    predictor.train(
        X_train,
        y_train,
        X_val=X_test,
        y_val=y_test,
        preprocessor=preprocessor,
    )
    
    # Evaluate
    metrics = predictor.evaluate(X_test, y_test)
    
    # Get feature importance
    feature_importance = predictor.get_feature_importance()
    
    # Save model
    save_path = predictor.save()
    
    results = {
        "timestamp": datetime.utcnow().isoformat(),
        "training_samples": len(y_train),
        "test_samples": len(y_test),
        "metrics": metrics,
        "feature_importance": feature_importance,
        "model_path": str(save_path),
    }
    
    logger.info("training_completed", **results)
    
    return predictor, results


def retrain_if_needed(
    new_data: pd.DataFrame,
    drift_threshold: float = 0.1,
    min_samples: int = 100,
) -> Optional[tuple[DiscountPredictor, dict]]:
    """Retrain model if drift is detected.
    
    Args:
        new_data: New data to check for drift and potentially train on.
        drift_threshold: Threshold for triggering retraining.
        min_samples: Minimum samples required for retraining.
        
    Returns:
        Tuple of (predictor, results) if retrained, None otherwise.
    """
    from src.ml.drift import detect_drift
    
    if len(new_data) < min_samples:
        logger.info("insufficient_samples_for_retraining", samples=len(new_data))
        return None
    
    # Load current model
    predictor = DiscountPredictor()
    
    if not predictor.model_path.exists():
        logger.info("no_existing_model_training_new")
        return train_discount_model()
    
    predictor.load()
    
    # Check for drift
    drift_detected, drift_score = detect_drift(
        predictor,
        new_data,
        threshold=drift_threshold,
    )
    
    if drift_detected:
        logger.info("drift_detected_retraining", drift_score=drift_score)
        
        # Versioned model path
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        versioned_path = settings.model_path.parent / f"discount_predictor_{timestamp}.joblib"
        
        return train_discount_model(model_path=versioned_path)
    
    logger.info("no_drift_detected", drift_score=drift_score)
    return None


if __name__ == "__main__":
    # CLI training
    import argparse
    
    parser = argparse.ArgumentParser(description="Train discount prediction model")
    parser.add_argument("--data", type=Path, help="Path to training data")
    parser.add_argument("--output", type=Path, help="Path to save model")
    parser.add_argument("--sample", action="store_true", help="Use sample data")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split ratio")
    
    args = parser.parse_args()
    
    predictor, results = train_discount_model(
        data_path=args.data,
        model_path=args.output,
        test_size=args.test_size,
        use_sample_data=args.sample,
    )
    
    print("\nTraining Results:")
    print(f"  RMSE: {results['metrics']['rmse']:.4f}")
    print(f"  MAE:  {results['metrics']['mae']:.4f}")
    print(f"  R²:   {results['metrics']['r2']:.4f}")
    print(f"\nModel saved to: {results['model_path']}")
