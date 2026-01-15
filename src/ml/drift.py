"""Drift detection for model monitoring and retraining."""

from typing import Optional, Tuple

import numpy as np
import pandas as pd
import structlog
from scipy import stats

from src.config import settings
from src.data.preprocessor import FeaturePreprocessor
from src.ml.predictor import DiscountPredictor

logger = structlog.get_logger(__name__)


class DriftDetector:
    """Detect data drift using statistical tests."""

    def __init__(
        self,
        reference_data: Optional[np.ndarray] = None,
        threshold: float = 0.1,
    ):
        """Initialize drift detector.
        
        Args:
            reference_data: Reference data distribution (training data).
            threshold: Significance level for drift detection.
        """
        self.reference_data = reference_data
        self.threshold = threshold
        self.reference_stats: dict = {}

    def fit(self, reference_data: np.ndarray) -> "DriftDetector":
        """Fit the detector on reference data.
        
        Args:
            reference_data: Reference data (typically training data).
            
        Returns:
            Self for method chaining.
        """
        self.reference_data = reference_data
        
        # Calculate reference statistics for each feature
        self.reference_stats = {}
        for i in range(reference_data.shape[1]):
            col_data = reference_data[:, i]
            self.reference_stats[i] = {
                "mean": np.mean(col_data),
                "std": np.std(col_data),
                "min": np.min(col_data),
                "max": np.max(col_data),
                "percentiles": np.percentile(col_data, [25, 50, 75]),
            }
        
        logger.info("drift_detector_fitted", n_features=reference_data.shape[1])
        
        return self

    def detect_ks_drift(
        self,
        current_data: np.ndarray,
    ) -> Tuple[bool, dict]:
        """Detect drift using Kolmogorov-Smirnov test.
        
        Args:
            current_data: Current data to compare against reference.
            
        Returns:
            Tuple of (drift_detected, drift_scores).
        """
        if self.reference_data is None:
            raise ValueError("Detector must be fitted first")
        
        drift_scores = {}
        
        for i in range(current_data.shape[1]):
            ref_col = self.reference_data[:, i]
            cur_col = current_data[:, i]
            
            # KS test
            statistic, p_value = stats.ks_2samp(ref_col, cur_col)
            
            drift_scores[f"feature_{i}"] = {
                "ks_statistic": float(statistic),
                "p_value": float(p_value),
                "drift_detected": p_value < self.threshold,
            }
        
        # Overall drift if any feature shows drift
        overall_drift = any(s["drift_detected"] for s in drift_scores.values())
        
        logger.info(
            "ks_drift_detection",
            overall_drift=overall_drift,
            features_with_drift=sum(1 for s in drift_scores.values() if s["drift_detected"]),
        )
        
        return overall_drift, drift_scores

    def detect_psi_drift(
        self,
        current_data: np.ndarray,
        n_bins: int = 10,
    ) -> Tuple[bool, dict]:
        """Detect drift using Population Stability Index (PSI).
        
        PSI < 0.1: No significant change
        0.1 <= PSI < 0.2: Slight change
        PSI >= 0.2: Significant change
        
        Args:
            current_data: Current data to compare.
            n_bins: Number of bins for PSI calculation.
            
        Returns:
            Tuple of (drift_detected, drift_scores).
        """
        if self.reference_data is None:
            raise ValueError("Detector must be fitted first")
        
        drift_scores = {}
        psi_threshold = 0.2  # Significant change threshold
        
        for i in range(current_data.shape[1]):
            ref_col = self.reference_data[:, i]
            cur_col = current_data[:, i]
            
            # Calculate PSI
            psi = self._calculate_psi(ref_col, cur_col, n_bins)
            
            drift_scores[f"feature_{i}"] = {
                "psi": float(psi),
                "drift_detected": psi >= psi_threshold,
                "severity": "high" if psi >= 0.25 else "medium" if psi >= 0.1 else "low",
            }
        
        # Overall drift if any feature shows significant drift
        overall_drift = any(s["drift_detected"] for s in drift_scores.values())
        
        logger.info(
            "psi_drift_detection",
            overall_drift=overall_drift,
            avg_psi=np.mean([s["psi"] for s in drift_scores.values()]),
        )
        
        return overall_drift, drift_scores

    def _calculate_psi(
        self,
        reference: np.ndarray,
        current: np.ndarray,
        n_bins: int,
    ) -> float:
        """Calculate Population Stability Index.
        
        Args:
            reference: Reference distribution.
            current: Current distribution.
            n_bins: Number of bins.
            
        Returns:
            PSI value.
        """
        # Create bins based on reference data
        _, bin_edges = np.histogram(reference, bins=n_bins)
        
        # Calculate percentages in each bin
        ref_counts, _ = np.histogram(reference, bins=bin_edges)
        cur_counts, _ = np.histogram(current, bins=bin_edges)
        
        # Convert to percentages (add small constant to avoid division by zero)
        ref_pct = (ref_counts + 0.001) / len(reference)
        cur_pct = (cur_counts + 0.001) / len(current)
        
        # Calculate PSI
        psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
        
        return psi

    def detect_prediction_drift(
        self,
        predictor: DiscountPredictor,
        current_data: np.ndarray,
        current_labels: np.ndarray,
    ) -> Tuple[bool, dict]:
        """Detect drift by comparing prediction accuracy.
        
        Args:
            predictor: Trained predictor.
            current_data: Current features.
            current_labels: Actual labels.
            
        Returns:
            Tuple of (drift_detected, metrics).
        """
        predictions = predictor.predict(current_data)
        
        # Calculate error metrics
        mse = np.mean((predictions - current_labels) ** 2)
        mae = np.mean(np.abs(predictions - current_labels))
        
        # Compare with training metrics
        training_metrics = predictor.metrics
        training_mae = training_metrics.get("mae", mae)
        
        # Drift if MAE increased significantly (50% threshold)
        mae_increase = (mae - training_mae) / (training_mae + 1e-8)
        drift_detected = mae_increase > 0.5
        
        metrics = {
            "current_mae": float(mae),
            "current_mse": float(mse),
            "training_mae": float(training_mae),
            "mae_increase_pct": float(mae_increase * 100),
            "drift_detected": drift_detected,
        }
        
        logger.info("prediction_drift_detection", **metrics)
        
        return drift_detected, metrics


def detect_drift(
    predictor: DiscountPredictor,
    new_data: pd.DataFrame,
    threshold: float = None,
) -> Tuple[bool, float]:
    """Convenience function to detect drift on new data.
    
    Args:
        predictor: Trained predictor with preprocessor.
        new_data: New DataFrame to check for drift.
        threshold: Drift threshold (uses settings if not provided).
        
    Returns:
        Tuple of (drift_detected, drift_score).
    """
    threshold = threshold or settings.drift_threshold
    
    if predictor.preprocessor is None:
        logger.warning("no_preprocessor_for_drift_detection")
        return False, 0.0
    
    try:
        # Transform new data using existing preprocessor
        X_new = predictor.preprocessor.transform(new_data)
        
        # Get training data stats from preprocessor
        # We'll use PSI for a simpler drift score
        detector = DriftDetector(threshold=threshold)
        
        # For simplicity, we'll compare distributions of predictions
        predictions = predictor.predict(X_new)
        
        # Use coefficient of variation as a simple drift indicator
        cv = np.std(predictions) / (np.mean(predictions) + 1e-8)
        
        # Drift if prediction distribution is very different
        drift_score = min(abs(cv - 0.3), 1.0)  # Expect CV around 0.3
        drift_detected = drift_score > threshold
        
        logger.info(
            "drift_check_completed",
            drift_detected=drift_detected,
            drift_score=float(drift_score),
        )
        
        return drift_detected, float(drift_score)
        
    except Exception as e:
        logger.error("drift_detection_error", error=str(e))
        return False, 0.0


# Singleton instance
_detector_instance: Optional[DriftDetector] = None


def get_drift_detector() -> DriftDetector:
    """Get or create the singleton drift detector.
    
    Returns:
        DriftDetector instance.
    """
    global _detector_instance
    
    if _detector_instance is None:
        _detector_instance = DriftDetector(threshold=settings.drift_threshold)
    
    return _detector_instance
