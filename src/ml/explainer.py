"""SHAP-based model explainability."""

from typing import Any, Optional

import numpy as np
import pandas as pd
import shap
import structlog

from src.ml.predictor import DiscountPredictor

logger = structlog.get_logger(__name__)


class ShapExplainer:
    """SHAP-based explainer for the discount prediction model."""

    def __init__(self, predictor: DiscountPredictor):
        """Initialize the explainer.
        
        Args:
            predictor: Trained DiscountPredictor instance.
        """
        self.predictor = predictor
        self._explainer: Optional[shap.TreeExplainer] = None
        self._background_data: Optional[np.ndarray] = None

    def _create_explainer(
        self,
        background_data: Optional[np.ndarray] = None,
    ) -> shap.TreeExplainer:
        """Create SHAP TreeExplainer.
        
        Args:
            background_data: Background data for SHAP calculations.
            
        Returns:
            Configured SHAP explainer.
        """
        if self.predictor.model is None:
            raise ValueError("Model must be trained before creating explainer")
        
        if self._explainer is None:
            logger.info("creating_shap_explainer")
            
            # TreeExplainer is efficient for tree-based models like LightGBM
            self._explainer = shap.TreeExplainer(
                self.predictor.model,
                feature_perturbation="tree_path_dependent",
            )
            
            if background_data is not None:
                self._background_data = background_data
        
        return self._explainer

    def explain(
        self,
        X: np.ndarray,
        feature_names: Optional[list[str]] = None,
    ) -> dict:
        """Generate SHAP explanation for predictions.
        
        Args:
            X: Feature array to explain (single sample or batch).
            feature_names: Names of features.
            
        Returns:
            Dictionary with SHAP values and explanation.
        """
        explainer = self._create_explainer()
        
        # Ensure X is 2D
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        # Calculate SHAP values
        shap_values = explainer.shap_values(X)
        
        # Get feature names
        if feature_names is None:
            feature_names = self.predictor.feature_names or [
                f"feature_{i}" for i in range(X.shape[1])
            ]
        
        # Build explanation for first sample
        explanation = {
            "base_value": float(explainer.expected_value),
            "shap_values": [],
            "prediction": float(self.predictor.predict(X)[0]),
        }
        
        for i, (name, value, shap_val) in enumerate(zip(
            feature_names, X[0], shap_values[0]
        )):
            explanation["shap_values"].append({
                "feature_name": name,
                "value": float(value),
                "shap_value": float(shap_val),
                "contribution": "positive" if shap_val > 0 else "negative",
            })
        
        # Sort by absolute SHAP value
        explanation["shap_values"].sort(
            key=lambda x: abs(x["shap_value"]),
            reverse=True,
        )
        
        return explanation

    def get_feature_importance(self, X: np.ndarray) -> dict[str, float]:
        """Get global feature importance using SHAP.
        
        Args:
            X: Feature array for computing importance.
            
        Returns:
            Dictionary of feature importance scores.
        """
        explainer = self._create_explainer()
        shap_values = explainer.shap_values(X)
        
        # Mean absolute SHAP value per feature
        importance = np.mean(np.abs(shap_values), axis=0)
        
        feature_names = self.predictor.feature_names or [
            f"feature_{i}" for i in range(X.shape[1])
        ]
        
        return dict(zip(feature_names, importance.tolist()))

    def generate_summary(self, explanation: dict) -> str:
        """Generate human-readable explanation summary.
        
        Args:
            explanation: SHAP explanation dictionary.
            
        Returns:
            Text summary of the explanation.
        """
        prediction = explanation["prediction"]
        base_value = explanation["base_value"]
        shap_values = explanation["shap_values"]
        
        summary_parts = [
            f"The predicted discount is {prediction:.1f}%.",
            f"The baseline (average) prediction is {base_value:.1f}%.",
            "\nKey factors influencing this prediction:",
        ]
        
        for sv in shap_values[:4]:  # Top 4 factors
            direction = "increases" if sv["shap_value"] > 0 else "decreases"
            impact = abs(sv["shap_value"])
            summary_parts.append(
                f"  - {sv['feature_name']} ({sv['value']:.2f}) {direction} "
                f"the predicted discount by {impact:.1f}%"
            )
        
        return "\n".join(summary_parts)


def explain_prediction(
    predictor: DiscountPredictor,
    category: str,
    actual_price: float,
    rating: float,
    rating_count: int,
) -> dict[str, Any]:
    """Generate SHAP explanation for a single prediction.
    
    Args:
        predictor: Trained predictor.
        category: Product category.
        actual_price: Product price.
        rating: Product rating.
        rating_count: Number of ratings.
        
    Returns:
        Explanation dictionary for API response.
    """
    if predictor.preprocessor is None:
        raise ValueError("Predictor must have a fitted preprocessor")
    
    # Create DataFrame with input
    df = pd.DataFrame([{
        "category": category,
        "actual_price": actual_price,
        "rating": rating,
        "rating_count": rating_count,
    }])
    
    # Transform features
    X = predictor.preprocessor.transform(df)
    
    # Create explainer and get explanation
    explainer = ShapExplainer(predictor)
    explanation = explainer.explain(X, predictor.feature_names)
    
    # Generate summary
    summary = explainer.generate_summary(explanation)
    
    return {
        "predicted_discount": explanation["prediction"],
        "shap_values": explanation["shap_values"],
        "base_value": explanation["base_value"],
        "explanation_summary": summary,
    }


def plot_explanation(
    explanation: dict,
    output_path: Optional[str] = None,
) -> Optional[Any]:
    """Create SHAP waterfall plot.
    
    Args:
        explanation: SHAP explanation dictionary.
        output_path: Path to save plot (optional).
        
    Returns:
        Matplotlib figure if output_path is None.
    """
    try:
        import matplotlib.pyplot as plt
        
        feature_names = [sv["feature_name"] for sv in explanation["shap_values"]]
        shap_vals = [sv["shap_value"] for sv in explanation["shap_values"]]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = ["green" if v > 0 else "red" for v in shap_vals]
        y_pos = np.arange(len(feature_names))
        
        ax.barh(y_pos, shap_vals, color=colors, alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(feature_names)
        ax.set_xlabel("SHAP Value (Impact on Prediction)")
        ax.set_title("Feature Contributions to Discount Prediction")
        ax.axvline(x=0, color="black", linestyle="-", linewidth=0.5)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close()
            return None
        
        return fig
        
    except ImportError:
        logger.warning("matplotlib_not_available_for_plotting")
        return None
