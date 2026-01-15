"""Prediction API routes."""

from fastapi import APIRouter, HTTPException
import structlog

from src.api.schemas import (
    ExplainRequest,
    ExplainResponse,
    FeatureExplanation,
    ModelStatusResponse,
    PredictDiscountRequest,
    PredictDiscountResponse,
    ShapValues,
    TrainModelRequest,
    TrainModelResponse,
)
from src.config import settings
from src.ml.predictor import get_predictor

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/predict", tags=["Prediction"])


@router.post("/discount", response_model=PredictDiscountResponse)
async def predict_discount(request: PredictDiscountRequest) -> PredictDiscountResponse:
    """Predict discount percentage for a product.
    
    This endpoint uses a LightGBM model trained on historical sales data
    to predict the optimal discount percentage for a product based on
    its category, price, rating, and popularity.
    """
    logger.info(
        "predict_discount_request",
        category=request.category,
        price=request.actual_price,
        rating=request.rating,
    )
    
    try:
        predictor = get_predictor()
        
        if not predictor.is_loaded:
            raise HTTPException(
                status_code=503,
                detail="Model not loaded. Please train the model first using POST /predict/train",
            )
        
        # Make prediction
        result = predictor.predict_from_features(
            category=request.category,
            actual_price=request.actual_price,
            rating=request.rating,
            rating_count=request.rating_count,
        )
        
        # Get feature importance for explanation
        importance = predictor.get_feature_importance()
        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        
        explanation = FeatureExplanation(
            top_features=[f[0] for f in sorted_features[:3]],
            importance_scores={f[0]: round(f[1], 4) for f in sorted_features},
        )
        
        return PredictDiscountResponse(
            predicted_discount=result["predicted_discount"],
            confidence=result["confidence"],
            explanation=explanation,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("prediction_error", error=str(e))
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.get("/status", response_model=ModelStatusResponse)
async def get_model_status() -> ModelStatusResponse:
    """Get the status of the ML model."""
    predictor = get_predictor()
    
    return ModelStatusResponse(
        loaded=predictor.is_loaded,
        model_path=str(predictor.model_path),
        metrics=predictor.metrics if predictor.is_loaded else {},
        feature_importance=predictor.get_feature_importance() if predictor.is_loaded else {},
    )


@router.post("/train", response_model=TrainModelResponse)
async def train_model(request: TrainModelRequest) -> TrainModelResponse:
    """Train or retrain the discount prediction model.
    
    This endpoint triggers model training using either the real dataset
    or generated sample data for testing purposes.
    """
    from src.ml.trainer import train_discount_model
    
    logger.info(
        "train_model_request",
        use_sample_data=request.use_sample_data,
        test_size=request.test_size,
    )
    
    try:
        predictor, results = train_discount_model(
            use_sample_data=request.use_sample_data,
            test_size=request.test_size,
        )
        
        return TrainModelResponse(
            status="success",
            metrics=results["metrics"],
            feature_importance=results["feature_importance"],
            model_path=results["model_path"],
            training_samples=results["training_samples"],
            test_samples=results["test_samples"],
        )
        
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail=str(e),
        )
    except Exception as e:
        logger.error("training_error", error=str(e))
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


@router.post("/explain", response_model=ExplainResponse)
async def explain_prediction(request: ExplainRequest) -> ExplainResponse:
    """Get SHAP explanation for a discount prediction.
    
    This endpoint provides interpretable explanations for why the model
    predicted a certain discount percentage using SHAP values.
    """
    from src.ml.explainer import explain_prediction as get_explanation
    
    logger.info("explain_request", category=request.category, price=request.actual_price)
    
    try:
        predictor = get_predictor()
        
        if not predictor.is_loaded:
            raise HTTPException(
                status_code=503,
                detail="Model not loaded. Please train the model first.",
            )
        
        explanation = get_explanation(
            predictor=predictor,
            category=request.category,
            actual_price=request.actual_price,
            rating=request.rating,
            rating_count=request.rating_count,
        )
        
        return ExplainResponse(**explanation)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("explain_error", error=str(e))
        raise HTTPException(status_code=500, detail=f"Explanation failed: {str(e)}")
