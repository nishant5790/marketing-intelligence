"""Analysis API routes for EDA and data insights."""

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
import structlog

from src.analysis.eda import get_analyzer, DatasetAnalyzer

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/analysis", tags=["Analysis"])


@router.get("/summary")
async def get_dataset_summary(force_refresh: bool = False) -> Dict[str, Any]:
    """Get comprehensive dataset summary and statistics.
    
    This endpoint provides a complete overview of the dataset including:
    - Basic statistics (total products, features)
    - Category distributions
    - Price analysis with percentiles
    - Rating distribution
    - Discount analysis
    - Text feature analysis
    - Missing value summary
    - Feature correlations
    """
    logger.info("get_dataset_summary", force_refresh=force_refresh)
    
    try:
        analyzer = get_analyzer()
        summary = analyzer.get_summary(force_refresh=force_refresh)
        return summary
    except Exception as e:
        logger.error("summary_error", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to generate summary: {str(e)}")


@router.get("/categories")
async def get_category_analysis() -> Dict[str, Any]:
    """Get detailed category breakdown and distribution."""
    logger.info("get_category_analysis")
    
    try:
        analyzer = get_analyzer()
        summary = analyzer.get_summary()
        return summary.get("categories", {})
    except Exception as e:
        logger.error("category_analysis_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/categories/{category}")
async def get_category_details(category: str) -> Dict[str, Any]:
    """Get detailed statistics for a specific category.
    
    Args:
        category: Main category name to analyze.
    """
    logger.info("get_category_details", category=category)
    
    try:
        analyzer = get_analyzer()
        details = analyzer.get_category_summary(category)
        
        if "error" in details:
            raise HTTPException(status_code=404, detail=details["error"])
        
        return details
    except HTTPException:
        raise
    except Exception as e:
        logger.error("category_details_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/prices")
async def get_price_analysis() -> Dict[str, Any]:
    """Get price distribution and statistics."""
    logger.info("get_price_analysis")
    
    try:
        analyzer = get_analyzer()
        summary = analyzer.get_summary()
        return summary.get("price_analysis", {})
    except Exception as e:
        logger.error("price_analysis_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/ratings")
async def get_rating_analysis() -> Dict[str, Any]:
    """Get rating distribution and statistics."""
    logger.info("get_rating_analysis")
    
    try:
        analyzer = get_analyzer()
        summary = analyzer.get_summary()
        return summary.get("rating_analysis", {})
    except Exception as e:
        logger.error("rating_analysis_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/discounts")
async def get_discount_analysis() -> Dict[str, Any]:
    """Get discount distribution and statistics by category."""
    logger.info("get_discount_analysis")
    
    try:
        analyzer = get_analyzer()
        summary = analyzer.get_summary()
        return summary.get("discount_analysis", {})
    except Exception as e:
        logger.error("discount_analysis_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/correlations")
async def get_correlations() -> Dict[str, Any]:
    """Get feature correlation matrix and key correlations with discount percentage."""
    logger.info("get_correlations")
    
    try:
        analyzer = get_analyzer()
        summary = analyzer.get_summary()
        return summary.get("correlations", {})
    except Exception as e:
        logger.error("correlations_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/top-products")
async def get_top_products(
    by: str = Query("rating", description="Metric to sort by: rating, discount_percentage, rating_count"),
    n: int = Query(10, ge=1, le=100, description="Number of products to return"),
    category: Optional[str] = Query(None, description="Filter by main category"),
) -> Dict[str, Any]:
    """Get top products by specified metric.
    
    Args:
        by: Metric to sort by (rating, discount_percentage, rating_count)
        n: Number of products to return (1-100)
        category: Optional main category filter
    """
    logger.info("get_top_products", by=by, n=n, category=category)
    
    valid_metrics = ["rating", "discount_percentage", "rating_count", "actual_price"]
    if by not in valid_metrics:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid metric. Must be one of: {valid_metrics}"
        )
    
    try:
        analyzer = get_analyzer()
        products = analyzer.get_top_products(by=by, n=n, category=category)
        return {
            "metric": by,
            "count": len(products),
            "category_filter": category,
            "products": products,
        }
    except Exception as e:
        logger.error("top_products_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/missing-values")
async def get_missing_values() -> Dict[str, Any]:
    """Get analysis of missing values in the dataset."""
    logger.info("get_missing_values")
    
    try:
        analyzer = get_analyzer()
        summary = analyzer.get_summary()
        return summary.get("missing_values", {})
    except Exception as e:
        logger.error("missing_values_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/text-features")
async def get_text_analysis() -> Dict[str, Any]:
    """Get analysis of text-based features including sentiment."""
    logger.info("get_text_analysis")
    
    try:
        analyzer = get_analyzer()
        summary = analyzer.get_summary()
        return summary.get("text_analysis", {})
    except Exception as e:
        logger.error("text_analysis_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/dataset-info")
async def get_dataset_info() -> Dict[str, Any]:
    """Get basic dataset information."""
    logger.info("get_dataset_info")
    
    try:
        analyzer = get_analyzer()
        summary = analyzer.get_summary()
        return summary.get("dataset_info", {})
    except Exception as e:
        logger.error("dataset_info_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))
