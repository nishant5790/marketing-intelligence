"""Exploratory Data Analysis (EDA) module for marketing data."""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import structlog

from src.config import settings
from src.data.loader import load_amazon_dataset, get_sample_data

logger = structlog.get_logger(__name__)


class DatasetAnalyzer:
    """Comprehensive analyzer for the Amazon dataset."""

    def __init__(self, df: Optional[pd.DataFrame] = None):
        """Initialize the analyzer.
        
        Args:
            df: DataFrame to analyze. Loads dataset if not provided.
        """
        self._df = df
        self._summary_cache: Optional[Dict] = None

    @property
    def df(self) -> pd.DataFrame:
        """Get or load the dataset."""
        if self._df is None:
            try:
                self._df = load_amazon_dataset()
            except FileNotFoundError:
                logger.warning("dataset_not_found_using_sample")
                self._df = get_sample_data()
        return self._df

    def get_summary(self, force_refresh: bool = False) -> Dict[str, Any]:
        """Get comprehensive dataset summary.
        
        Args:
            force_refresh: Force recalculation of summary.
            
        Returns:
            Dictionary with dataset summary.
        """
        if self._summary_cache is not None and not force_refresh:
            return self._summary_cache

        df = self.df
        
        summary = {
            "dataset_info": {
                "total_products": len(df),
                "total_features": len(df.columns),
                "features": list(df.columns),
                "memory_usage_mb": round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2),
            },
            "categories": self._analyze_categories(),
            "price_analysis": self._analyze_prices(),
            "rating_analysis": self._analyze_ratings(),
            "discount_analysis": self._analyze_discounts(),
            "text_analysis": self._analyze_text_features(),
            "missing_values": self._analyze_missing_values(),
            "correlations": self._get_correlations(),
        }
        
        self._summary_cache = summary
        logger.info("dataset_summary_generated")
        
        return summary

    def _analyze_categories(self) -> Dict[str, Any]:
        """Analyze category distribution."""
        df = self.df
        result = {}
        
        if "main_category" in df.columns:
            main_cats = df["main_category"].value_counts()
            result["main_categories"] = {
                "unique_count": len(main_cats),
                "distribution": main_cats.head(15).to_dict(),
            }
        
        if "sub_category" in df.columns:
            sub_cats = df["sub_category"].value_counts()
            result["sub_categories"] = {
                "unique_count": len(sub_cats),
                "distribution": sub_cats.head(15).to_dict(),
            }
        
        if "price_tier" in df.columns:
            result["price_tiers"] = df["price_tier"].value_counts().to_dict()
        
        if "category_depth" in df.columns:
            result["category_depth"] = {
                "min": int(df["category_depth"].min()),
                "max": int(df["category_depth"].max()),
                "mean": round(float(df["category_depth"].mean()), 2),
                "distribution": df["category_depth"].value_counts().sort_index().to_dict(),
            }
        
        return result

    def _analyze_prices(self) -> Dict[str, Any]:
        """Analyze price distribution."""
        df = self.df
        result = {}
        
        for col in ["actual_price", "discounted_price"]:
            if col in df.columns:
                valid_data = df[col].dropna()
                if len(valid_data) > 0:
                    result[col] = {
                        "min": round(float(valid_data.min()), 2),
                        "max": round(float(valid_data.max()), 2),
                        "mean": round(float(valid_data.mean()), 2),
                        "median": round(float(valid_data.median()), 2),
                        "std": round(float(valid_data.std()), 2),
                        "percentiles": {
                            "25%": round(float(valid_data.quantile(0.25)), 2),
                            "50%": round(float(valid_data.quantile(0.50)), 2),
                            "75%": round(float(valid_data.quantile(0.75)), 2),
                            "90%": round(float(valid_data.quantile(0.90)), 2),
                            "99%": round(float(valid_data.quantile(0.99)), 2),
                        },
                    }
        
        # Price distribution bins
        if "actual_price" in df.columns:
            bins = [0, 500, 1000, 2000, 5000, 10000, 50000, float("inf")]
            labels = ["0-500", "500-1K", "1K-2K", "2K-5K", "5K-10K", "10K-50K", "50K+"]
            price_bins = pd.cut(df["actual_price"].fillna(0), bins=bins, labels=labels)
            result["price_distribution"] = price_bins.value_counts().sort_index().to_dict()
        
        return result

    def _analyze_ratings(self) -> Dict[str, Any]:
        """Analyze rating distribution."""
        df = self.df
        result = {}
        
        if "rating" in df.columns:
            valid_ratings = df["rating"].dropna()
            if len(valid_ratings) > 0:
                result["rating_stats"] = {
                    "min": round(float(valid_ratings.min()), 2),
                    "max": round(float(valid_ratings.max()), 2),
                    "mean": round(float(valid_ratings.mean()), 2),
                    "median": round(float(valid_ratings.median()), 2),
                    "std": round(float(valid_ratings.std()), 2),
                }
                
                # Rating distribution
                bins = [0, 1, 2, 3, 4, 4.5, 5.01]
                labels = ["0-1", "1-2", "2-3", "3-4", "4-4.5", "4.5-5"]
                rating_bins = pd.cut(valid_ratings, bins=bins, labels=labels)
                result["rating_distribution"] = rating_bins.value_counts().sort_index().to_dict()
        
        if "rating_count" in df.columns:
            valid_counts = df["rating_count"].dropna()
            if len(valid_counts) > 0:
                result["rating_count_stats"] = {
                    "min": int(valid_counts.min()),
                    "max": int(valid_counts.max()),
                    "mean": round(float(valid_counts.mean()), 2),
                    "median": round(float(valid_counts.median()), 2),
                    "total_reviews": int(valid_counts.sum()),
                }
        
        return result

    def _analyze_discounts(self) -> Dict[str, Any]:
        """Analyze discount distribution."""
        df = self.df
        result = {}
        
        if "discount_percentage" in df.columns:
            valid_discounts = df["discount_percentage"].dropna()
            if len(valid_discounts) > 0:
                result["discount_stats"] = {
                    "min": round(float(valid_discounts.min()), 2),
                    "max": round(float(valid_discounts.max()), 2),
                    "mean": round(float(valid_discounts.mean()), 2),
                    "median": round(float(valid_discounts.median()), 2),
                    "std": round(float(valid_discounts.std()), 2),
                }
                
                # Discount distribution
                bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 101]
                labels = ["0-10%", "10-20%", "20-30%", "30-40%", "40-50%", 
                         "50-60%", "60-70%", "70-80%", "80-90%", "90-100%"]
                discount_bins = pd.cut(valid_discounts, bins=bins, labels=labels)
                result["discount_distribution"] = discount_bins.value_counts().sort_index().to_dict()
                
                # Discount by category
                if "main_category" in df.columns:
                    cat_discounts = df.groupby("main_category")["discount_percentage"].agg(
                        ["mean", "median", "count"]
                    ).round(2)
                    result["discount_by_category"] = cat_discounts.to_dict("index")
        
        return result

    def _analyze_text_features(self) -> Dict[str, Any]:
        """Analyze text-based features."""
        df = self.df
        result = {}
        
        text_cols = {
            "name_length": "product_name",
            "description_length": "about_product",
        }
        
        for length_col, source_col in text_cols.items():
            if length_col in df.columns:
                valid_lengths = df[length_col].dropna()
                if len(valid_lengths) > 0:
                    result[length_col] = {
                        "min": int(valid_lengths.min()),
                        "max": int(valid_lengths.max()),
                        "mean": round(float(valid_lengths.mean()), 2),
                        "median": round(float(valid_lengths.median()), 2),
                    }
        
        # Review sentiment if available
        if "review_sentiment" in df.columns:
            valid_sentiment = df["review_sentiment"].dropna()
            if len(valid_sentiment) > 0:
                result["review_sentiment"] = {
                    "min": round(float(valid_sentiment.min()), 3),
                    "max": round(float(valid_sentiment.max()), 3),
                    "mean": round(float(valid_sentiment.mean()), 3),
                    "positive_count": int((valid_sentiment > 0).sum()),
                    "negative_count": int((valid_sentiment < 0).sum()),
                    "neutral_count": int((valid_sentiment == 0).sum()),
                }
        
        return result

    def _analyze_missing_values(self) -> Dict[str, Any]:
        """Analyze missing values in dataset."""
        df = self.df
        missing = df.isnull().sum()
        total = len(df)
        
        return {
            "counts": missing[missing > 0].to_dict(),
            "percentages": {
                col: round(count / total * 100, 2) 
                for col, count in missing[missing > 0].items()
            },
            "complete_rows": int((~df.isnull().any(axis=1)).sum()),
            "complete_rows_percentage": round((~df.isnull().any(axis=1)).sum() / total * 100, 2),
        }

    def _get_correlations(self) -> Dict[str, Any]:
        """Get correlation analysis for numerical features."""
        df = self.df
        
        numerical_cols = [
            "actual_price", "discounted_price", "discount_percentage",
            "rating", "rating_count", "category_depth", "name_length",
            "description_length", "rating_count_log"
        ]
        
        available_cols = [col for col in numerical_cols if col in df.columns]
        
        if len(available_cols) < 2:
            return {}
        
        corr_matrix = df[available_cols].corr()
        
        # Get top correlations with discount_percentage
        result = {"correlation_matrix": {}}
        
        for col in available_cols:
            result["correlation_matrix"][col] = corr_matrix[col].round(3).to_dict()
        
        # Correlations with target (discount_percentage)
        if "discount_percentage" in available_cols:
            target_corr = corr_matrix["discount_percentage"].drop("discount_percentage")
            result["correlations_with_discount"] = target_corr.sort_values(
                key=abs, ascending=False
            ).round(3).to_dict()
        
        return result

    def get_top_products(
        self,
        by: str = "rating",
        n: int = 10,
        category: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get top products by specified metric.
        
        Args:
            by: Metric to sort by (rating, discount_percentage, rating_count).
            n: Number of products to return.
            category: Filter by main category.
            
        Returns:
            List of top products.
        """
        df = self.df
        
        if by not in df.columns:
            return []
        
        filtered_df = df.copy()
        
        if category and "main_category" in df.columns:
            filtered_df = filtered_df[filtered_df["main_category"] == category]
        
        sorted_df = filtered_df.nlargest(n, by)
        
        result = []
        for _, row in sorted_df.iterrows():
            product = {
                "product_id": str(row.get("product_id", "")),
                "product_name": str(row.get("product_name", ""))[:100],
                "main_category": str(row.get("main_category", "")),
            }
            
            for field in [by, "actual_price", "discount_percentage", "rating", "rating_count"]:
                if field in row and pd.notna(row[field]):
                    product[field] = round(float(row[field]), 2) if isinstance(row[field], float) else row[field]
            
            result.append(product)
        
        return result

    def get_category_summary(self, category: str) -> Dict[str, Any]:
        """Get summary for a specific category.
        
        Args:
            category: Main category name.
            
        Returns:
            Category summary statistics.
        """
        df = self.df
        
        if "main_category" not in df.columns:
            return {"error": "Category column not found"}
        
        cat_df = df[df["main_category"] == category]
        
        if len(cat_df) == 0:
            return {"error": f"Category '{category}' not found"}
        
        return {
            "category": category,
            "product_count": len(cat_df),
            "avg_price": round(float(cat_df["actual_price"].mean()), 2) if "actual_price" in cat_df else None,
            "avg_discount": round(float(cat_df["discount_percentage"].mean()), 2) if "discount_percentage" in cat_df else None,
            "avg_rating": round(float(cat_df["rating"].mean()), 2) if "rating" in cat_df else None,
            "total_reviews": int(cat_df["rating_count"].sum()) if "rating_count" in cat_df else None,
            "sub_categories": cat_df["sub_category"].nunique() if "sub_category" in cat_df else None,
            "price_range": {
                "min": round(float(cat_df["actual_price"].min()), 2),
                "max": round(float(cat_df["actual_price"].max()), 2),
            } if "actual_price" in cat_df else None,
        }


# Singleton instance
_analyzer_instance: Optional[DatasetAnalyzer] = None


def get_analyzer() -> DatasetAnalyzer:
    """Get or create the singleton analyzer instance.
    
    Returns:
        DatasetAnalyzer instance.
    """
    global _analyzer_instance
    
    if _analyzer_instance is None:
        _analyzer_instance = DatasetAnalyzer()
    
    return _analyzer_instance


def analyze_dataset(df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """Convenience function to get dataset analysis.
    
    Args:
        df: Optional DataFrame to analyze.
        
    Returns:
        Dataset summary dictionary.
    """
    if df is not None:
        analyzer = DatasetAnalyzer(df)
    else:
        analyzer = get_analyzer()
    
    return analyzer.get_summary()
