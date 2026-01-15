"""Data loading utilities for Amazon Sales Dataset."""

import re
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import structlog

from src.config import settings

logger = structlog.get_logger(__name__)


def clean_price(price_str: str) -> Optional[float]:
    """Clean price string and convert to float.
    
    Args:
        price_str: Price string (e.g., "₹1,299" or "$99.99")
        
    Returns:
        Cleaned float value or None if parsing fails.
    """
    if pd.isna(price_str):
        return None
    
    # Remove currency symbols and commas
    cleaned = re.sub(r"[₹$,]", "", str(price_str))
    
    try:
        return float(cleaned)
    except (ValueError, TypeError):
        return None


def clean_rating(rating_str: str) -> Optional[float]:
    """Clean rating string and convert to float.
    
    Args:
        rating_str: Rating string (e.g., "4.5 out of 5 stars" or "4.5")
        
    Returns:
        Cleaned float value or None if parsing fails.
    """
    if pd.isna(rating_str):
        return None
    
    # Extract numeric rating
    match = re.search(r"(\d+\.?\d*)", str(rating_str))
    if match:
        try:
            return float(match.group(1))
        except (ValueError, TypeError):
            return None
    return None


def clean_rating_count(count_str: str) -> Optional[int]:
    """Clean rating count string and convert to integer.
    
    Args:
        count_str: Rating count string (e.g., "1,234 ratings" or "1,234")
        
    Returns:
        Cleaned integer value or None if parsing fails.
    """
    if pd.isna(count_str):
        return None
    
    # Remove commas and extract numbers
    cleaned = re.sub(r"[,\s]", "", str(count_str))
    match = re.search(r"(\d+)", cleaned)
    
    if match:
        try:
            return int(match.group(1))
        except (ValueError, TypeError):
            return None
    return None


def clean_discount(discount_str: str) -> Optional[float]:
    """Clean discount string and convert to float.
    
    Args:
        discount_str: Discount string (e.g., "50%" or "50")
        
    Returns:
        Cleaned float value (percentage) or None if parsing fails.
    """
    if pd.isna(discount_str):
        return None
    
    # Extract numeric discount
    match = re.search(r"(\d+\.?\d*)", str(discount_str))
    if match:
        try:
            return float(match.group(1))
        except (ValueError, TypeError):
            return None
    return None


def parse_hierarchical_category(category_str: str) -> Tuple[str, str, int]:
    """Parse hierarchical category string.
    
    Args:
        category_str: Category string (e.g., "Computers&Accessories|Cables|USBCables")
        
    Returns:
        Tuple of (main_category, sub_category, category_depth)
    """
    if pd.isna(category_str) or not category_str:
        return "Unknown", "Unknown", 0
    
    # Split by pipe delimiter
    parts = str(category_str).split("|")
    
    main_category = parts[0].strip() if parts else "Unknown"
    sub_category = parts[1].strip() if len(parts) > 1 else main_category
    category_depth = len(parts)
    
    return main_category, sub_category, category_depth


def get_text_length(text: str) -> int:
    """Get length of text, handling NaN values.
    
    Args:
        text: Text string.
        
    Returns:
        Length of text or 0 if NaN.
    """
    if pd.isna(text):
        return 0
    return len(str(text))


def count_reviews(review_content: str) -> int:
    """Count number of reviews in concatenated review content.
    
    Args:
        review_content: Comma-separated review content.
        
    Returns:
        Number of distinct reviews.
    """
    if pd.isna(review_content) or not review_content:
        return 0
    
    # Reviews are typically separated by commas in the dataset
    reviews = str(review_content).split(",")
    return len([r for r in reviews if r.strip()])


def load_amazon_dataset(filepath: Optional[Path] = None) -> pd.DataFrame:
    """Load and clean the Amazon Sales Dataset.
    
    Args:
        filepath: Path to CSV file. Defaults to settings.data_path.
        
    Returns:
        Cleaned DataFrame with standardized columns and engineered features.
        
    Raises:
        FileNotFoundError: If the dataset file doesn't exist.
    """
    filepath = filepath or settings.data_path
    
    if not filepath.exists():
        raise FileNotFoundError(
            f"Dataset not found at {filepath}. "
            "Please place the Amazon Sales Dataset CSV at this location."
        )
    
    logger.info("loading_dataset", filepath=str(filepath))
    
    # Load the CSV
    df = pd.read_csv(filepath)
    
    logger.info("dataset_loaded", rows=len(df), columns=list(df.columns))
    
    # Standardize column names (lowercase, underscores)
    df.columns = df.columns.str.lower().str.replace(" ", "_").str.replace("-", "_")
    
    # Expected columns mapping (handle variations in column names)
    column_mapping = {
        "product_id": ["product_id", "productid", "id"],
        "product_name": ["product_name", "productname", "name", "title"],
        "category": ["category", "main_category", "category_name"],
        "actual_price": ["actual_price", "actualprice", "original_price", "price"],
        "discounted_price": ["discounted_price", "discountedprice", "sale_price"],
        "discount_percentage": ["discount_percentage", "discountpercentage", "discount"],
        "rating": ["rating", "ratings", "avg_rating", "average_rating"],
        "rating_count": ["rating_count", "ratingcount", "num_ratings"],
        "about_product": ["about_product", "description", "product_description"],
        "user_id": ["user_id", "userid"],
        "user_name": ["user_name", "username"],
        "review_id": ["review_id", "reviewid"],
        "review_title": ["review_title", "reviewtitle"],
        "review_content": ["review_content", "reviews", "review_text"],
        "img_link": ["img_link", "image_link", "image_url"],
        "product_link": ["product_link", "product_url", "url"],
    }
    
    # Find and rename columns
    renamed_columns = {}
    for standard_name, alternatives in column_mapping.items():
        for alt in alternatives:
            if alt in df.columns:
                renamed_columns[alt] = standard_name
                break
    
    df = df.rename(columns=renamed_columns)
    
    # Clean numeric columns
    if "actual_price" in df.columns:
        df["actual_price"] = df["actual_price"].apply(clean_price)
    
    if "discounted_price" in df.columns:
        df["discounted_price"] = df["discounted_price"].apply(clean_price)
    
    if "discount_percentage" in df.columns:
        df["discount_percentage"] = df["discount_percentage"].apply(clean_discount)
    
    if "rating" in df.columns:
        df["rating"] = df["rating"].apply(clean_rating)
    
    if "rating_count" in df.columns:
        df["rating_count"] = df["rating_count"].apply(clean_rating_count)
    
    # Calculate discount if not present but prices are available
    if "discount_percentage" not in df.columns and all(
        col in df.columns for col in ["actual_price", "discounted_price"]
    ):
        df["discount_percentage"] = (
            (df["actual_price"] - df["discounted_price"]) / df["actual_price"] * 100
        ).round(2)
    
    # === ENHANCED FEATURE ENGINEERING ===
    
    # Parse hierarchical categories
    if "category" in df.columns:
        category_parsed = df["category"].apply(parse_hierarchical_category)
        df["main_category"] = category_parsed.apply(lambda x: x[0])
        df["sub_category"] = category_parsed.apply(lambda x: x[1])
        df["category_depth"] = category_parsed.apply(lambda x: x[2])
    else:
        df["main_category"] = "Unknown"
        df["sub_category"] = "Unknown"
        df["category_depth"] = 0
    
    # Text length features
    if "product_name" in df.columns:
        df["name_length"] = df["product_name"].apply(get_text_length)
    else:
        df["name_length"] = 0
    
    if "about_product" in df.columns:
        df["description_length"] = df["about_product"].apply(get_text_length)
    else:
        df["description_length"] = 0
    
    # Review count from review content
    if "review_content" in df.columns:
        df["review_text_count"] = df["review_content"].apply(count_reviews)
    else:
        df["review_text_count"] = 0
    
    # Price tier categorization
    if "actual_price" in df.columns:
        df["price_tier"] = pd.cut(
            df["actual_price"].fillna(0),
            bins=[0, 500, 2000, 10000, float("inf")],
            labels=["budget", "mid", "premium", "luxury"],
        ).astype(str)
        df["price_tier"] = df["price_tier"].replace("nan", "unknown")
    else:
        df["price_tier"] = "unknown"
    
    # Log-transformed rating count (handles skewness)
    if "rating_count" in df.columns:
        df["rating_count_log"] = np.log1p(df["rating_count"].fillna(0))
    else:
        df["rating_count_log"] = 0
    
    # Drop rows with missing critical values
    critical_columns = ["actual_price", "discount_percentage"]
    available_critical = [col for col in critical_columns if col in df.columns]
    
    if available_critical:
        initial_rows = len(df)
        df = df.dropna(subset=available_critical)
        dropped = initial_rows - len(df)
        if dropped > 0:
            logger.info("dropped_missing_values", dropped_rows=dropped)
    
    logger.info(
        "dataset_cleaned",
        final_rows=len(df),
        features=list(df.columns),
        main_categories=df["main_category"].nunique() if "main_category" in df.columns else 0,
    )
    
    return df


def get_dataset_summary(df: pd.DataFrame) -> dict:
    """Get summary statistics for the dataset.
    
    Args:
        df: DataFrame to summarize.
        
    Returns:
        Dictionary with summary statistics.
    """
    summary = {
        "total_products": len(df),
        "total_features": len(df.columns),
        "features": list(df.columns),
    }
    
    # Categorical summaries
    if "main_category" in df.columns:
        summary["main_categories"] = df["main_category"].nunique()
        summary["top_categories"] = df["main_category"].value_counts().head(10).to_dict()
    
    if "sub_category" in df.columns:
        summary["sub_categories"] = df["sub_category"].nunique()
    
    # Numerical summaries
    numerical_cols = ["actual_price", "discounted_price", "discount_percentage", 
                      "rating", "rating_count"]
    
    for col in numerical_cols:
        if col in df.columns:
            summary[f"{col}_stats"] = {
                "min": float(df[col].min()) if pd.notna(df[col].min()) else None,
                "max": float(df[col].max()) if pd.notna(df[col].max()) else None,
                "mean": float(df[col].mean()) if pd.notna(df[col].mean()) else None,
                "median": float(df[col].median()) if pd.notna(df[col].median()) else None,
                "std": float(df[col].std()) if pd.notna(df[col].std()) else None,
            }
    
    # Missing values
    summary["missing_values"] = df.isnull().sum().to_dict()
    
    return summary


def get_sample_data() -> pd.DataFrame:
    """Generate sample data for testing when no dataset is available.
    
    Returns:
        Sample DataFrame with realistic e-commerce data.
    """
    np.random.seed(42)
    n_samples = 100
    
    main_categories = ["Electronics", "Clothing", "Home&Kitchen", "Books", "Sports"]
    sub_categories = ["Accessories", "Gadgets", "Apparel", "Furniture", "Equipment"]
    price_tiers = ["budget", "mid", "premium", "luxury"]
    
    data = {
        "product_id": [f"PROD{i:04d}" for i in range(n_samples)],
        "product_name": [f"Product {i} - High Quality Item" for i in range(n_samples)],
        "main_category": np.random.choice(main_categories, n_samples),
        "sub_category": np.random.choice(sub_categories, n_samples),
        "category_depth": np.random.randint(2, 6, n_samples),
        "actual_price": np.random.uniform(100, 5000, n_samples).round(2),
        "rating": np.random.uniform(2.5, 5.0, n_samples).round(1),
        "rating_count": np.random.randint(10, 50000, n_samples),
        "about_product": [f"Description for product {i}. " * 5 for i in range(n_samples)],
        "review_content": [f"Great product, highly recommend. " * 3 for i in range(n_samples)],
        "name_length": np.random.randint(20, 100, n_samples),
        "description_length": np.random.randint(100, 1000, n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Add price tier
    df["price_tier"] = pd.cut(
        df["actual_price"],
        bins=[0, 500, 2000, 10000, float("inf")],
        labels=price_tiers,
    ).astype(str)
    
    # Log-transformed rating count
    df["rating_count_log"] = np.log1p(df["rating_count"])
    
    # Generate realistic discounts based on features
    base_discount = 10 + np.random.uniform(-5, 5, n_samples)
    
    # Higher price = higher discount potential
    price_factor = (df["actual_price"] / 1000) * 5
    
    # Lower rating = higher discount
    rating_factor = (5 - df["rating"]) * 3
    
    # Category factor
    category_discount = {
        "Electronics": 5,
        "Clothing": 10,
        "Home&Kitchen": 7,
        "Books": 3,
        "Sports": 8,
    }
    cat_factor = df["main_category"].map(category_discount).fillna(5)
    
    df["discount_percentage"] = np.clip(
        base_discount + price_factor + rating_factor + cat_factor + np.random.normal(0, 5, n_samples),
        0,
        70,
    ).round(2)
    
    df["discounted_price"] = (
        df["actual_price"] * (1 - df["discount_percentage"] / 100)
    ).round(2)
    
    # Add review text count
    df["review_text_count"] = np.random.randint(1, 10, n_samples)
    
    return df
