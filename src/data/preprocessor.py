"""Data preprocessing and feature engineering for ML models."""

from typing import Optional, Tuple

import numpy as np
import pandas as pd
import structlog
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

logger = structlog.get_logger(__name__)


def get_sentiment_score(text: str) -> float:
    """Calculate sentiment polarity score for text.
    
    Args:
        text: Text to analyze.
        
    Returns:
        Sentiment polarity score (-1 to 1).
    """
    if pd.isna(text) or not text:
        return 0.0
    
    try:
        from textblob import TextBlob
        # Truncate very long text to avoid performance issues
        text_sample = str(text)[:2000]
        blob = TextBlob(text_sample)
        return blob.sentiment.polarity
    except Exception:
        return 0.0


def get_subjectivity_score(text: str) -> float:
    """Calculate subjectivity score for text.
    
    Args:
        text: Text to analyze.
        
    Returns:
        Subjectivity score (0 to 1).
    """
    if pd.isna(text) or not text:
        return 0.5
    
    try:
        from textblob import TextBlob
        text_sample = str(text)[:2000]
        blob = TextBlob(text_sample)
        return blob.sentiment.subjectivity
    except Exception:
        return 0.5


class FeaturePreprocessor:
    """Preprocessor for discount prediction features with advanced engineering."""

    def __init__(self):
        """Initialize the preprocessor."""
        self.label_encoders: dict[str, LabelEncoder] = {}
        self.scaler: Optional[StandardScaler] = None
        self.feature_columns: list[str] = []
        
        # Enhanced feature definitions
        self.categorical_columns: list[str] = [
            "main_category", 
            "sub_category", 
            "price_tier"
        ]
        self.numerical_columns: list[str] = [
            "actual_price", 
            "rating", 
            "rating_count",
            "category_depth",
            "name_length",
            "description_length",
            "rating_count_log",
            "review_sentiment",
        ]
        self._fitted = False

    def fit(self, df: pd.DataFrame) -> "FeaturePreprocessor":
        """Fit the preprocessor on training data.
        
        Args:
            df: Training DataFrame with features.
            
        Returns:
            Self for method chaining.
        """
        logger.info("fitting_preprocessor", rows=len(df))
        
        # Add sentiment features if review content exists
        df = self._add_sentiment_features(df)
        
        # Fit label encoders for categorical columns
        for col in self.categorical_columns:
            if col in df.columns:
                self.label_encoders[col] = LabelEncoder()
                # Handle missing values by filling with 'Unknown'
                values = df[col].fillna("Unknown").astype(str)
                self.label_encoders[col].fit(values)
        
        # Fit scaler for numerical columns
        numerical_data = self._prepare_numerical(df)
        if numerical_data is not None and len(numerical_data.columns) > 0:
            self.scaler = StandardScaler()
            self.scaler.fit(numerical_data)
        
        # Store feature columns
        self.feature_columns = list(self.label_encoders.keys()) + list(
            numerical_data.columns if numerical_data is not None else []
        )
        
        self._fitted = True
        logger.info("preprocessor_fitted", features=self.feature_columns)
        
        return self

    def _add_sentiment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add sentiment features from review content.
        
        Args:
            df: DataFrame with potential review content.
            
        Returns:
            DataFrame with sentiment features added.
        """
        df = df.copy()
        
        # Add sentiment from review content
        if "review_content" in df.columns and "review_sentiment" not in df.columns:
            logger.info("calculating_sentiment_scores", rows=len(df))
            df["review_sentiment"] = df["review_content"].apply(get_sentiment_score)
        elif "review_sentiment" not in df.columns:
            df["review_sentiment"] = 0.0
        
        return df

    def _prepare_numerical(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare numerical features.
        
        Args:
            df: Input DataFrame.
            
        Returns:
            DataFrame with numerical features.
        """
        available_cols = [col for col in self.numerical_columns if col in df.columns]
        
        if not available_cols:
            return pd.DataFrame()
        
        numerical_df = df[available_cols].copy()
        
        # Fill missing values with median
        for col in available_cols:
            if numerical_df[col].dtype in ['float64', 'int64', 'float32', 'int32']:
                median_val = numerical_df[col].median()
                if pd.isna(median_val):
                    median_val = 0
                numerical_df[col] = numerical_df[col].fillna(median_val)
            else:
                numerical_df[col] = pd.to_numeric(numerical_df[col], errors='coerce').fillna(0)
        
        return numerical_df

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform features for model input.
        
        Args:
            df: Input DataFrame with features.
            
        Returns:
            Numpy array of transformed features.
            
        Raises:
            ValueError: If preprocessor hasn't been fitted.
        """
        if not self._fitted:
            raise ValueError("Preprocessor must be fitted before transform")
        
        # Add sentiment features
        df = self._add_sentiment_features(df)
        
        features = []
        
        # Transform categorical columns
        for col, encoder in self.label_encoders.items():
            if col in df.columns:
                values = df[col].fillna("Unknown").astype(str)
                # Handle unseen categories
                transformed = []
                for val in values:
                    if val in encoder.classes_:
                        transformed.append(encoder.transform([val])[0])
                    else:
                        # Map unknown to most frequent class (index 0)
                        transformed.append(0)
                features.append(np.array(transformed).reshape(-1, 1))
            else:
                # Column not present, use zeros
                features.append(np.zeros((len(df), 1)))
        
        # Transform numerical columns
        numerical_data = self._prepare_numerical(df)
        if self.scaler is not None and len(numerical_data.columns) > 0:
            scaled = self.scaler.transform(numerical_data)
            features.append(scaled)
        
        if not features:
            raise ValueError("No features to transform")
        
        return np.hstack(features)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """Fit and transform in one step.
        
        Args:
            df: Input DataFrame.
            
        Returns:
            Transformed feature array.
        """
        self.fit(df)
        return self.transform(df)

    def get_feature_names(self) -> list[str]:
        """Get names of all features.
        
        Returns:
            List of feature names.
        """
        names = list(self.label_encoders.keys())
        available_numerical = [
            col for col in self.numerical_columns 
            if col in self.feature_columns or col in self.numerical_columns
        ]
        names.extend(available_numerical)
        return names
    
    def get_feature_info(self) -> dict:
        """Get detailed information about features.
        
        Returns:
            Dictionary with feature information.
        """
        return {
            "categorical_features": list(self.label_encoders.keys()),
            "numerical_features": [
                col for col in self.numerical_columns 
                if self.scaler is not None
            ],
            "total_features": len(self.feature_columns),
            "encoder_classes": {
                col: list(enc.classes_) 
                for col, enc in self.label_encoders.items()
            },
        }


def prepare_training_data(
    df: pd.DataFrame,
    target_column: str = "discount_percentage",
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, FeaturePreprocessor]:
    """Prepare data for model training.
    
    Args:
        df: Input DataFrame with features and target.
        target_column: Name of the target column.
        test_size: Fraction of data to use for testing.
        random_state: Random seed for reproducibility.
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test, preprocessor)
    """
    logger.info(
        "preparing_training_data",
        total_rows=len(df),
        target=target_column,
        test_size=test_size,
    )
    
    # Validate target column exists
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame")
    
    # Remove rows with missing target
    df_clean = df.dropna(subset=[target_column]).copy()
    
    # Extract target
    y = df_clean[target_column].values
    
    # Split data
    train_idx, test_idx = train_test_split(
        np.arange(len(df_clean)),
        test_size=test_size,
        random_state=random_state,
    )
    
    df_train = df_clean.iloc[train_idx]
    df_test = df_clean.iloc[test_idx]
    
    # Fit preprocessor on training data only
    preprocessor = FeaturePreprocessor()
    X_train = preprocessor.fit_transform(df_train)
    X_test = preprocessor.transform(df_test)
    
    y_train = y[train_idx]
    y_test = y[test_idx]
    
    logger.info(
        "training_data_prepared",
        train_samples=len(y_train),
        test_samples=len(y_test),
        n_features=X_train.shape[1],
        feature_names=preprocessor.get_feature_names(),
    )
    
    return X_train, X_test, y_train, y_test, preprocessor


def prepare_rag_documents(df: pd.DataFrame) -> list[dict]:
    """Prepare product data for RAG indexing.
    
    Args:
        df: DataFrame with product information.
        
    Returns:
        List of document dictionaries for indexing.
    """
    documents = []
    
    for idx, row in df.iterrows():
        # Build document text from available columns
        text_parts = []
        
        if "product_name" in row and pd.notna(row.get("product_name")):
            text_parts.append(f"Product: {row['product_name']}")
        
        if "main_category" in row and pd.notna(row.get("main_category")):
            text_parts.append(f"Category: {row['main_category']}")
        
        if "sub_category" in row and pd.notna(row.get("sub_category")):
            text_parts.append(f"Sub-category: {row['sub_category']}")
        
        if "about_product" in row and pd.notna(row.get("about_product")):
            # Truncate long descriptions
            desc = str(row["about_product"])[:800]
            text_parts.append(f"Description: {desc}")
        
        if "review_content" in row and pd.notna(row.get("review_content")):
            # Truncate long reviews
            review = str(row["review_content"])[:500]
            text_parts.append(f"Reviews: {review}")
        
        if "review_title" in row and pd.notna(row.get("review_title")):
            text_parts.append(f"Review Highlights: {row['review_title']}")
        
        document_text = "\n".join(text_parts)
        
        # Build metadata
        metadata = {
            "product_id": str(row.get("product_id", idx)),
            "product_name": str(row.get("product_name", "Unknown")),
        }
        
        # Add numeric metadata if available
        numeric_fields = [
            ("actual_price", float),
            ("discounted_price", float),
            ("discount_percentage", float),
            ("rating", float),
            ("rating_count", int),
        ]
        
        for field, dtype in numeric_fields:
            if field in row and pd.notna(row.get(field)):
                try:
                    metadata[field] = dtype(row[field])
                except (ValueError, TypeError):
                    pass
        
        # Add categorical metadata
        for field in ["main_category", "sub_category", "price_tier"]:
            if field in row and pd.notna(row.get(field)):
                metadata[field] = str(row[field])
        
        documents.append({
            "id": str(row.get("product_id", idx)),
            "text": document_text,
            "metadata": metadata,
        })
    
    logger.info("rag_documents_prepared", count=len(documents))
    
    return documents


def compute_feature_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """Compute correlation matrix for numerical features.
    
    Args:
        df: DataFrame with features.
        
    Returns:
        Correlation matrix as DataFrame.
    """
    numerical_cols = [
        "actual_price", "discounted_price", "discount_percentage",
        "rating", "rating_count", "category_depth", "name_length",
        "description_length", "rating_count_log"
    ]
    
    available_cols = [col for col in numerical_cols if col in df.columns]
    
    if not available_cols:
        return pd.DataFrame()
    
    return df[available_cols].corr()
