# Machine Learning Pipeline

This document details the ML pipeline architecture for discount prediction.

## Overview

The ML system uses LightGBM (Gradient Boosting) to predict optimal discount percentages for products based on historical sales data and product attributes.

```mermaid
graph TB
    subgraph "Training Pipeline"
        RawData[(Raw CSV Data)]
        DataLoader[Data Loader]
        Preprocessor[Feature Preprocessor]
        FeatureEng[Feature Engineering]
        Split[Train/Test Split]
        Training[Model Training]
        Evaluation[Model Evaluation]
        SaveModel[(Save Model)]
    end
    
    subgraph "Inference Pipeline"
        APIRequest[API Request]
        LoadModel[Load Model]
        Transform[Feature Transform]
        Predict[LightGBM Predict]
        PostProcess[Post-Processing]
        APIResponse[API Response]
    end
    
    subgraph "Monitoring"
        DriftDetect[Drift Detection]
        Retraining[Auto-Retraining]
    end
    
    RawData --> DataLoader --> Preprocessor --> FeatureEng --> Split
    Split --> Training --> Evaluation --> SaveModel
    
    APIRequest --> LoadModel --> Transform --> Predict --> PostProcess --> APIResponse
    
    Predict --> DriftDetect
    DriftDetect -->|Drift Detected| Retraining --> Training
```

## Feature Engineering

### Input Features

```mermaid
graph LR
    subgraph "Categorical Features"
        MC[main_category]
        SC[sub_category]
        PT[price_tier]
    end
    
    subgraph "Numerical Features"
        AP[actual_price]
        R[rating]
        RC[rating_count]
        CD[category_depth]
        NL[name_length]
        DL[description_length]
        RCL[rating_count_log]
        RS[review_sentiment]
    end
    
    subgraph "Derived Features"
        SentAnalysis[Sentiment Analysis]
        TextBlob[TextBlob]
    end
    
    MC & SC & PT --> Encoder[Label Encoder]
    AP & R & RC & CD & NL & DL & RCL & RS --> Scaler[Standard Scaler]
    
    review_content --> SentAnalysis --> TextBlob --> RS
    
    Encoder --> Features[Feature Vector]
    Scaler --> Features
```

### Feature Descriptions

| Feature | Type | Description | Transformation |
|---------|------|-------------|----------------|
| `main_category` | Categorical | Top-level product category | Label Encoding |
| `sub_category` | Categorical | Sub-category | Label Encoding |
| `price_tier` | Categorical | Price bucket (budget/mid/premium/luxury) | Label Encoding |
| `actual_price` | Numerical | Original price | Standard Scaling |
| `rating` | Numerical | Product rating (1-5) | Standard Scaling |
| `rating_count` | Numerical | Number of reviews | Standard Scaling |
| `category_depth` | Numerical | Hierarchy depth | Standard Scaling |
| `name_length` | Numerical | Product name length | Standard Scaling |
| `description_length` | Numerical | Description length | Standard Scaling |
| `rating_count_log` | Numerical | Log-transformed rating count | Standard Scaling |
| `review_sentiment` | Numerical | Sentiment score (-1 to 1) | Standard Scaling |

## Model Architecture

### LightGBM Configuration

```python
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
}
```

### Model Training Flow

```mermaid
sequenceDiagram
    participant T as Trainer
    participant L as Loader
    participant P as Preprocessor
    participant M as Model
    participant E as Evaluator
    participant S as Storage
    
    T->>L: load_amazon_dataset()
    L-->>T: DataFrame
    
    T->>P: fit(df_train)
    P-->>T: Fitted preprocessor
    
    T->>P: transform(df_train)
    P-->>T: X_train
    
    T->>M: fit(X_train, y_train)
    Note over M: Early stopping on validation
    M-->>T: Trained model
    
    T->>E: evaluate(X_test, y_test)
    E-->>T: Metrics (RMSE, MAE, R²)
    
    T->>S: save(model, preprocessor)
    S-->>T: model_path
```

## Model Evaluation

### Metrics Tracked

```mermaid
graph TD
    subgraph "Regression Metrics"
        RMSE[RMSE<br/>Root Mean Square Error]
        MAE[MAE<br/>Mean Absolute Error]
        R2[R²<br/>Coefficient of Determination]
        MAPE[MAPE<br/>Mean Absolute % Error]
    end
    
    subgraph "Model Artifacts"
        FI[Feature Importance]
        SHAP[SHAP Values]
    end
    
    Predictions --> RMSE & MAE & R2 & MAPE
    Model --> FI
    Model --> SHAP
```

### Evaluation Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **RMSE** | √(Σ(y - ŷ)²/n) | Lower is better; same units as target |
| **MAE** | Σ\|y - ŷ\|/n | Average absolute error |
| **R²** | 1 - SS_res/SS_tot | 0-1; higher means better fit |
| **MAPE** | Σ\|y - ŷ\|/y * 100 | Percentage error |

## Explainability with SHAP

### SHAP Integration

```mermaid
graph LR
    subgraph "SHAP Explainer"
        Model[LightGBM Model]
        TreeExplainer[Tree Explainer]
        ShapValues[SHAP Values]
    end
    
    subgraph "Explanation Output"
        BaseValue[Base Value<br/>Expected prediction]
        FeatureContrib[Feature Contributions]
        Summary[Text Summary]
    end
    
    Model --> TreeExplainer
    TreeExplainer --> ShapValues
    
    ShapValues --> BaseValue
    ShapValues --> FeatureContrib
    FeatureContrib --> Summary
```

### SHAP Explanation Structure

```json
{
    "base_value": 28.5,
    "prediction": 35.2,
    "shap_values": [
        {
            "feature_name": "category",
            "value": "Electronics",
            "shap_value": 4.2,
            "contribution": "positive"
        },
        {
            "feature_name": "rating_count",
            "value": 1250,
            "shap_value": 2.1,
            "contribution": "positive"
        }
    ],
    "explanation_summary": "The predicted discount is 35.2%..."
}
```

## Drift Detection

### Drift Detection Methods

```mermaid
graph TB
    subgraph "Statistical Tests"
        KS[Kolmogorov-Smirnov Test]
        PSI[Population Stability Index]
        PredDrift[Prediction Drift]
    end
    
    subgraph "Thresholds"
        KSThresh[p-value < 0.1]
        PSIThresh[PSI >= 0.2]
        MAEThresh[MAE increase > 50%]
    end
    
    subgraph "Actions"
        Alert[Alert]
        Retrain[Trigger Retraining]
    end
    
    KS --> KSThresh
    PSI --> PSIThresh
    PredDrift --> MAEThresh
    
    KSThresh -->|Significant| Alert
    PSIThresh -->|Significant| Retrain
    MAEThresh -->|Significant| Retrain
```

### PSI Interpretation

| PSI Value | Interpretation |
|-----------|----------------|
| < 0.1 | No significant change |
| 0.1 - 0.2 | Slight change, monitor |
| ≥ 0.2 | Significant change, investigate |
| ≥ 0.25 | Major shift, retrain model |

## Model Persistence

### Saved Artifacts

```mermaid
graph LR
    subgraph "Model Bundle"
        Model[LightGBM Model]
        Preprocessor[Feature Preprocessor]
        FeatureNames[Feature Names]
        Metrics[Training Metrics]
    end
    
    Bundle[discount_predictor.joblib]
    
    Model --> Bundle
    Preprocessor --> Bundle
    FeatureNames --> Bundle
    Metrics --> Bundle
```

### File Structure

```
models/
├── discount_predictor.joblib          # Current model
├── discount_predictor_20240115.joblib # Versioned backup
└── discount_predictor_20240110.joblib # Previous version
```

## Inference Pipeline

### Prediction Flow

```mermaid
sequenceDiagram
    participant API as API Endpoint
    participant P as Predictor
    participant Pre as Preprocessor
    participant M as Model
    
    API->>P: predict_from_features(category, price, rating, count)
    
    P->>P: Determine price_tier
    P->>P: Build feature DataFrame
    
    P->>Pre: transform(df)
    Pre-->>P: Feature vector X
    
    P->>M: predict(X)
    M-->>P: Raw prediction
    
    P->>P: Clip to [0, 100]
    P->>P: Calculate confidence
    
    P-->>API: {predicted_discount, confidence, features_used}
```

### Confidence Calculation

Confidence is reduced based on:
- Extreme prices (< $10 or > $100,000): -20%
- Invalid ratings (< 1 or > 5): -30%
- Low rating count (< 5): -20%

Minimum confidence: 30%

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict/discount` | POST | Make discount prediction |
| `/predict/status` | GET | Get model status |
| `/predict/train` | POST | Train/retrain model |
| `/predict/explain` | POST | Get SHAP explanation |

## Best Practices

### Model Training
1. Always validate on held-out test set
2. Use early stopping to prevent overfitting
3. Monitor feature importance for data quality
4. Version models with timestamps

### Production
1. Load model once at startup
2. Use singleton pattern for efficiency
3. Clip predictions to valid range
4. Return confidence scores

### Monitoring
1. Track prediction latency
2. Monitor prediction distribution
3. Set up drift detection alerts
4. Log all predictions for analysis

## Related Documentation

- [Data Flow](./data-flow.md) - Data processing details
- [API Documentation](../api/endpoints.md) - API reference
- [Deployment](../deployment/docker.md) - Deployment guide
