# Data Flow Architecture

This document describes how data flows through the Marketing Data Intelligence system.

## Overview

```mermaid
graph TB
    subgraph "Data Sources"
        CSV[(Amazon CSV<br/>Raw Data)]
        API[API Requests]
    end
    
    subgraph "Data Processing"
        Loader[Data Loader]
        Cleaner[Data Cleaner]
        FeatEng[Feature Engineering]
        Preprocessor[Preprocessor]
    end
    
    subgraph "Storage"
        Models[(Model Files)]
        Vectors[(Qdrant Vectors)]
    end
    
    subgraph "Output"
        Predictions[Predictions]
        Answers[Q&A Answers]
        Analytics[Analytics]
    end
    
    CSV --> Loader --> Cleaner --> FeatEng
    FeatEng --> Preprocessor --> Models
    FeatEng --> Vectors
    
    API --> Preprocessor
    Models --> Predictions
    Vectors --> Answers
    CSV --> Analytics
```

## Data Loading Pipeline

### CSV Processing Flow

```mermaid
graph LR
    subgraph "Input"
        CSV[CSV File]
    end
    
    subgraph "Column Mapping"
        Raw[Raw Columns]
        Std[Standardized Names]
    end
    
    subgraph "Data Cleaning"
        CleanPrice[Clean Prices]
        CleanRating[Clean Ratings]
        CleanDiscount[Clean Discounts]
    end
    
    subgraph "Feature Engineering"
        ParseCat[Parse Categories]
        CalcLen[Calculate Lengths]
        AddTier[Add Price Tier]
        LogTrans[Log Transform]
    end
    
    subgraph "Output"
        CleanDF[Cleaned DataFrame]
    end
    
    CSV --> Raw --> Std
    Std --> CleanPrice & CleanRating & CleanDiscount
    CleanPrice & CleanRating & CleanDiscount --> ParseCat
    ParseCat --> CalcLen --> AddTier --> LogTrans --> CleanDF
```

### Data Cleaning Functions

```mermaid
graph TB
    subgraph "Price Cleaning"
        PriceStr["₹1,299 or $99.99"]
        RemoveSym[Remove ₹ $ ,]
        ParseFloat[Parse to Float]
        PriceNum[1299.0 or 99.99]
    end
    
    subgraph "Rating Cleaning"
        RatingStr["4.5 out of 5 stars"]
        ExtractNum[Extract Numeric]
        RatingNum[4.5]
    end
    
    subgraph "Discount Cleaning"
        DiscountStr["50%"]
        ExtractPct[Extract Percentage]
        DiscountNum[50.0]
    end
    
    PriceStr --> RemoveSym --> ParseFloat --> PriceNum
    RatingStr --> ExtractNum --> RatingNum
    DiscountStr --> ExtractPct --> DiscountNum
```

### Column Mapping

| Original Column | Standardized Name | Description |
|-----------------|-------------------|-------------|
| `product_id`, `productid` | `product_id` | Unique identifier |
| `product_name`, `title` | `product_name` | Product name |
| `category`, `main_category` | `category` | Category hierarchy |
| `actual_price`, `price` | `actual_price` | Original price |
| `discount_percentage` | `discount_percentage` | Discount % |
| `rating`, `avg_rating` | `rating` | Average rating |
| `rating_count`, `num_ratings` | `rating_count` | Review count |

## Feature Engineering Pipeline

### Derived Features

```mermaid
graph TB
    subgraph "Input Features"
        Cat[category]
        Price[actual_price]
        Name[product_name]
        Desc[about_product]
        Review[review_content]
        RatingCount[rating_count]
    end
    
    subgraph "Derived Features"
        MainCat[main_category]
        SubCat[sub_category]
        CatDepth[category_depth]
        PriceTier[price_tier]
        NameLen[name_length]
        DescLen[description_length]
        Sentiment[review_sentiment]
        RCLog[rating_count_log]
    end
    
    Cat --> MainCat & SubCat & CatDepth
    Price --> PriceTier
    Name --> NameLen
    Desc --> DescLen
    Review --> Sentiment
    RatingCount --> RCLog
```

### Category Parsing

```mermaid
graph LR
    Input["Computers&Accessories|Cables|USBCables"]
    Split[Split by pipe]
    Parts["['Computers&Accessories', 'Cables', 'USBCables']"]
    
    MainCat[main_category = 'Computers&Accessories']
    SubCat[sub_category = 'Cables']
    Depth[category_depth = 3]
    
    Input --> Split --> Parts
    Parts --> MainCat & SubCat & Depth
```

### Price Tier Classification

```mermaid
graph TB
    Price[actual_price]
    
    Budget["Budget<br/>0 - 500"]
    Mid["Mid<br/>500 - 2,000"]
    Premium["Premium<br/>2,000 - 10,000"]
    Luxury["Luxury<br/>10,000+"]
    
    Price --> Budget & Mid & Premium & Luxury
```

## ML Feature Preprocessing

### Preprocessing Pipeline

```mermaid
graph TB
    subgraph "Raw Features"
        CatFeatures[Categorical Features]
        NumFeatures[Numerical Features]
    end
    
    subgraph "Transformations"
        LabelEnc[Label Encoding]
        StdScale[Standard Scaling]
    end
    
    subgraph "Output"
        FeatureVector[Feature Vector]
    end
    
    CatFeatures --> LabelEnc
    NumFeatures --> StdScale
    
    LabelEnc --> FeatureVector
    StdScale --> FeatureVector
```

### Categorical Encoding

```mermaid
graph LR
    subgraph "Before"
        Cat1["Electronics"]
        Cat2["Clothing"]
        Cat3["Home&Kitchen"]
    end
    
    subgraph "After"
        Enc1[0]
        Enc2[1]
        Enc3[2]
    end
    
    Cat1 --> Enc1
    Cat2 --> Enc2
    Cat3 --> Enc3
```

### Numerical Scaling

```mermaid
graph LR
    subgraph "Before"
        Price1["1500.00"]
        Price2["50.00"]
        Price3["10000.00"]
    end
    
    subgraph "After (z-score)"
        Scaled1["-0.2"]
        Scaled2["-1.5"]
        Scaled3["2.1"]
    end
    
    Price1 --> Scaled1
    Price2 --> Scaled2
    Price3 --> Scaled3
```

## RAG Document Preparation

### Document Construction

```mermaid
graph TB
    subgraph "Product Row"
        Name[product_name]
        Cat[category]
        Desc[about_product]
        Reviews[review_content]
    end
    
    subgraph "Document Text"
        TextParts["Product: ...<br/>Category: ...<br/>Description: ...<br/>Reviews: ..."]
    end
    
    subgraph "Metadata"
        Meta["product_id, price,<br/>rating, discount, etc."]
    end
    
    subgraph "Output Document"
        Doc["{id, text, metadata}"]
    end
    
    Name & Cat & Desc & Reviews --> TextParts
    Name & Cat --> Meta
    
    TextParts --> Doc
    Meta --> Doc
```

### Document Indexing Flow

```mermaid
sequenceDiagram
    participant DF as DataFrame
    participant Prep as Preprocessor
    participant Emb as Embedder
    participant Idx as Indexer
    participant Q as Qdrant
    
    DF->>Prep: prepare_rag_documents()
    Note over Prep: Build text and metadata
    Prep-->>Idx: documents[]
    
    Idx->>Emb: embed_documents(texts)
    Emb-->>Idx: embeddings[]
    
    loop Batch upsert
        Idx->>Q: upsert(points)
    end
```

## Inference Data Flow

### Prediction Request Flow

```mermaid
sequenceDiagram
    participant Client
    participant API
    participant Predictor
    participant Preprocessor
    participant Model
    
    Client->>API: POST /predict_discount
    Note over API: {category, price, rating, count}
    
    API->>Predictor: predict_from_features()
    
    Predictor->>Predictor: Build DataFrame
    Note over Predictor: Add derived features<br/>(price_tier, etc.)
    
    Predictor->>Preprocessor: transform(df)
    Note over Preprocessor: Apply saved<br/>encoders & scaler
    Preprocessor-->>Predictor: feature_vector
    
    Predictor->>Model: predict(X)
    Model-->>Predictor: raw_prediction
    
    Predictor->>Predictor: Clip to [0, 100]
    Predictor->>Predictor: Calculate confidence
    
    Predictor-->>API: {discount, confidence}
    API-->>Client: JSON Response
```

### Q&A Request Flow

```mermaid
sequenceDiagram
    participant Client
    participant API
    participant Retriever
    participant Embedder
    participant Qdrant
    participant Gemini
    
    Client->>API: POST /answer_question
    Note over API: {question, filters}
    
    API->>Retriever: search(query)
    Retriever->>Embedder: embed_query(question)
    Embedder-->>Retriever: query_vector
    
    Retriever->>Qdrant: search(vector, filters)
    Qdrant-->>Retriever: documents[]
    
    Retriever-->>API: search_results
    
    API->>API: Build context string
    API->>API: Construct prompt
    
    API->>Gemini: generate(prompt)
    Gemini-->>API: answer
    
    API->>API: Extract sources
    API-->>Client: {answer, sources, grounded}
```

## Analytics Data Flow

### EDA Pipeline

```mermaid
graph TB
    subgraph "Input"
        Data[(Dataset)]
    end
    
    subgraph "Analysis"
        CatAnalysis[Category Analysis]
        PriceAnalysis[Price Analysis]
        RatingAnalysis[Rating Analysis]
        DiscountAnalysis[Discount Analysis]
        TextAnalysis[Text Analysis]
        CorrAnalysis[Correlation Analysis]
    end
    
    subgraph "Output"
        Summary[Summary Dict]
    end
    
    Data --> CatAnalysis & PriceAnalysis & RatingAnalysis
    Data --> DiscountAnalysis & TextAnalysis & CorrAnalysis
    
    CatAnalysis & PriceAnalysis & RatingAnalysis --> Summary
    DiscountAnalysis & TextAnalysis & CorrAnalysis --> Summary
```

### Analysis Output Structure

```json
{
    "dataset_info": {
        "total_products": 10000,
        "total_features": 15
    },
    "categories": {
        "main_categories": {"unique_count": 10, "distribution": {...}},
        "sub_categories": {"unique_count": 50, "distribution": {...}}
    },
    "price_analysis": {
        "actual_price": {"min": 10, "max": 50000, "mean": 1500}
    },
    "rating_analysis": {
        "rating_stats": {"min": 1.0, "max": 5.0, "mean": 4.2}
    },
    "discount_analysis": {
        "discount_stats": {"min": 0, "max": 80, "mean": 25}
    },
    "correlations": {
        "correlations_with_discount": {...}
    }
}
```

## Data Validation

### Input Validation

```mermaid
graph TB
    subgraph "API Input"
        Request[Request Body]
    end
    
    subgraph "Pydantic Validation"
        TypeCheck[Type Checking]
        RangeCheck[Range Validation]
        Required[Required Fields]
    end
    
    subgraph "Business Rules"
        PriceValid["price > 0"]
        RatingValid["0 <= rating <= 5"]
        CountValid["count >= 0"]
    end
    
    Request --> TypeCheck --> RangeCheck --> Required
    RangeCheck --> PriceValid & RatingValid & CountValid
```

### Validation Rules

| Field | Type | Constraints |
|-------|------|-------------|
| `category` | string | Required |
| `actual_price` | float | > 0 |
| `rating` | float | 0-5 |
| `rating_count` | int | >= 0 |
| `question` | string | 3-1000 chars |

## Data Storage

### File System Structure

```
Marketing-Data-Intelligence/
├── data/
│   └── amazon.csv              # Raw dataset
├── models/
│   ├── discount_predictor.joblib    # Current model
│   └── discount_predictor_*.joblib  # Versioned models
└── logs/
    └── app.log                 # Application logs
```

### Docker Volumes

```mermaid
graph LR
    subgraph "Container"
        App[Application]
    end
    
    subgraph "Volumes"
        DataVol[./data:/app/data:ro]
        ModelVol[./models:/app/models]
        LogVol[app_logs:/app/logs]
    end
    
    subgraph "External"
        QdrantVol[qdrant_data]
    end
    
    DataVol --> App
    ModelVol --> App
    LogVol --> App
    App --> QdrantVol
```

## Error Handling in Data Flow

### Error Points and Handling

```mermaid
graph TB
    subgraph "Data Loading"
        E1[File Not Found]
        E2[Parse Error]
        E3[Missing Columns]
    end
    
    subgraph "Processing"
        E4[Invalid Values]
        E5[Transform Error]
    end
    
    subgraph "Storage"
        E6[Qdrant Connection]
        E7[Model Not Found]
    end
    
    subgraph "Handling"
        H1[Use Sample Data]
        H2[Fill Missing Values]
        H3[Return Error Response]
    end
    
    E1 --> H1
    E2 & E3 --> H2
    E4 & E5 --> H2
    E6 & E7 --> H3
```

## Performance Considerations

### Bottlenecks and Solutions

| Operation | Bottleneck | Solution |
|-----------|------------|----------|
| Data Loading | Large CSV | Chunked reading |
| Feature Engineering | Sentiment Analysis | Batch processing |
| Embedding | Model inference | Batch embedding |
| Vector Search | Query latency | Index optimization |
| LLM Generation | API latency | Streaming responses |

### Caching Strategy

```mermaid
graph TB
    subgraph "Cache Layers"
        ModelCache[Model Singleton]
        EmbedderCache[Embedder Singleton]
        AnalysisCache[Analysis Cache]
    end
    
    subgraph "Benefits"
        NoReload[No repeated loading]
        FastInference[Fast inference]
        QuickAnalysis[Quick analysis]
    end
    
    ModelCache --> NoReload
    EmbedderCache --> FastInference
    AnalysisCache --> QuickAnalysis
```

## Related Documentation

- [Architecture Overview](./overview.md) - System architecture
- [ML Pipeline](./ml-pipeline.md) - ML model details
- [RAG System](./rag-system.md) - RAG architecture
- [API Documentation](../api/endpoints.md) - API reference
