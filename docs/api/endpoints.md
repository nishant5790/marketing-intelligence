# API Reference

Complete API documentation for the Marketing Data Intelligence system.

## Base URL

- **Development**: `http://localhost:8000`
- **Production**: Configured via environment

## Authentication

Currently, the API does not require authentication. For production, implement API key or OAuth2.

## Common Response Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 400 | Bad Request - Invalid input |
| 404 | Not Found |
| 422 | Validation Error |
| 500 | Internal Server Error |
| 503 | Service Unavailable (model not loaded, etc.) |

---

## Root Endpoints

### GET /

Get API information.

**Response**

```json
{
    "name": "Marketing Data Intelligence",
    "version": "0.1.0",
    "environment": "development",
    "docs": "/docs",
    "endpoints": {
        "predict_discount": "POST /predict_discount",
        "answer_question": "POST /answer_question",
        "analysis_summary": "GET /analysis/summary",
        "health": "GET /health",
        "metrics": "GET /metrics"
    }
}
```

### GET /health

Check system health status.

**Response**

```json
{
    "status": "healthy",
    "components": {
        "ml_model": {
            "status": "healthy",
            "path": "models/discount_predictor.joblib"
        },
        "qdrant": {
            "status": "healthy",
            "collection": "products"
        },
        "gemini": {
            "status": "healthy",
            "configured": true,
            "available_models": ["gemini-2.0-flash"]
        }
    },
    "version": "0.1.0"
}
```

### GET /metrics

Prometheus metrics endpoint.

**Response**: Plain text Prometheus metrics format

---

## Prediction Endpoints

### POST /predict_discount

Predict optimal discount percentage for a product.

**Request Body**

```json
{
    "category": "Electronics",
    "actual_price": 1499.99,
    "rating": 4.2,
    "rating_count": 1250
}
```

| Field | Type | Required | Constraints | Description |
|-------|------|----------|-------------|-------------|
| `category` | string | Yes | - | Product category |
| `actual_price` | float | Yes | > 0 | Original price |
| `rating` | float | Yes | 0-5 | Product rating |
| `rating_count` | int | Yes | >= 0 | Number of reviews |

**Response** (200)

```json
{
    "predicted_discount": 23.5,
    "confidence": 0.85,
    "explanation": {
        "top_features": ["category", "rating_count", "actual_price"],
        "importance_scores": {
            "category": 0.35,
            "rating_count": 0.28,
            "actual_price": 0.15
        }
    }
}
```

**Errors**

| Code | Condition |
|------|-----------|
| 503 | Model not loaded |
| 422 | Invalid input values |

---

### GET /predict/status

Get ML model status.

**Response** (200)

```json
{
    "loaded": true,
    "model_path": "models/discount_predictor.joblib",
    "metrics": {
        "rmse": 5.23,
        "mae": 3.89,
        "r2": 0.78,
        "mape": 12.5
    },
    "feature_importance": {
        "category": 0.35,
        "rating_count": 0.28,
        "actual_price": 0.15
    }
}
```

---

### POST /predict/train

Train or retrain the ML model.

**Request Body**

```json
{
    "use_sample_data": false,
    "test_size": 0.2
}
```

| Field | Type | Default | Constraints | Description |
|-------|------|---------|-------------|-------------|
| `use_sample_data` | bool | false | - | Use generated sample data |
| `test_size` | float | 0.2 | 0.1-0.5 | Test split ratio |

**Response** (200)

```json
{
    "status": "success",
    "metrics": {
        "rmse": 5.23,
        "mae": 3.89,
        "r2": 0.78,
        "mape": 12.5
    },
    "feature_importance": {
        "category": 0.35,
        "rating_count": 0.28
    },
    "model_path": "models/discount_predictor.joblib",
    "training_samples": 8000,
    "test_samples": 2000
}
```

**Errors**

| Code | Condition |
|------|-----------|
| 404 | Dataset not found |
| 500 | Training failed |

---

### POST /predict/explain

Get SHAP explanation for a prediction.

**Request Body**

```json
{
    "category": "Electronics",
    "actual_price": 1499.99,
    "rating": 4.2,
    "rating_count": 1250
}
```

**Response** (200)

```json
{
    "predicted_discount": 23.5,
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
    "base_value": 28.5,
    "explanation_summary": "The predicted discount is 23.5%. The baseline prediction is 28.5%..."
}
```

---

## Question Answering Endpoints

### POST /answer_question

Answer a product-related question using RAG.

**Request Body**

```json
{
    "question": "What are the best rated headphones under $100?",
    "filter_category": "Electronics",
    "filter_min_price": null,
    "filter_max_price": 100,
    "filter_min_rating": 4.0,
    "top_k": 5
}
```

| Field | Type | Required | Constraints | Description |
|-------|------|----------|-------------|-------------|
| `question` | string | Yes | 3-1000 chars | User question |
| `filter_category` | string | No | - | Filter by category |
| `filter_min_price` | float | No | >= 0 | Minimum price |
| `filter_max_price` | float | No | >= 0 | Maximum price |
| `filter_min_rating` | float | No | 0-5 | Minimum rating |
| `top_k` | int | No | 1-20 | Number of products to consider |

**Response** (200)

```json
{
    "answer": "Based on our catalog, the top-rated headphones under $100 are:\n\n1. **Sony WH-CH520** ($89.99) - Rating: 4.5/5\n...",
    "sources": [
        {
            "product": "Sony WH-CH520",
            "relevance": 0.92,
            "id": "PROD001"
        },
        {
            "product": "JBL Tune 510BT",
            "relevance": 0.87,
            "id": "PROD002"
        }
    ],
    "grounded": true,
    "question": "What are the best rated headphones under $100?"
}
```

**Errors**

| Code | Condition |
|------|-----------|
| 503 | Gemini API not configured |
| 500 | Generation failed |

---

### POST /qa/answer/stream

Stream answer generation.

**Request Body**: Same as `/answer_question`

**Response**: `text/plain` stream

```
Based on our catalog...
```

---

### POST /qa/index

Index product data into the RAG system.

**Request Body**

```json
{
    "recreate_collection": true
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `recreate_collection` | bool | false | Delete existing collection first |

**Response** (200)

```json
{
    "status": "success",
    "documents_indexed": 1465,
    "collection_info": {
        "name": "products",
        "vectors_count": 1465,
        "points_count": 1465,
        "status": "green"
    }
}
```

---

### GET /qa/search

Semantic product search without LLM generation.

**Query Parameters**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `query` | string | Yes | - | Search query |
| `top_k` | int | No | 5 | Number of results |
| `category` | string | No | - | Category filter |
| `min_price` | float | No | - | Minimum price |
| `max_price` | float | No | - | Maximum price |
| `min_rating` | float | No | - | Minimum rating |

**Example**

```
GET /qa/search?query=wireless+headphones&top_k=3&max_price=100
```

**Response** (200)

```json
{
    "results": [
        {
            "id": "uuid-1",
            "score": 0.92,
            "text": "Product: Sony WH-CH520...",
            "product_name": "Sony WH-CH520",
            "category": "Electronics",
            "actual_price": 89.99,
            "discounted_price": 67.49,
            "discount_percentage": 25.0,
            "rating": 4.5,
            "rating_count": 1234
        }
    ],
    "count": 1
}
```

---

### GET /qa/health

Check RAG system health.

**Response** (200)

```json
{
    "qdrant": {
        "status": "healthy",
        "collection": "products"
    },
    "gemini": {
        "status": "healthy",
        "configured": true,
        "available_models": ["gemini-2.0-flash"]
    }
}
```

---

## Analysis Endpoints

### GET /analysis/summary

Get comprehensive dataset analysis.

**Response** (200)

```json
{
    "dataset_info": {
        "total_products": 1465,
        "total_features": 15,
        "memory_usage_mb": 2.5
    },
    "categories": {
        "main_categories": {
            "unique_count": 10,
            "distribution": {
                "Electronics": 500,
                "Clothing": 300
            }
        }
    },
    "price_analysis": {
        "actual_price": {
            "min": 10.0,
            "max": 50000.0,
            "mean": 1500.0,
            "median": 800.0
        }
    },
    "rating_analysis": {
        "rating_stats": {
            "min": 1.0,
            "max": 5.0,
            "mean": 4.2
        }
    },
    "discount_analysis": {
        "discount_stats": {
            "min": 0.0,
            "max": 80.0,
            "mean": 25.0
        }
    }
}
```

---

### GET /analysis/top-products

Get top products by metric.

**Query Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `by` | string | rating | Sort metric |
| `n` | int | 10 | Number of results |
| `category` | string | - | Filter by category |

**Example**

```
GET /analysis/top-products?by=discount_percentage&n=5&category=Electronics
```

**Response** (200)

```json
[
    {
        "product_id": "PROD001",
        "product_name": "Sony WH-1000XM5",
        "main_category": "Electronics",
        "discount_percentage": 35.0,
        "actual_price": 349.99,
        "rating": 4.8,
        "rating_count": 12500
    }
]
```

---

## Error Response Format

All errors follow this format:

```json
{
    "detail": "Error message here",
    "error_type": "ValueError"
}
```

### Validation Errors (422)

```json
{
    "detail": [
        {
            "type": "value_error",
            "loc": ["body", "rating"],
            "msg": "Input should be less than or equal to 5",
            "input": 6.5
        }
    ]
}
```

---

## Rate Limiting

Currently not implemented. For production, consider:

```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1609459200
```

---

## Request Headers

### Recommended Headers

| Header | Description |
|--------|-------------|
| `Content-Type` | `application/json` for POST requests |
| `X-Request-ID` | Correlation ID for request tracing |

### Response Headers

| Header | Description |
|--------|-------------|
| `X-Response-Time` | Request processing time |

---

## SDK Examples

### Python

```python
import httpx

# Predict discount
response = httpx.post(
    "http://localhost:8000/predict_discount",
    json={
        "category": "Electronics",
        "actual_price": 999.99,
        "rating": 4.5,
        "rating_count": 1000
    }
)
result = response.json()
print(f"Predicted discount: {result['predicted_discount']}%")

# Ask question
response = httpx.post(
    "http://localhost:8000/answer_question",
    json={
        "question": "What are the best deals on laptops?",
        "filter_category": "Electronics"
    }
)
result = response.json()
print(result["answer"])
```

### cURL

```bash
# Predict discount
curl -X POST http://localhost:8000/predict_discount \
  -H "Content-Type: application/json" \
  -d '{"category": "Electronics", "actual_price": 999.99, "rating": 4.5, "rating_count": 1000}'

# Ask question
curl -X POST http://localhost:8000/answer_question \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the best deals?", "filter_max_price": 100}'
```

### JavaScript

```javascript
// Predict discount
const response = await fetch('http://localhost:8000/predict_discount', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
        category: 'Electronics',
        actual_price: 999.99,
        rating: 4.5,
        rating_count: 1000
    })
});
const result = await response.json();
console.log(`Predicted discount: ${result.predicted_discount}%`);
```

---

## API Documentation

Interactive documentation is available at:

- **Swagger UI**: `/docs`
- **ReDoc**: `/redoc`
- **OpenAPI JSON**: `/openapi.json`
