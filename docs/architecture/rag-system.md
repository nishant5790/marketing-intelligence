# RAG (Retrieval-Augmented Generation) System

This document details the RAG architecture for product Q&A functionality.

## Overview

The RAG system combines semantic search with LLM generation to answer product-related questions grounded in actual catalog data.

```mermaid
graph TB
    subgraph "RAG Pipeline"
        Query[User Query]
        Embed[Query Embedding]
        Search[Vector Search]
        Context[Context Building]
        Prompt[Prompt Construction]
        Generate[LLM Generation]
        Response[Grounded Response]
    end
    
    subgraph "Components"
        Embedder[Sentence Transformers<br/>all-MiniLM-L6-v2]
        VectorDB[(Qdrant)]
        LLM[Google Gemini<br/>2.0 Flash]
    end
    
    Query --> Embed
    Embed --> Search
    Search --> Context
    Context --> Prompt
    Prompt --> Generate
    Generate --> Response
    
    Embed -.-> Embedder
    Search -.-> VectorDB
    Generate -.-> LLM
```

## System Components

### 1. Text Embedder

```mermaid
graph LR
    subgraph "Embedding Pipeline"
        Text[Input Text]
        Tokenize[Tokenization]
        Encode[BERT Encoding]
        Pool[Mean Pooling]
        Normalize[L2 Normalization]
        Vector[384-dim Vector]
    end
    
    Text --> Tokenize --> Encode --> Pool --> Normalize --> Vector
```

#### Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| Model | `all-MiniLM-L6-v2` | Sentence Transformers model |
| Dimension | 384 | Embedding vector size |
| Normalization | L2 | Enables cosine similarity via dot product |
| Batch Size | 32 | Documents per batch |

### 2. Vector Database (Qdrant)

```mermaid
graph TB
    subgraph "Qdrant Collection"
        Collection[products collection]
        
        subgraph "Vector Storage"
            V1[Vector 1]
            V2[Vector 2]
            VN[Vector N]
        end
        
        subgraph "Payload Storage"
            P1[Payload 1]
            P2[Payload 2]
            PN[Payload N]
        end
        
        subgraph "Indexes"
            CatIdx[category index]
            PriceIdx[price index]
            RatingIdx[rating index]
        end
    end
    
    Collection --> V1 & V2 & VN
    V1 --> P1
    V2 --> P2
    VN --> PN
    
    Collection --> CatIdx & PriceIdx & RatingIdx
```

#### Collection Schema

```json
{
    "name": "products",
    "vectors_config": {
        "size": 384,
        "distance": "Cosine"
    },
    "payload_indexes": [
        {"field_name": "category", "type": "keyword"},
        {"field_name": "actual_price", "type": "float"},
        {"field_name": "rating", "type": "float"}
    ]
}
```

#### Document Payload

```json
{
    "text": "Product description...",
    "product_id": "PROD001",
    "product_name": "Sony WH-1000XM5",
    "category": "Electronics",
    "actual_price": 349.99,
    "discounted_price": 279.99,
    "discount_percentage": 20.0,
    "rating": 4.8,
    "rating_count": 12500
}
```

### 3. Document Indexer

```mermaid
sequenceDiagram
    participant API as API
    participant I as Indexer
    participant E as Embedder
    participant Q as Qdrant
    
    API->>I: index_documents(documents)
    
    loop For each batch
        I->>E: embed_documents(texts)
        E-->>I: embeddings[]
        
        I->>I: Prepare points with payloads
        
        I->>Q: upsert(points)
        Q-->>I: Success
    end
    
    I-->>API: Documents indexed count
```

### 4. Document Retriever

```mermaid
graph TB
    subgraph "Search Flow"
        Query[Search Query]
        EmbedQ[Embed Query]
        BuildFilter[Build Filters]
        VectorSearch[Vector Search]
        FormatResults[Format Results]
        Results[Search Results]
    end
    
    subgraph "Filters"
        CatFilter[Category Filter]
        PriceFilter[Price Range]
        RatingFilter[Min Rating]
    end
    
    Query --> EmbedQ --> VectorSearch
    BuildFilter --> VectorSearch
    
    CatFilter --> BuildFilter
    PriceFilter --> BuildFilter
    RatingFilter --> BuildFilter
    
    VectorSearch --> FormatResults --> Results
```

#### Search Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `top_k` | 5 | Number of results |
| `score_threshold` | 0.7 | Minimum similarity |
| `filter_category` | None | Category filter |
| `filter_min_price` | None | Minimum price |
| `filter_max_price` | None | Maximum price |
| `filter_min_rating` | None | Minimum rating |

### 5. LLM Client (Gemini)

```mermaid
graph TB
    subgraph "Gemini Integration"
        Client[Gemini Client]
        Config[Generation Config]
        SystemPrompt[System Prompt]
        QAPrompt[Q&A Prompt]
        Generate[Generate Content]
        Stream[Stream Content]
    end
    
    Client --> Config
    Config --> SystemPrompt
    
    SystemPrompt --> Generate
    QAPrompt --> Generate
    
    SystemPrompt --> Stream
    QAPrompt --> Stream
```

#### Generation Configuration

```python
config = {
    "temperature": 0.7,
    "max_output_tokens": 1024,
    "top_p": 0.9,
    "top_k": 40,
    "system_instruction": SYSTEM_PROMPT
}
```

## RAG Workflow

### Complete Q&A Flow

```mermaid
sequenceDiagram
    participant U as User
    participant API as FastAPI
    participant R as Retriever
    participant E as Embedder
    participant Q as Qdrant
    participant G as Gemini
    
    U->>API: POST /answer_question
    Note over API: {question, filters}
    
    API->>R: search(query, filters)
    R->>E: embed_query(question)
    E-->>R: query_vector
    
    R->>Q: search(vector, filters)
    Q-->>R: similar_documents
    
    R-->>API: search_results
    
    API->>API: Build context from results
    API->>API: Construct prompt
    
    API->>G: generate(prompt)
    G-->>API: answer
    
    API-->>U: {answer, sources, grounded}
```

### Context Building

```mermaid
graph LR
    subgraph "Context Construction"
        Results[Search Results]
        Format[Format Each Result]
        Combine[Combine into Context]
        Context[Context String]
    end
    
    Results --> Format --> Combine --> Context
```

#### Context Format

```
Relevant products from our catalog:

1. Sony WH-1000XM5
   Category: Electronics
   Price: $349.99
   Discount: 20.0%
   Rating: 4.8/5 (12500 reviews)
   Details: Premium noise-canceling wireless headphones...
   (Relevance: 92.5%)

2. Bose QuietComfort...
```

## Prompt Engineering

### System Prompt

```
You are an intelligent marketing assistant for an e-commerce platform.
Your role is to help users find products, answer questions about our 
catalog, and provide recommendations.

Guidelines:
1. Always base your answers on the provided product context
2. Be helpful, accurate, and concise
3. Never make up product information
4. Include relevant details like price, rating, discount
5. Format prices in a user-friendly way
```

### Q&A Prompt Template

```mermaid
graph TB
    subgraph "Prompt Structure"
        Context[Product Context]
        Question[User Question]
        Instructions[Instructions]
        Answer[Answer Placeholder]
    end
    
    Context --> Template
    Question --> Template
    Instructions --> Template
    Answer --> Template
    
    Template[Complete Prompt]
```

### Few-Shot Examples

The system includes few-shot examples for:
- Product comparisons
- Deal/discount queries
- Recommendation requests

## Data Indexing Pipeline

### Document Preparation

```mermaid
graph TB
    subgraph "Data Processing"
        CSV[(Amazon CSV)]
        Load[Load Dataset]
        Clean[Clean Data]
        PrepDocs[Prepare Documents]
        Index[Index to Qdrant]
    end
    
    CSV --> Load --> Clean --> PrepDocs --> Index
```

### Document Text Structure

```python
text_parts = [
    f"Product: {product_name}",
    f"Category: {main_category}",
    f"Sub-category: {sub_category}",
    f"Description: {about_product[:800]}",
    f"Reviews: {review_content[:500]}",
]
document_text = "\n".join(text_parts)
```

## Filtering Capabilities

### Filter Types

```mermaid
graph TB
    subgraph "Filter Conditions"
        Keyword[Keyword Match<br/>category = 'Electronics']
        Range[Range Filter<br/>price >= 50, price <= 200]
        Combined[Combined Filters<br/>AND logic]
    end
    
    Keyword --> Combined
    Range --> Combined
```

### Filter Implementation

```python
filter_conditions = []

if filter_category:
    filter_conditions.append(
        FieldCondition(key="category", match=MatchValue(value=filter_category))
    )

if filter_max_price:
    filter_conditions.append(
        FieldCondition(key="actual_price", range=Range(lte=filter_max_price))
    )

query_filter = Filter(must=filter_conditions)
```

## Performance Optimization

### Embedding Caching

```mermaid
graph LR
    Query[Query] --> Cache{In Cache?}
    Cache -->|Yes| Return[Return Cached]
    Cache -->|No| Compute[Compute Embedding]
    Compute --> Store[Store in Cache]
    Store --> Return
```

### Batch Processing

```python
# Index documents in batches
for i in range(0, len(documents), batch_size):
    batch = documents[i:i + batch_size]
    embeddings = embedder.embed_documents([d["text"] for d in batch])
    indexer.upsert(batch, embeddings)
```

## Error Handling

### Graceful Degradation

```mermaid
graph TB
    Request[Request]
    CheckQdrant{Qdrant Available?}
    CheckGemini{Gemini Available?}
    
    Request --> CheckQdrant
    CheckQdrant -->|Yes| RAGFlow[Full RAG]
    CheckQdrant -->|No| NoContext[Answer without context]
    
    RAGFlow --> CheckGemini
    NoContext --> CheckGemini
    
    CheckGemini -->|Yes| Generate[Generate Answer]
    CheckGemini -->|No| Error[Return Error]
```

### Health Checks

```python
def health_check():
    return {
        "qdrant": retriever.health_check(),  # Collection exists?
        "gemini": gemini_client.health_check()  # API accessible?
    }
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/qa/answer` | POST | Answer question with RAG |
| `/qa/answer/stream` | POST | Stream answer generation |
| `/qa/index` | POST | Index product data |
| `/qa/search` | GET | Semantic product search |
| `/qa/health` | GET | RAG system health |

## Monitoring

### Metrics

- `qa_requests_total` - Total Q&A requests
- `qa_latency_seconds` - Response time histogram
- `rag_results_count` - Number of retrieved results
- `grounded_responses` - % of grounded answers

## Best Practices

### Indexing
1. Re-index when data changes significantly
2. Use batch processing for large datasets
3. Include all searchable fields in text

### Retrieval
1. Tune `score_threshold` based on precision needs
2. Use filters to narrow search space
3. Return enough context (top_k=5-10)

### Generation
1. Include clear instructions in prompts
2. Use few-shot examples for consistency
3. Always cite sources in responses

## Related Documentation

- [Architecture Overview](./overview.md) - System architecture
- [ML Pipeline](./ml-pipeline.md) - ML model details
- [API Documentation](../api/endpoints.md) - API reference
