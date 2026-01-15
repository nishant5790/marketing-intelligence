# Development Setup Guide

Complete guide for setting up the Marketing Data Intelligence development environment.

## Prerequisites

### Required Software

| Software | Version | Purpose |
|----------|---------|---------|
| Python | 3.11+ | Runtime |
| uv | Latest | Package manager |
| Docker | 20.10+ | Containers |
| Git | 2.0+ | Version control |

### Optional Software

| Software | Purpose |
|----------|---------|
| VS Code | IDE with Python support |
| Cursor | AI-powered IDE |
| Postman | API testing |

## Quick Setup

```bash
# 1. Clone repository
git clone https://github.com/your-org/Marketing-Data-Intelligence.git
cd Marketing-Data-Intelligence

# 2. Install dependencies
uv sync --all-extras

# 3. Create environment file
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY

# 4. Start Qdrant
docker run -d -p 6333:6333 --name qdrant qdrant/qdrant

# 5. Run the application
uv run uvicorn src.main:app --reload

# 6. Open in browser
open http://localhost:8000/docs
```

## Detailed Setup

### 1. Python Environment

#### Using uv (Recommended)

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv sync --all-extras

# Activate (optional, uv run handles this)
source .venv/bin/activate
```

#### Using pip (Alternative)

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e ".[dev]"
```

### 2. Environment Configuration

Create `.env` file:

```bash
# Application
APP_ENV=development
DEBUG=true
LOG_LEVEL=DEBUG

# API
API_HOST=0.0.0.0
API_PORT=8000

# Gemini (Required)
GEMINI_API_KEY=your-api-key-here

# Qdrant
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION=products

# ML
MODEL_PATH=models/discount_predictor.joblib
DRIFT_THRESHOLD=0.1

# RAG
EMBEDDING_MODEL=all-MiniLM-L6-v2
TOP_K_RESULTS=5
SIMILARITY_THRESHOLD=0.7
```

### 3. Data Setup

Place the Amazon dataset at `data/amazon.csv`:

```
Marketing-Data-Intelligence/
└── data/
    └── amazon.csv
```

Expected columns:
- `product_id`, `product_name`
- `category`
- `actual_price`, `discounted_price`, `discount_percentage`
- `rating`, `rating_count`
- `about_product`, `review_content`

### 4. Qdrant Setup

#### Docker (Recommended)

```bash
# Start Qdrant
docker run -d \
  -p 6333:6333 \
  -p 6334:6334 \
  --name qdrant \
  -v qdrant_data:/qdrant/storage \
  qdrant/qdrant

# Verify
curl http://localhost:6333/healthz
```

#### Local Installation

```bash
# Download binary
curl -L https://github.com/qdrant/qdrant/releases/latest/download/qdrant-x86_64-unknown-linux-gnu.tar.gz | tar xz

# Run
./qdrant
```

## Running the Application

### Development Server

```bash
# With auto-reload
uv run uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

# Or using python directly
uv run python -m src.main
```

### Initialize Data

```bash
# Train ML model
curl -X POST http://localhost:8000/predict/train \
  -H "Content-Type: application/json" \
  -d '{"use_sample_data": true}'

# Index RAG data
curl -X POST http://localhost:8000/qa/index \
  -H "Content-Type: application/json" \
  -d '{"recreate_collection": true}'
```

## Project Structure

```
Marketing-Data-Intelligence/
├── src/                    # Application source code
│   ├── __init__.py
│   ├── main.py            # FastAPI application
│   ├── config.py          # Settings management
│   ├── api/               # API routes and schemas
│   │   ├── routes/
│   │   │   ├── predict.py
│   │   │   ├── qa.py
│   │   │   └── analysis.py
│   │   └── schemas.py
│   ├── ml/                # Machine learning
│   │   ├── predictor.py
│   │   ├── trainer.py
│   │   ├── explainer.py
│   │   └── drift.py
│   ├── rag/               # RAG system
│   │   ├── embedder.py
│   │   ├── indexer.py
│   │   └── retriever.py
│   ├── llm/               # LLM integration
│   │   ├── gemini_client.py
│   │   └── prompts.py
│   ├── data/              # Data processing
│   │   ├── loader.py
│   │   └── preprocessor.py
│   ├── analysis/          # EDA
│   │   └── eda.py
│   └── observability/     # Logging & metrics
│       ├── logging.py
│       └── metrics.py
├── tests/                 # Test suite
│   ├── unit/
│   ├── integration/
│   └── load/
├── docs/                  # Documentation
├── data/                  # Dataset directory
├── models/                # ML model storage
├── monitoring/            # Prometheus/Grafana config
├── pyproject.toml         # Project configuration
├── Dockerfile
├── docker-compose.yml
└── README.md
```

## IDE Setup

### VS Code

#### Recommended Extensions

```json
{
  "recommendations": [
    "ms-python.python",
    "ms-python.vscode-pylance",
    "charliermarsh.ruff",
    "ms-azuretools.vscode-docker",
    "redhat.vscode-yaml"
  ]
}
```

#### Settings

```json
{
  "python.defaultInterpreterPath": ".venv/bin/python",
  "python.analysis.typeCheckingMode": "basic",
  "[python]": {
    "editor.formatOnSave": true,
    "editor.defaultFormatter": "charliermarsh.ruff"
  },
  "ruff.organizeImports": true
}
```

### Cursor

The project works well with Cursor AI IDE. Use the built-in AI features for:
- Code completion
- Documentation generation
- Test generation
- Bug fixing

## Common Tasks

### Adding a New Endpoint

1. Define schema in `src/api/schemas.py`:

```python
class NewRequest(BaseModel):
    field: str = Field(..., description="Description")

class NewResponse(BaseModel):
    result: str
```

2. Create route in `src/api/routes/`:

```python
@router.post("/new", response_model=NewResponse)
async def new_endpoint(request: NewRequest) -> NewResponse:
    # Implementation
    return NewResponse(result="...")
```

3. Register in `src/main.py`:

```python
from src.api.routes import new_route
app.include_router(new_route.router)
```

### Adding a New Feature

1. Create feature module in appropriate directory
2. Add tests in `tests/unit/`
3. Update documentation
4. Run linter and tests

### Debugging

```python
# Add breakpoint
import pdb; pdb.set_trace()

# Or use debugger
breakpoint()
```

### Running Specific Modules

```bash
# Run trainer directly
uv run python -m src.ml.trainer --sample

# Test embedder
uv run python -c "from src.rag.embedder import get_embedder; print(get_embedder().embedding_dim)"
```

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| Import errors | Run `uv sync` |
| Qdrant connection | Check Docker is running |
| Model not found | Run training endpoint |
| Gemini API error | Check API key in `.env` |

### Reset Development Environment

```bash
# Remove virtual environment
rm -rf .venv

# Remove cached files
find . -type d -name __pycache__ -exec rm -rf {} +
find . -type f -name "*.pyc" -delete

# Reinstall
uv sync --all-extras
```

### Debug Logging

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Or set in .env
LOG_LEVEL=DEBUG
```

## Next Steps

- [Testing Guide](./testing.md) - Run and write tests
- [API Documentation](../api/endpoints.md) - API reference
- [Architecture](../architecture/overview.md) - System design
