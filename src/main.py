"""FastAPI application entry point."""

import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram, generate_latest
from starlette.responses import Response

from src import __version__
from src.api.routes import predict, qa, analysis
from src.api.schemas import HealthResponse
from src.config import settings

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"],
)

REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "endpoint"],
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """Application lifespan handler."""
    logger.info(
        "application_starting",
        app_name=settings.app_name,
        environment=settings.app_env,
        version=__version__,
    )
    
    # Startup: Try to load the ML model if it exists
    try:
        from src.ml.predictor import get_predictor
        predictor = get_predictor()
        if predictor.is_loaded:
            logger.info("ml_model_loaded", path=str(predictor.model_path))
    except Exception as e:
        logger.warning("ml_model_not_loaded", error=str(e))
    
    yield
    
    # Shutdown
    logger.info("application_shutting_down")


# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    description="Intelligent ML system for marketing data analysis with discount prediction and RAG-powered Q&A",
    version=__version__,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    """Middleware to collect request metrics."""
    start_time = time.time()
    
    response = await call_next(request)
    
    # Record metrics
    duration = time.time() - start_time
    endpoint = request.url.path
    method = request.method
    status = response.status_code
    
    REQUEST_COUNT.labels(method=method, endpoint=endpoint, status=status).inc()
    REQUEST_LATENCY.labels(method=method, endpoint=endpoint).observe(duration)
    
    # Add timing header
    response.headers["X-Response-Time"] = f"{duration:.4f}s"
    
    return response


@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    """Middleware to log requests."""
    request_id = request.headers.get("X-Request-ID", str(time.time_ns()))
    
    logger.info(
        "request_started",
        request_id=request_id,
        method=request.method,
        path=request.url.path,
        client=request.client.host if request.client else None,
    )
    
    response = await call_next(request)
    
    logger.info(
        "request_completed",
        request_id=request_id,
        status_code=response.status_code,
    )
    
    return response


# Include routers
app.include_router(predict.router)
app.include_router(qa.router)
app.include_router(analysis.router)


# Convenience endpoints at root level
@app.post("/predict_discount")
async def predict_discount_shortcut(request: predict.PredictDiscountRequest):
    """Shortcut endpoint for discount prediction."""
    return await predict.predict_discount(request)


@app.post("/answer_question")
async def answer_question_shortcut(request: qa.AnswerQuestionRequest):
    """Shortcut endpoint for question answering."""
    return await qa.answer_question(request)


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Check overall system health."""
    from src.ml.predictor import get_predictor
    from src.rag.retriever import get_retriever
    from src.llm.gemini_client import get_gemini_client
    
    # Check components
    predictor = get_predictor()
    retriever = get_retriever()
    gemini_client = get_gemini_client()
    
    components = {
        "ml_model": {
            "status": "healthy" if predictor.is_loaded else "not_loaded",
            "path": str(predictor.model_path),
        },
        "qdrant": {
            "status": "healthy" if retriever.health_check() else "unhealthy",
            "collection": settings.qdrant_collection,
        },
        "gemini": gemini_client.health_check(),
    }
    
    # Determine overall status
    all_healthy = predictor.is_loaded and retriever.health_check()
    status = "healthy" if all_healthy else "degraded"
    
    return HealthResponse(
        status=status,
        components=components,
        version=__version__,
    )


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(
        content=generate_latest(),
        media_type="text/plain",
    )


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": settings.app_name,
        "version": __version__,
        "environment": settings.app_env,
        "docs": "/docs",
        "endpoints": {
            "predict_discount": "POST /predict_discount",
            "answer_question": "POST /answer_question",
            "analysis_summary": "GET /analysis/summary",
            "health": "GET /health",
            "metrics": "GET /metrics",
        },
    }


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger.error(
        "unhandled_exception",
        path=request.url.path,
        method=request.method,
        error=str(exc),
        error_type=type(exc).__name__,
    )
    
    return JSONResponse(
        status_code=500,
        content={
            "detail": "An unexpected error occurred",
            "error_type": type(exc).__name__,
        },
    )


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "src.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
    )
