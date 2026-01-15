"""Question Answering (RAG) API routes."""

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
import structlog

from src.api.schemas import (
    AnswerQuestionRequest,
    AnswerQuestionResponse,
    IndexDataRequest,
    IndexDataResponse,
    SourceReference,
)
from src.config import settings
from src.llm.gemini_client import get_gemini_client
from src.rag.retriever import get_retriever

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/qa", tags=["Question Answering"])


@router.post("/answer", response_model=AnswerQuestionResponse)
async def answer_question(request: AnswerQuestionRequest) -> AnswerQuestionResponse:
    """Answer a product-related question using RAG.
    
    This endpoint uses retrieval-augmented generation to answer questions
    about products in the catalog. It retrieves relevant product information
    from the vector database and uses Gemini to generate contextual answers.
    """
    logger.info(
        "answer_question_request",
        question=request.question[:100],
        filters={
            "category": request.filter_category,
            "min_price": request.filter_min_price,
            "max_price": request.filter_max_price,
            "min_rating": request.filter_min_rating,
        },
    )
    
    try:
        client = get_gemini_client()
        
        # Build filter kwargs
        filter_kwargs = {}
        if request.filter_category:
            filter_kwargs["filter_category"] = request.filter_category
        if request.filter_min_price is not None:
            filter_kwargs["filter_min_price"] = request.filter_min_price
        if request.filter_max_price is not None:
            filter_kwargs["filter_max_price"] = request.filter_max_price
        if request.filter_min_rating is not None:
            filter_kwargs["filter_min_rating"] = request.filter_min_rating
        if request.top_k is not None:
            filter_kwargs["top_k"] = request.top_k
        
        # Get answer from LLM with RAG
        result = client.answer_question(
            question=request.question,
            use_rag=True,
            **filter_kwargs,
        )
        
        # Convert sources to Pydantic models
        sources = [
            SourceReference(
                product=s["product"],
                relevance=s["relevance"],
                id=s.get("id"),
            )
            for s in result.get("sources", [])
        ]
        
        return AnswerQuestionResponse(
            answer=result["answer"],
            sources=sources,
            grounded=result["grounded"],
            question=result["question"],
        )
        
    except ValueError as e:
        # API key not configured
        raise HTTPException(
            status_code=503,
            detail=str(e),
        )
    except Exception as e:
        logger.error("answer_question_error", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to generate answer: {str(e)}")


@router.post("/answer/stream")
async def stream_answer(request: AnswerQuestionRequest):
    """Stream answer generation for a product question.
    
    This endpoint streams the response as it's being generated,
    providing a better user experience for longer responses.
    """
    logger.info("stream_answer_request", question=request.question[:100])
    
    try:
        client = get_gemini_client()
        retriever = get_retriever()
        
        # Get context from RAG
        context = None
        if retriever.health_check():
            filter_kwargs = {}
            if request.filter_category:
                filter_kwargs["filter_category"] = request.filter_category
            if request.filter_max_price is not None:
                filter_kwargs["filter_max_price"] = request.filter_max_price
            if request.filter_min_rating is not None:
                filter_kwargs["filter_min_rating"] = request.filter_min_rating
            
            context = retriever.get_context_for_query(
                request.question,
                top_k=request.top_k or 5,
                **filter_kwargs,
            )
        
        async def generate():
            for chunk in client.stream_answer(request.question, context):
                yield chunk
        
        return StreamingResponse(
            generate(),
            media_type="text/plain",
        )
        
    except ValueError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error("stream_answer_error", error=str(e))
        raise HTTPException(status_code=500, detail=f"Streaming failed: {str(e)}")


@router.post("/index", response_model=IndexDataResponse)
async def index_data(request: IndexDataRequest) -> IndexDataResponse:
    """Index product data into the RAG system.
    
    This endpoint loads product data from the dataset and indexes it
    into the Qdrant vector database for retrieval.
    """
    from src.data.loader import get_sample_data, load_amazon_dataset
    from src.data.preprocessor import prepare_rag_documents
    from src.rag.indexer import get_indexer
    
    logger.info("index_data_request", recreate=request.recreate_collection)
    
    try:
        # Load data
        try:
            df = load_amazon_dataset()
        except FileNotFoundError:
            logger.info("using_sample_data_for_indexing")
            df = get_sample_data()
        
        # Prepare documents
        documents = prepare_rag_documents(df)
        
        # Index documents
        indexer = get_indexer()
        
        if request.recreate_collection:
            indexer.create_collection(recreate=True)
        
        count = indexer.index_documents(documents)
        
        # Get collection info
        info = indexer.get_collection_info()
        
        return IndexDataResponse(
            status="success",
            documents_indexed=count,
            collection_info=info,
        )
        
    except Exception as e:
        logger.error("index_data_error", error=str(e))
        raise HTTPException(status_code=500, detail=f"Indexing failed: {str(e)}")


@router.get("/search")
async def search_products(
    query: str,
    top_k: int = 5,
    category: str = None,
    min_price: float = None,
    max_price: float = None,
    min_rating: float = None,
):
    """Search for products using semantic search.
    
    This endpoint performs semantic search on the product catalog
    without generating an LLM response. Useful for product discovery.
    """
    logger.info("search_request", query=query[:100], top_k=top_k)
    
    try:
        retriever = get_retriever()
        
        if not retriever.health_check():
            raise HTTPException(
                status_code=503,
                detail="Vector database not available. Please index data first.",
            )
        
        results = retriever.search(
            query=query,
            top_k=top_k,
            filter_category=category,
            filter_min_price=min_price,
            filter_max_price=max_price,
            filter_min_rating=min_rating,
        )
        
        return {"results": results, "count": len(results)}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("search_error", error=str(e))
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.get("/health")
async def rag_health():
    """Check RAG system health."""
    retriever = get_retriever()
    client = get_gemini_client()
    
    qdrant_healthy = retriever.health_check()
    gemini_status = client.health_check()
    
    return {
        "qdrant": {
            "status": "healthy" if qdrant_healthy else "unhealthy",
            "collection": settings.qdrant_collection,
        },
        "gemini": gemini_status,
    }
