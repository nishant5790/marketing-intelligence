"""Document retrieval from Qdrant vector database."""

from typing import Any, List, Optional

import structlog
from qdrant_client import QdrantClient
from qdrant_client.http import models

from src.config import settings
from src.rag.embedder import get_embedder

logger = structlog.get_logger(__name__)


class DocumentRetriever:
    """Retrieve relevant documents from Qdrant."""

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        collection_name: Optional[str] = None,
    ):
        """Initialize the retriever.
        
        Args:
            host: Qdrant server host.
            port: Qdrant server port.
            collection_name: Name of the collection.
        """
        self.host = host or settings.qdrant_host
        self.port = port or settings.qdrant_port
        self.collection_name = collection_name or settings.qdrant_collection
        self._client: Optional[QdrantClient] = None
        self.embedder = get_embedder()

    @property
    def client(self) -> QdrantClient:
        """Get or create Qdrant client (lazy loaded)."""
        if self._client is None:
            self._client = QdrantClient(host=self.host, port=self.port)
        return self._client

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
        filter_category: Optional[str] = None,
        filter_min_price: Optional[float] = None,
        filter_max_price: Optional[float] = None,
        filter_min_rating: Optional[float] = None,
    ) -> List[dict]:
        """Search for relevant documents.
        
        Args:
            query: Search query text.
            top_k: Number of results to return.
            score_threshold: Minimum similarity score.
            filter_category: Filter by category.
            filter_min_price: Minimum price filter.
            filter_max_price: Maximum price filter.
            filter_min_rating: Minimum rating filter.
            
        Returns:
            List of result dictionaries with score and payload.
        """
        top_k = top_k or settings.top_k_results
        score_threshold = score_threshold or settings.similarity_threshold
        
        logger.info(
            "searching",
            query=query[:100],
            top_k=top_k,
            threshold=score_threshold,
        )
        
        # Generate query embedding
        query_embedding = self.embedder.embed_query(query)
        
        # Build filters
        filter_conditions = []
        
        if filter_category:
            filter_conditions.append(
                models.FieldCondition(
                    key="category",
                    match=models.MatchValue(value=filter_category),
                )
            )
        
        if filter_min_price is not None:
            filter_conditions.append(
                models.FieldCondition(
                    key="actual_price",
                    range=models.Range(gte=filter_min_price),
                )
            )
        
        if filter_max_price is not None:
            filter_conditions.append(
                models.FieldCondition(
                    key="actual_price",
                    range=models.Range(lte=filter_max_price),
                )
            )
        
        if filter_min_rating is not None:
            filter_conditions.append(
                models.FieldCondition(
                    key="rating",
                    range=models.Range(gte=filter_min_rating),
                )
            )
        
        # Create filter if conditions exist
        query_filter = None
        if filter_conditions:
            query_filter = models.Filter(must=filter_conditions)
        
        # Execute search
        try:
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding.tolist(),
                limit=top_k,
                score_threshold=score_threshold,
                query_filter=query_filter,
            )
        except Exception as e:
            logger.error("search_error", error=str(e))
            return []
        
        # Format results
        formatted_results = []
        for result in results:
            formatted_results.append({
                "id": result.id,
                "score": float(result.score),
                "text": result.payload.get("text", ""),
                "product_name": result.payload.get("product_name", "Unknown"),
                "category": result.payload.get("category"),
                "actual_price": result.payload.get("actual_price"),
                "discounted_price": result.payload.get("discounted_price"),
                "discount_percentage": result.payload.get("discount_percentage"),
                "rating": result.payload.get("rating"),
                "rating_count": result.payload.get("rating_count"),
            })
        
        logger.info("search_completed", results=len(formatted_results))
        
        return formatted_results

    def get_context_for_query(
        self,
        query: str,
        top_k: int = 5,
        **filters: Any,
    ) -> str:
        """Get formatted context string for LLM from search results.
        
        Args:
            query: Search query.
            top_k: Number of results.
            **filters: Additional filters.
            
        Returns:
            Formatted context string.
        """
        results = self.search(query, top_k=top_k, **filters)
        
        if not results:
            return "No relevant products found in the database."
        
        context_parts = ["Relevant products from our catalog:\n"]
        
        for i, result in enumerate(results, 1):
            product_info = [f"\n{i}. {result['product_name']}"]
            
            if result.get("category"):
                product_info.append(f"   Category: {result['category']}")
            
            if result.get("actual_price"):
                product_info.append(f"   Price: ${result['actual_price']:.2f}")
            
            if result.get("discount_percentage"):
                product_info.append(f"   Discount: {result['discount_percentage']:.1f}%")
            
            if result.get("rating"):
                rating_str = f"   Rating: {result['rating']:.1f}/5"
                if result.get("rating_count"):
                    rating_str += f" ({result['rating_count']} reviews)"
                product_info.append(rating_str)
            
            # Add truncated text content
            if result.get("text"):
                text = result["text"][:300]
                if len(result["text"]) > 300:
                    text += "..."
                product_info.append(f"   Details: {text}")
            
            product_info.append(f"   (Relevance: {result['score']:.2%})")
            
            context_parts.append("\n".join(product_info))
        
        return "\n".join(context_parts)

    def get_sources(self, results: List[dict]) -> List[dict]:
        """Extract source information from search results.
        
        Args:
            results: Search results.
            
        Returns:
            List of source dictionaries.
        """
        return [
            {
                "product": result.get("product_name", "Unknown"),
                "relevance": round(result.get("score", 0), 2),
                "id": result.get("id"),
            }
            for result in results
        ]

    def health_check(self) -> bool:
        """Check if Qdrant is healthy and collection exists.
        
        Returns:
            True if healthy, False otherwise.
        """
        try:
            collections = self.client.get_collections().collections
            collection_names = [c.name for c in collections]
            return self.collection_name in collection_names
        except Exception as e:
            logger.error("health_check_failed", error=str(e))
            return False


# Singleton instance
_retriever_instance: Optional[DocumentRetriever] = None


def get_retriever() -> DocumentRetriever:
    """Get or create the singleton retriever instance.
    
    Returns:
        DocumentRetriever instance.
    """
    global _retriever_instance
    
    if _retriever_instance is None:
        _retriever_instance = DocumentRetriever()
    
    return _retriever_instance
