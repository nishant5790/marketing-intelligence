"""Document indexing for Qdrant vector database."""

import uuid
from typing import List, Optional

import structlog
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams

from src.config import settings
from src.rag.embedder import get_embedder

logger = structlog.get_logger(__name__)


class DocumentIndexer:
    """Index documents into Qdrant vector database."""

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        collection_name: Optional[str] = None,
    ):
        """Initialize the indexer.
        
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
            logger.info(
                "connecting_to_qdrant",
                host=self.host,
                port=self.port,
            )
            self._client = QdrantClient(host=self.host, port=self.port)
        return self._client

    def create_collection(
        self,
        recreate: bool = False,
    ) -> None:
        """Create the vector collection.
        
        Args:
            recreate: If True, delete existing collection first.
        """
        # Check if collection exists
        collections = self.client.get_collections().collections
        collection_names = [c.name for c in collections]
        
        if self.collection_name in collection_names:
            if recreate:
                logger.info("deleting_existing_collection", name=self.collection_name)
                self.client.delete_collection(self.collection_name)
            else:
                logger.info("collection_exists", name=self.collection_name)
                return
        
        # Create collection
        logger.info(
            "creating_collection",
            name=self.collection_name,
            vector_size=self.embedder.embedding_dim,
        )
        
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=self.embedder.embedding_dim,
                distance=Distance.COSINE,
            ),
        )
        
        # Create payload indexes for filtering
        self.client.create_payload_index(
            collection_name=self.collection_name,
            field_name="category",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )
        
        self.client.create_payload_index(
            collection_name=self.collection_name,
            field_name="actual_price",
            field_schema=models.PayloadSchemaType.FLOAT,
        )
        
        self.client.create_payload_index(
            collection_name=self.collection_name,
            field_name="rating",
            field_schema=models.PayloadSchemaType.FLOAT,
        )
        
        logger.info("collection_created", name=self.collection_name)

    def index_documents(
        self,
        documents: List[dict],
        batch_size: int = 100,
    ) -> int:
        """Index documents into Qdrant.
        
        Args:
            documents: List of document dicts with 'id', 'text', and 'metadata'.
            batch_size: Batch size for indexing.
            
        Returns:
            Number of documents indexed.
        """
        if not documents:
            logger.warning("no_documents_to_index")
            return 0
        
        logger.info("indexing_documents", count=len(documents))
        
        # Ensure collection exists
        self.create_collection(recreate=False)
        
        # Extract texts for embedding
        texts = [doc["text"] for doc in documents]
        
        # Generate embeddings
        embeddings = self.embedder.embed_documents(texts, batch_size=batch_size)
        
        # Prepare points for upsert
        points = []
        for i, doc in enumerate(documents):
            point_id = doc.get("id", str(uuid.uuid4()))
            
            # Ensure point_id is a valid format (UUID or integer)
            try:
                # Try to parse as UUID
                point_id = str(uuid.UUID(point_id))
            except ValueError:
                # Generate new UUID for invalid IDs
                point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, str(point_id)))
            
            payload = {
                "text": doc["text"],
                **doc.get("metadata", {}),
            }
            
            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=embeddings[i].tolist(),
                    payload=payload,
                )
            )
        
        # Upsert in batches
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            self.client.upsert(
                collection_name=self.collection_name,
                points=batch,
            )
            logger.debug("batch_indexed", start=i, size=len(batch))
        
        logger.info("documents_indexed", count=len(documents))
        
        return len(documents)

    def delete_collection(self) -> None:
        """Delete the collection."""
        logger.info("deleting_collection", name=self.collection_name)
        self.client.delete_collection(self.collection_name)

    def get_collection_info(self) -> dict:
        """Get collection information.
        
        Returns:
            Dictionary with collection info.
        """
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "name": self.collection_name,
                "vectors_count": info.vectors_count,
                "points_count": info.points_count,
                "status": info.status.value,
            }
        except Exception as e:
            logger.warning("collection_info_error", error=str(e))
            return {"name": self.collection_name, "error": str(e)}


# Singleton instance
_indexer_instance: Optional[DocumentIndexer] = None


def get_indexer() -> DocumentIndexer:
    """Get or create the singleton indexer instance.
    
    Returns:
        DocumentIndexer instance.
    """
    global _indexer_instance
    
    if _indexer_instance is None:
        _indexer_instance = DocumentIndexer()
    
    return _indexer_instance
