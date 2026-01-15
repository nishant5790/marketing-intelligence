"""Text embedding utilities using Sentence Transformers."""

from typing import List, Optional, Union

import numpy as np
import structlog
from sentence_transformers import SentenceTransformer

from src.config import settings

logger = structlog.get_logger(__name__)


class TextEmbedder:
    """Text embedding using Sentence Transformers."""

    def __init__(self, model_name: Optional[str] = None):
        """Initialize the embedder.
        
        Args:
            model_name: Name of the sentence transformer model.
        """
        self.model_name = model_name or settings.embedding_model
        self._model: Optional[SentenceTransformer] = None
        self._embedding_dim: Optional[int] = None

    def _load_model(self) -> SentenceTransformer:
        """Lazy load the sentence transformer model.
        
        Returns:
            Loaded SentenceTransformer model.
        """
        if self._model is None:
            logger.info("loading_embedding_model", model=self.model_name)
            self._model = SentenceTransformer(self.model_name)
            self._embedding_dim = self._model.get_sentence_embedding_dimension()
            logger.info(
                "embedding_model_loaded",
                model=self.model_name,
                embedding_dim=self._embedding_dim,
            )
        return self._model

    @property
    def model(self) -> SentenceTransformer:
        """Get the sentence transformer model (lazy loaded)."""
        return self._load_model()

    @property
    def embedding_dim(self) -> int:
        """Get the embedding dimension."""
        if self._embedding_dim is None:
            self._load_model()
        return self._embedding_dim  # type: ignore

    def embed(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        show_progress: bool = False,
        normalize: bool = True,
    ) -> np.ndarray:
        """Generate embeddings for texts.
        
        Args:
            texts: Single text or list of texts to embed.
            batch_size: Batch size for encoding.
            show_progress: Show progress bar.
            normalize: Whether to normalize embeddings.
            
        Returns:
            Numpy array of embeddings.
        """
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=normalize,
            convert_to_numpy=True,
        )
        
        return embeddings

    def embed_query(self, query: str) -> np.ndarray:
        """Generate embedding for a single query.
        
        Args:
            query: Query text.
            
        Returns:
            1D embedding array.
        """
        return self.embed(query, normalize=True)[0]

    def embed_documents(
        self,
        documents: List[str],
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> np.ndarray:
        """Generate embeddings for multiple documents.
        
        Args:
            documents: List of document texts.
            batch_size: Batch size for encoding.
            show_progress: Show progress bar.
            
        Returns:
            2D array of embeddings.
        """
        logger.info("embedding_documents", count=len(documents))
        
        embeddings = self.embed(
            documents,
            batch_size=batch_size,
            show_progress=show_progress,
            normalize=True,
        )
        
        logger.info("documents_embedded", shape=embeddings.shape)
        
        return embeddings

    def similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray,
    ) -> float:
        """Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding.
            embedding2: Second embedding.
            
        Returns:
            Cosine similarity score.
        """
        # Ensure 1D arrays
        e1 = embedding1.flatten()
        e2 = embedding2.flatten()
        
        # For normalized vectors, dot product = cosine similarity
        return float(np.dot(e1, e2))


# Singleton instance
_embedder_instance: Optional[TextEmbedder] = None


def get_embedder() -> TextEmbedder:
    """Get or create the singleton embedder instance.
    
    Returns:
        TextEmbedder instance.
    """
    global _embedder_instance
    
    if _embedder_instance is None:
        _embedder_instance = TextEmbedder()
    
    return _embedder_instance
