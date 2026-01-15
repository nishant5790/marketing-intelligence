"""Unit tests for text embedding."""

import numpy as np
import pytest

from src.rag.embedder import TextEmbedder


class TestTextEmbedder:
    """Tests for TextEmbedder class."""

    @pytest.fixture
    def embedder(self):
        """Create embedder instance."""
        return TextEmbedder()

    def test_embed_single_text(self, embedder):
        text = "This is a test sentence."
        embedding = embedder.embed(text)
        
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (1, embedder.embedding_dim)

    def test_embed_multiple_texts(self, embedder):
        texts = ["First sentence.", "Second sentence.", "Third sentence."]
        embeddings = embedder.embed(texts)
        
        assert embeddings.shape == (3, embedder.embedding_dim)

    def test_embed_query(self, embedder):
        query = "What are the best headphones?"
        embedding = embedder.embed_query(query)
        
        assert isinstance(embedding, np.ndarray)
        assert embedding.ndim == 1
        assert len(embedding) == embedder.embedding_dim

    def test_embeddings_normalized(self, embedder):
        text = "Test sentence for normalization."
        embedding = embedder.embed(text, normalize=True)[0]
        
        # Normalized vectors should have unit length
        norm = np.linalg.norm(embedding)
        assert np.isclose(norm, 1.0, atol=1e-5)

    def test_similarity(self, embedder):
        text1 = "I love programming"
        text2 = "I enjoy coding"
        text3 = "The weather is nice today"
        
        emb1 = embedder.embed_query(text1)
        emb2 = embedder.embed_query(text2)
        emb3 = embedder.embed_query(text3)
        
        # Similar texts should have higher similarity
        sim_related = embedder.similarity(emb1, emb2)
        sim_unrelated = embedder.similarity(emb1, emb3)
        
        assert sim_related > sim_unrelated

    def test_embedding_dim(self, embedder):
        # all-MiniLM-L6-v2 has 384 dimensions
        assert embedder.embedding_dim == 384

    def test_embed_documents(self, embedder):
        documents = [
            "Product description one.",
            "Product description two.",
        ]
        
        embeddings = embedder.embed_documents(documents)
        
        assert embeddings.shape == (2, embedder.embedding_dim)

    def test_empty_text(self, embedder):
        # Empty string should still produce an embedding
        embedding = embedder.embed("")
        assert embedding.shape == (1, embedder.embedding_dim)

    def test_long_text(self, embedder):
        # Long text should be handled (truncated by the model)
        long_text = "This is a test. " * 1000
        embedding = embedder.embed(long_text)
        
        assert embedding.shape == (1, embedder.embedding_dim)
