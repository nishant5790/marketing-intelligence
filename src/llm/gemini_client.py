"""Gemini LLM client for generating responses."""

from typing import Any, Optional

import structlog
from google import genai
from google.genai import types

from src.config import settings
from src.llm.prompts import SYSTEM_PROMPT, build_qa_prompt
from src.rag.retriever import get_retriever

logger = structlog.get_logger(__name__)


class GeminiClient:
    """Client for Google Gemini LLM using the new google-genai SDK."""

    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-2.0-flash"):
        """Initialize the Gemini client.
        
        Args:
            api_key: Gemini API key. Defaults to settings.gemini_api_key.
            model_name: Model name to use.
        """
        self.api_key = api_key or settings.gemini_api_key
        self.model_name = model_name
        self._client: Optional[genai.Client] = None
        self._configured = False

    def _configure(self) -> genai.Client:
        """Configure and return the Gemini client."""
        if self._client is None:
            if not self.api_key:
                raise ValueError(
                    "Gemini API key not set. Please set GEMINI_API_KEY in your .env file."
                )
            self._client = genai.Client(api_key=self.api_key)
            self._configured = True
            logger.info("gemini_client_configured", model=self.model_name)
        return self._client

    @property
    def client(self) -> genai.Client:
        """Get or create the Gemini client."""
        return self._configure()

    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Generate a response from the LLM.
        
        Args:
            prompt: The prompt to send to the model.
            temperature: Override default temperature.
            max_tokens: Override default max tokens.
            
        Returns:
            Generated text response.
            
        Raises:
            Exception: If generation fails.
        """
        logger.info("generating_response", prompt_length=len(prompt))
        
        try:
            # Build generation config
            config = types.GenerateContentConfig(
                temperature=temperature if temperature is not None else 0.7,
                max_output_tokens=max_tokens if max_tokens is not None else 1024,
                top_p=0.9,
                top_k=40,
                system_instruction=SYSTEM_PROMPT,
            )
            
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=config,
            )
            
            if response.text:
                logger.info("response_generated", response_length=len(response.text))
                return response.text
            else:
                logger.warning("empty_response")
                return "I apologize, but I couldn't generate a response. Please try rephrasing your question."
                
        except Exception as e:
            logger.error("generation_error", error=str(e))
            raise

    def answer_question(
        self,
        question: str,
        context: Optional[str] = None,
        use_rag: bool = True,
        **filter_kwargs: Any,
    ) -> dict:
        """Answer a question, optionally with RAG.
        
        Args:
            question: User's question.
            context: Optional pre-fetched context.
            use_rag: Whether to use RAG for context retrieval.
            **filter_kwargs: Filters for RAG retrieval.
            
        Returns:
            Dictionary with answer, sources, and metadata.
        """
        sources = []
        grounded = False
        
        # Get context from RAG if not provided
        if context is None and use_rag:
            retriever = get_retriever()
            
            # Check if retriever is healthy
            if retriever.health_check():
                results = retriever.search(question, **filter_kwargs)
                
                if results:
                    context = retriever.get_context_for_query(question, **filter_kwargs)
                    sources = retriever.get_sources(results)
                    grounded = True
        
        # Build prompt
        if context:
            prompt = build_qa_prompt(context, question)
        else:
            prompt = f"""User Question: {question}

Note: I don't have access to the product database right now. I'll do my best to provide a helpful response based on general knowledge.

Answer:"""
        
        # Generate response
        answer = self.generate(prompt)
        
        return {
            "answer": answer,
            "sources": sources,
            "grounded": grounded,
            "question": question,
        }

    def stream_answer(
        self,
        question: str,
        context: Optional[str] = None,
    ):
        """Stream answer generation.
        
        Args:
            question: User's question.
            context: Optional context.
            
        Yields:
            Response text chunks.
        """
        if context:
            prompt = build_qa_prompt(context, question)
        else:
            prompt = question
        
        try:
            config = types.GenerateContentConfig(
                temperature=0.7,
                max_output_tokens=1024,
                system_instruction=SYSTEM_PROMPT,
            )
            
            # Stream response
            for chunk in self.client.models.generate_content_stream(
                model=self.model_name,
                contents=prompt,
                config=config,
            ):
                if chunk.text:
                    yield chunk.text
                    
        except Exception as e:
            logger.error("stream_error", error=str(e))
            yield f"Error generating response: {str(e)}"

    def health_check(self) -> dict:
        """Check if the Gemini API is accessible.
        
        Returns:
            Health status dictionary.
        """
        try:
            client = self._configure()
            
            # List models to verify API access
            models = client.models.list()
            model_names = [m.name for m in models if "gemini" in m.name.lower()]
            
            return {
                "status": "healthy",
                "configured": True,
                "available_models": model_names[:5],  # First 5
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "configured": bool(self.api_key),
                "error": str(e),
            }


# Singleton instance
_client_instance: Optional[GeminiClient] = None


def get_gemini_client() -> GeminiClient:
    """Get or create the singleton Gemini client.
    
    Returns:
        GeminiClient instance.
    """
    global _client_instance
    
    if _client_instance is None:
        _client_instance = GeminiClient()
    
    return _client_instance
