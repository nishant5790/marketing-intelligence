"""Structured logging configuration."""

import logging
import sys
from typing import Any, Optional

import structlog

from src.config import settings


def configure_logging(
    level: Optional[str] = None,
    json_format: bool = True,
) -> None:
    """Configure structured logging for the application.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        json_format: Whether to output JSON format.
    """
    level = level or settings.log_level
    
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level.upper()),
    )
    
    # Common processors
    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]
    
    if json_format:
        # JSON output for production
        processors = shared_processors + [
            structlog.processors.JSONRenderer(),
        ]
    else:
        # Human-readable output for development
        processors = shared_processors + [
            structlog.dev.ConsoleRenderer(colors=True),
        ]
    
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a configured logger instance.
    
    Args:
        name: Logger name (typically __name__).
        
    Returns:
        Configured structlog logger.
    """
    return structlog.get_logger(name)


class RequestLogger:
    """Context manager for request-scoped logging."""
    
    def __init__(
        self,
        request_id: str,
        method: str,
        path: str,
        **extra: Any,
    ):
        """Initialize request logger.
        
        Args:
            request_id: Unique request identifier.
            method: HTTP method.
            path: Request path.
            **extra: Additional context fields.
        """
        self.request_id = request_id
        self.method = method
        self.path = path
        self.extra = extra
        self.logger = get_logger("request")
    
    def __enter__(self) -> structlog.stdlib.BoundLogger:
        """Enter request context."""
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(
            request_id=self.request_id,
            method=self.method,
            path=self.path,
            **self.extra,
        )
        
        self.logger.info("request_started")
        return self.logger
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit request context."""
        if exc_type is not None:
            self.logger.error(
                "request_failed",
                error_type=exc_type.__name__,
                error=str(exc_val),
            )
        else:
            self.logger.info("request_completed")
        
        structlog.contextvars.clear_contextvars()
        return False


# Configure logging on import if settings are available
if settings.log_level:
    # Use JSON format in production, console format in development
    configure_logging(
        level=settings.log_level,
        json_format=settings.is_production,
    )
