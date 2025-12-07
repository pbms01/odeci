"""
Módulo de Chat RAG para ODECI.

Fornece interface de chat com retrieval-augmented generation
usando Claude Sonnet 4.5 para respostas estruturadas e didáticas.
"""

from src.chat.claude_client import ClaudeClient, ClaudeClientFactory
from src.chat.context_builder import ContextBuilder, ContextBuilderFactory
from src.chat.models import (
    ChatConfig,
    ChatMessage,
    ChatResponse,
    ChatSession,
    MessageRole,
    ResponseStyle,
    RetrievalContext,
    SourceReference,
    StreamChunk,
)
from src.chat.prompt_templates import PromptTemplateManager
from src.chat.response_formatter import ResponseFormatter, ResponseFormatterFactory
from src.chat.service import ChatService, ChatServiceFactory

__all__ = [
    # Service
    "ChatService",
    "ChatServiceFactory",
    # Client
    "ClaudeClient",
    "ClaudeClientFactory",
    # Context
    "ContextBuilder",
    "ContextBuilderFactory",
    # Formatter
    "ResponseFormatter",
    "ResponseFormatterFactory",
    # Templates
    "PromptTemplateManager",
    # Models
    "ChatConfig",
    "ChatMessage",
    "ChatResponse",
    "ChatSession",
    "MessageRole",
    "ResponseStyle",
    "RetrievalContext",
    "SourceReference",
    "StreamChunk",
]
