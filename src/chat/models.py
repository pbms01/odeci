"""
Modelos de dados para o módulo de chat.

Define estruturas para mensagens, sessões e respostas do chat RAG.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class MessageRole(str, Enum):
    """Papéis de mensagens no chat."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ResponseStyle(str, Enum):
    """Estilos de resposta disponíveis."""

    CONCISE = "concise"        # Respostas curtas e diretas
    DETAILED = "detailed"      # Respostas completas e didáticas
    TECHNICAL = "technical"    # Respostas técnicas com terminologia específica


class SourceReference(BaseModel):
    """
    Referência a uma fonte usada na resposta.

    Attributes:
        chunk_id: ID do chunk fonte.
        text: Trecho relevante do texto.
        section: Seção do documento.
        document_name: Nome do documento fonte.
        score: Score de relevância.
        page_number: Número da página (se disponível).
    """

    chunk_id: str
    text: str
    section: str | None = None
    document_name: str | None = None
    score: float = 0.0
    page_number: int | None = None

    def format_citation(self) -> str:
        """Formata a citação para exibição."""
        parts = []
        if self.document_name:
            parts.append(self.document_name)
        if self.section:
            parts.append(f"Seção: {self.section}")
        if self.page_number:
            parts.append(f"Pág. {self.page_number}")

        return " | ".join(parts) if parts else "Fonte não identificada"


class ChatMessage(BaseModel):
    """
    Mensagem individual do chat.

    Attributes:
        id: Identificador único da mensagem.
        role: Papel do remetente (user/assistant/system).
        content: Conteúdo da mensagem.
        sources: Fontes usadas na resposta (para assistant).
        metadata: Metadados adicionais.
        created_at: Timestamp de criação.
    """

    id: UUID = Field(default_factory=uuid4)
    role: MessageRole
    content: str

    # Metadados de resposta (para assistant)
    sources: list[SourceReference] = Field(default_factory=list)
    response_style: ResponseStyle | None = None
    tokens_used: int | None = None
    model_used: str | None = None

    # Contexto recuperado (para debugging)
    context_used: str | None = None

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)

    def to_api_format(self) -> dict[str, str]:
        """Converte para formato da API Anthropic."""
        return {
            "role": self.role.value if self.role != MessageRole.SYSTEM else "user",
            "content": self.content
        }


class ChatSession(BaseModel):
    """
    Sessão de chat com histórico de mensagens.

    Mantém o contexto da conversa e configurações.

    Attributes:
        id: Identificador único da sessão.
        messages: Histórico de mensagens.
        collection: Coleção de documentos para busca.
        response_style: Estilo de resposta padrão.
        max_history: Máximo de mensagens no histórico.
        created_at: Timestamp de criação.
        updated_at: Timestamp de última atualização.
    """

    id: UUID = Field(default_factory=uuid4)
    messages: list[ChatMessage] = Field(default_factory=list)

    # Configurações da sessão
    collection: str = "default"
    response_style: ResponseStyle = ResponseStyle.DETAILED
    max_history: int = 10

    # Metadados
    title: str | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    def add_message(self, message: ChatMessage) -> None:
        """
        Adiciona mensagem ao histórico.

        Args:
            message: Mensagem a adicionar.
        """
        self.messages.append(message)
        self.updated_at = datetime.utcnow()

        # Gerar título automaticamente se não tiver
        if self.title is None and message.role == MessageRole.USER:
            self.title = message.content[:50] + ("..." if len(message.content) > 50 else "")

    def get_history(self, max_messages: int | None = None) -> list[ChatMessage]:
        """
        Retorna histórico de mensagens.

        Args:
            max_messages: Número máximo de mensagens (None = usa max_history).

        Returns:
            Lista de mensagens do histórico.
        """
        limit = max_messages or self.max_history
        return self.messages[-limit:]

    def get_history_for_api(self, max_messages: int | None = None) -> list[dict[str, str]]:
        """
        Retorna histórico formatado para API.

        Args:
            max_messages: Número máximo de mensagens.

        Returns:
            Lista de dicts no formato da API.
        """
        history = self.get_history(max_messages)
        return [
            msg.to_api_format()
            for msg in history
            if msg.role != MessageRole.SYSTEM
        ]

    def clear_history(self) -> None:
        """Limpa o histórico de mensagens."""
        self.messages = []
        self.updated_at = datetime.utcnow()

    @property
    def message_count(self) -> int:
        """Número de mensagens na sessão."""
        return len(self.messages)

    @property
    def last_message(self) -> ChatMessage | None:
        """Última mensagem da sessão."""
        return self.messages[-1] if self.messages else None


class RetrievalContext(BaseModel):
    """
    Contexto recuperado para geração de resposta.

    Attributes:
        chunks: Lista de chunks recuperados.
        query: Query original do usuário.
        reformulated_query: Query reformulada (se houver).
        total_tokens: Total de tokens no contexto.
    """

    chunks: list[SourceReference] = Field(default_factory=list)
    query: str
    reformulated_query: str | None = None
    total_tokens: int = 0

    def format_for_prompt(self, max_tokens: int = 8000) -> str:
        """
        Formata contexto para inclusão no prompt.

        Args:
            max_tokens: Limite de tokens.

        Returns:
            Contexto formatado como string.
        """
        if not self.chunks:
            return "Nenhum contexto relevante encontrado nos documentos."

        sections = []
        current_tokens = 0

        for i, chunk in enumerate(self.chunks, 1):
            # Aproximação: 1 token ≈ 4 caracteres
            chunk_tokens = len(chunk.text) // 4

            if current_tokens + chunk_tokens > max_tokens:
                break

            citation = chunk.format_citation()
            section = f"[Fonte {i}] {citation}\n{chunk.text}"
            sections.append(section)
            current_tokens += chunk_tokens

        return "\n\n---\n\n".join(sections)

    @property
    def has_context(self) -> bool:
        """Verifica se há contexto disponível."""
        return len(self.chunks) > 0


class ChatResponse(BaseModel):
    """
    Resposta completa do chat.

    Attributes:
        message: Mensagem de resposta.
        context: Contexto usado na geração.
        follow_up_questions: Perguntas sugeridas.
        processing_time: Tempo de processamento em segundos.
    """

    message: ChatMessage
    context: RetrievalContext | None = None
    follow_up_questions: list[str] = Field(default_factory=list)
    processing_time: float = 0.0

    @property
    def content(self) -> str:
        """Conteúdo da resposta."""
        return self.message.content

    @property
    def sources(self) -> list[SourceReference]:
        """Fontes usadas na resposta."""
        return self.message.sources


class StreamChunk(BaseModel):
    """
    Chunk de resposta em streaming.

    Attributes:
        text: Texto do chunk.
        is_final: Indica se é o chunk final.
        accumulated_text: Texto acumulado até este ponto.
    """

    text: str
    is_final: bool = False
    accumulated_text: str = ""

    # Metadados (preenchidos no final)
    sources: list[SourceReference] | None = None
    follow_up_questions: list[str] | None = None


class ChatConfig(BaseModel):
    """
    Configuração do serviço de chat.

    Attributes:
        model: Modelo Claude a usar.
        max_tokens: Máximo de tokens na resposta.
        temperature: Temperatura de geração.
        max_context_tokens: Máximo de tokens de contexto.
        stream: Habilita streaming.
        include_sources: Incluir citações de fontes.
        include_follow_up: Incluir perguntas de follow-up.
    """

    # Modelo
    model: str = "claude-sonnet-4-5-20250514"
    max_tokens: int = 4096
    temperature: float = 0.3

    # Contexto
    max_context_tokens: int = 8000
    max_sources: int = 5
    include_parent_context: bool = True

    # Resposta
    response_style: ResponseStyle = ResponseStyle.DETAILED
    include_sources: bool = True
    include_follow_up: bool = True

    # Streaming
    stream: bool = True

    # Histórico
    max_history_messages: int = 10
