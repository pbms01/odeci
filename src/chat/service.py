"""
Serviço principal de chat RAG.

Orquestra o fluxo completo: retrieval -> contexto -> geração -> formatação.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Generator
from typing import TYPE_CHECKING, Any
from uuid import UUID

from src.chat.claude_client import ClaudeClient, ClaudeClientFactory
from src.chat.context_builder import ContextBuilder
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

if TYPE_CHECKING:
    from src.retrieval.retriever import HybridRetriever

logger = logging.getLogger(__name__)


class ChatService:
    """
    Serviço principal de chat com RAG.

    Orquestra o fluxo completo de:
    1. Receber pergunta do usuário
    2. Recuperar contexto relevante
    3. Gerar resposta via Claude
    4. Formatar e retornar resposta

    Attributes:
        claude_client: Cliente para API Claude.
        context_builder: Construtor de contexto.
        formatter: Formatador de respostas.
        config: Configurações do chat.
    """

    def __init__(
        self,
        api_key: str,
        retriever: HybridRetriever,
        config: ChatConfig | None = None,
    ) -> None:
        """
        Inicializa o serviço de chat.

        Args:
            api_key: Chave da API Anthropic.
            retriever: Instância do retriever.
            config: Configurações do chat.
        """
        self._config = config or ChatConfig()

        # Inicializar componentes
        self._claude = ClaudeClientFactory.create(
            api_key=api_key,
            model=self._config.model,
            config=self._config,
        )
        self._context_builder = ContextBuilder(
            retriever=retriever,
            config=self._config,
        )
        self._formatter = ResponseFormatterFactory.create(
            style=self._config.response_style,
            include_sources=self._config.include_sources,
            include_follow_up=self._config.include_follow_up,
        )

        # Cache de sessões
        self._sessions: dict[UUID, ChatSession] = {}

        logger.info("ChatService inicializado")

    def create_session(
        self,
        collection: str = "default",
        response_style: ResponseStyle = ResponseStyle.DETAILED,
    ) -> ChatSession:
        """
        Cria nova sessão de chat.

        Args:
            collection: Coleção de documentos.
            response_style: Estilo de resposta padrão.

        Returns:
            Nova sessão de chat.
        """
        session = ChatSession(
            collection=collection,
            response_style=response_style,
            max_history=self._config.max_history_messages,
        )
        self._sessions[session.id] = session

        logger.info(f"Sessão criada: {session.id}")
        return session

    def get_session(self, session_id: UUID) -> ChatSession | None:
        """
        Recupera sessão existente.

        Args:
            session_id: ID da sessão.

        Returns:
            Sessão ou None se não encontrada.
        """
        return self._sessions.get(session_id)

    def chat(
        self,
        query: str,
        session: ChatSession | None = None,
        collection: str | None = None,
        style: ResponseStyle | None = None,
        stream: bool | None = None,
    ) -> ChatResponse | Generator[StreamChunk, None, ChatResponse]:
        """
        Processa uma mensagem de chat.

        Args:
            query: Pergunta do usuário.
            session: Sessão de chat (cria nova se None).
            collection: Coleção para buscar (usa da sessão se não especificado).
            style: Estilo de resposta (usa da sessão se não especificado).
            stream: Usar streaming (usa config se não especificado).

        Returns:
            Resposta formatada ou generator de chunks (se streaming).
        """
        start_time = time.time()

        # Usar ou criar sessão
        if session is None:
            session = self.create_session(
                collection=collection or "default",
                response_style=style or self._config.response_style,
            )

        # Determinar parâmetros
        target_collection = collection or session.collection
        target_style = style or session.response_style
        use_stream = stream if stream is not None else self._config.stream

        logger.info(
            f"Processando query: '{query[:50]}...' | "
            f"collection={target_collection} | style={target_style.value}"
        )

        # Adicionar mensagem do usuário à sessão
        user_message = ChatMessage(role=MessageRole.USER, content=query)
        session.add_message(user_message)

        # 1. Construir contexto
        context = self._context_builder.build(
            query=query,
            collection=target_collection,
            max_sources=self._config.max_sources,
            max_tokens=self._config.max_context_tokens,
        )

        # 2. Verificar se há contexto suficiente
        if not context.has_context:
            return self._handle_no_context(session, context, start_time)

        # 3. Obter prompts
        system_prompt = PromptTemplateManager.get_system_prompt(target_style)
        context_text = context.format_for_prompt()

        # 4. Obter histórico
        history = session.get_history_for_api(self._config.max_history_messages - 1)

        # 5. Gerar resposta
        if use_stream:
            return self._chat_stream(
                query=query,
                context=context,
                context_text=context_text,
                system_prompt=system_prompt,
                history=history,
                session=session,
                style=target_style,
                start_time=start_time,
            )
        else:
            return self._chat_complete(
                query=query,
                context=context,
                context_text=context_text,
                system_prompt=system_prompt,
                history=history,
                session=session,
                style=target_style,
                start_time=start_time,
            )

    def _chat_complete(
        self,
        query: str,
        context: RetrievalContext,
        context_text: str,
        system_prompt: str,
        history: list[dict[str, str]],
        session: ChatSession,
        style: ResponseStyle,
        start_time: float,
    ) -> ChatResponse:
        """
        Gera resposta completa (sem streaming).

        Args:
            query: Pergunta.
            context: Contexto recuperado.
            context_text: Contexto formatado.
            system_prompt: Prompt de sistema.
            history: Histórico de mensagens.
            session: Sessão de chat.
            style: Estilo de resposta.
            start_time: Timestamp de início.

        Returns:
            Resposta formatada.
        """
        # Gerar resposta
        response_text = self._claude.generate(
            query=query,
            context=context_text,
            system_prompt=system_prompt,
            history=history,
            max_tokens=self._config.max_tokens,
            temperature=self._config.temperature,
        )

        # Gerar follow-up questions se configurado
        follow_up = []
        if self._config.include_follow_up:
            follow_up = self._generate_follow_up(query, response_text)

        # Formatar resposta
        processing_time = time.time() - start_time
        response = self._formatter.format_response(
            content=response_text,
            context=context,
            follow_up_questions=follow_up,
            style=style,
            model_used=self._config.model,
        )
        response.processing_time = processing_time

        # Adicionar à sessão
        session.add_message(response.message)

        logger.info(f"Resposta gerada em {processing_time:.2f}s")
        return response

    def _chat_stream(
        self,
        query: str,
        context: RetrievalContext,
        context_text: str,
        system_prompt: str,
        history: list[dict[str, str]],
        session: ChatSession,
        style: ResponseStyle,
        start_time: float,
    ) -> Generator[StreamChunk, None, ChatResponse]:
        """
        Gera resposta com streaming.

        Args:
            query: Pergunta.
            context: Contexto recuperado.
            context_text: Contexto formatado.
            system_prompt: Prompt de sistema.
            history: Histórico de mensagens.
            session: Sessão de chat.
            style: Estilo de resposta.
            start_time: Timestamp de início.

        Yields:
            Chunks de resposta.

        Returns:
            Resposta final formatada.
        """
        accumulated_text = ""

        # Stream da resposta
        for chunk in self._claude.generate_stream(
            query=query,
            context=context_text,
            system_prompt=system_prompt,
            history=history,
            max_tokens=self._config.max_tokens,
            temperature=self._config.temperature,
        ):
            accumulated_text = chunk.accumulated_text

            if not chunk.is_final:
                yield chunk
            else:
                # Chunk final - processar resposta completa
                follow_up = []
                if self._config.include_follow_up:
                    follow_up = self._generate_follow_up(query, accumulated_text)

                # Criar chunk final com metadados
                final_chunk = StreamChunk(
                    text="",
                    is_final=True,
                    accumulated_text=accumulated_text,
                    sources=context.chunks,
                    follow_up_questions=follow_up,
                )
                yield final_chunk

        # Formatar resposta final
        processing_time = time.time() - start_time
        response = self._formatter.format_response(
            content=accumulated_text,
            context=context,
            follow_up_questions=follow_up if self._config.include_follow_up else [],
            style=style,
            model_used=self._config.model,
        )
        response.processing_time = processing_time

        # Adicionar à sessão
        session.add_message(response.message)

        logger.info(f"Stream completo em {processing_time:.2f}s")
        return response

    def _handle_no_context(
        self,
        session: ChatSession,
        context: RetrievalContext,
        start_time: float,
    ) -> ChatResponse:
        """
        Trata caso de contexto insuficiente.

        Args:
            session: Sessão de chat.
            context: Contexto vazio.
            start_time: Timestamp de início.

        Returns:
            Resposta de erro formatada.
        """
        no_context_text = PromptTemplateManager.get_no_context_response()

        response = self._formatter.format_response(
            content=no_context_text,
            context=context,
            style=ResponseStyle.DETAILED,
        )
        response.processing_time = time.time() - start_time

        session.add_message(response.message)

        logger.warning("Resposta gerada sem contexto suficiente")
        return response

    def _generate_follow_up(self, query: str, response: str) -> list[str]:
        """
        Gera perguntas de follow-up.

        Args:
            query: Pergunta original.
            response: Resposta gerada.

        Returns:
            Lista de perguntas sugeridas.
        """
        try:
            prompt = PromptTemplateManager.get_follow_up_prompt(query, response)

            result = self._claude.generate(
                query=prompt,
                context="",
                system_prompt="Você é um assistente que sugere perguntas de follow-up.",
                max_tokens=200,
                temperature=0.7,
            )

            return self._formatter.extract_follow_up_questions(result)
        except Exception as e:
            logger.warning(f"Erro ao gerar follow-up: {e}")
            return []

    def change_response_style(
        self,
        session: ChatSession,
        style: ResponseStyle,
    ) -> None:
        """
        Altera estilo de resposta da sessão.

        Args:
            session: Sessão de chat.
            style: Novo estilo.
        """
        session.response_style = style
        logger.info(f"Estilo alterado para: {style.value}")

    def clear_session_history(self, session: ChatSession) -> None:
        """
        Limpa histórico da sessão.

        Args:
            session: Sessão de chat.
        """
        session.clear_history()
        logger.info(f"Histórico limpo: sessão {session.id}")

    def get_available_styles(self) -> list[dict[str, str]]:
        """
        Retorna estilos de resposta disponíveis.

        Returns:
            Lista de estilos com descrições.
        """
        return [
            {
                "id": ResponseStyle.CONCISE.value,
                "name": "Conciso",
                "description": "Respostas curtas e diretas",
            },
            {
                "id": ResponseStyle.DETAILED.value,
                "name": "Detalhado",
                "description": "Respostas completas e didáticas",
            },
            {
                "id": ResponseStyle.TECHNICAL.value,
                "name": "Técnico",
                "description": "Respostas com rigor técnico-jurídico",
            },
        ]


class ChatServiceFactory:
    """Factory para criação do ChatService."""

    _instance: ChatService | None = None

    @classmethod
    def create(
        cls,
        api_key: str,
        retriever: HybridRetriever,
        config: ChatConfig | None = None,
    ) -> ChatService:
        """
        Cria instância do ChatService.

        Args:
            api_key: Chave da API.
            retriever: Instância do retriever.
            config: Configurações.

        Returns:
            Instância do ChatService.
        """
        return ChatService(
            api_key=api_key,
            retriever=retriever,
            config=config,
        )

    @classmethod
    def get_singleton(
        cls,
        api_key: str,
        retriever: HybridRetriever,
        config: ChatConfig | None = None,
    ) -> ChatService:
        """
        Obtém instância singleton do ChatService.

        Args:
            api_key: Chave da API.
            retriever: Instância do retriever.
            config: Configurações.

        Returns:
            Instância singleton do ChatService.
        """
        if cls._instance is None:
            cls._instance = cls.create(api_key, retriever, config)
        return cls._instance

    @classmethod
    def clear_singleton(cls) -> None:
        """Limpa instância singleton."""
        cls._instance = None
