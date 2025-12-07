"""
Cliente para API Anthropic (Claude).

Gerencia comunicação com a API Claude com suporte a streaming.
"""

from __future__ import annotations

import logging
from collections.abc import Generator
from typing import Any

import anthropic
from tenacity import retry, stop_after_attempt, wait_exponential

from src.chat.models import (
    ChatConfig,
    ChatMessage,
    MessageRole,
    ResponseStyle,
    StreamChunk,
)

logger = logging.getLogger(__name__)


class ClaudeClient:
    """
    Cliente para API Anthropic Claude.

    Suporta geração de texto com streaming e retry automático.

    Attributes:
        client: Cliente Anthropic.
        model: Modelo a utilizar.
        config: Configurações do chat.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-4-5-20250514",
        config: ChatConfig | None = None,
    ) -> None:
        """
        Inicializa o cliente Claude.

        Args:
            api_key: Chave da API Anthropic.
            model: ID do modelo Claude.
            config: Configurações do chat.
        """
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.config = config or ChatConfig()

        logger.info(f"ClaudeClient inicializado com modelo: {model}")

    def _build_messages(
        self,
        query: str,
        context: str,
        history: list[dict[str, str]] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Constrói lista de mensagens para a API.

        Args:
            query: Pergunta do usuário.
            context: Contexto recuperado dos documentos.
            history: Histórico de mensagens anteriores.

        Returns:
            Lista de mensagens formatadas.
        """
        messages = []

        # Adicionar histórico
        if history:
            messages.extend(history)

        # Adicionar mensagem atual com contexto
        user_message = self._format_user_message(query, context)
        messages.append({"role": "user", "content": user_message})

        return messages

    def _format_user_message(self, query: str, context: str) -> str:
        """
        Formata mensagem do usuário com contexto.

        Args:
            query: Pergunta do usuário.
            context: Contexto dos documentos.

        Returns:
            Mensagem formatada.
        """
        return f"""Contexto dos documentos:
<context>
{context}
</context>

Pergunta do usuário:
{query}"""

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def generate(
        self,
        query: str,
        context: str,
        system_prompt: str,
        history: list[dict[str, str]] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        """
        Gera resposta completa (sem streaming).

        Args:
            query: Pergunta do usuário.
            context: Contexto recuperado.
            system_prompt: Prompt de sistema.
            history: Histórico de mensagens.
            max_tokens: Máximo de tokens na resposta.
            temperature: Temperatura de geração.

        Returns:
            Resposta gerada.
        """
        messages = self._build_messages(query, context, history)

        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens or self.config.max_tokens,
            temperature=temperature or self.config.temperature,
            system=system_prompt,
            messages=messages,
        )

        content = response.content[0].text if response.content else ""

        logger.info(
            f"Resposta gerada - tokens: input={response.usage.input_tokens}, "
            f"output={response.usage.output_tokens}"
        )

        return content

    def generate_stream(
        self,
        query: str,
        context: str,
        system_prompt: str,
        history: list[dict[str, str]] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> Generator[StreamChunk, None, None]:
        """
        Gera resposta com streaming.

        Args:
            query: Pergunta do usuário.
            context: Contexto recuperado.
            system_prompt: Prompt de sistema.
            history: Histórico de mensagens.
            max_tokens: Máximo de tokens na resposta.
            temperature: Temperatura de geração.

        Yields:
            Chunks de resposta conforme são gerados.
        """
        messages = self._build_messages(query, context, history)
        accumulated_text = ""

        try:
            with self.client.messages.stream(
                model=self.model,
                max_tokens=max_tokens or self.config.max_tokens,
                temperature=temperature or self.config.temperature,
                system=system_prompt,
                messages=messages,
            ) as stream:
                for text in stream.text_stream:
                    accumulated_text += text
                    yield StreamChunk(
                        text=text,
                        is_final=False,
                        accumulated_text=accumulated_text,
                    )

                # Chunk final com metadados
                yield StreamChunk(
                    text="",
                    is_final=True,
                    accumulated_text=accumulated_text,
                )

                logger.info(
                    f"Stream completo - total caracteres: {len(accumulated_text)}"
                )

        except anthropic.APIError as e:
            logger.error(f"Erro na API Anthropic: {e}")
            raise

    def count_tokens(self, text: str) -> int:
        """
        Conta tokens em um texto.

        Args:
            text: Texto para contar tokens.

        Returns:
            Número aproximado de tokens.
        """
        # Usando a API de contagem de tokens da Anthropic
        try:
            result = self.client.count_tokens(text)
            return result
        except Exception:
            # Fallback: aproximação de 4 caracteres por token
            return len(text) // 4

    def validate_api_key(self) -> bool:
        """
        Valida se a API key está funcionando.

        Returns:
            True se a key é válida.
        """
        try:
            # Fazer uma chamada mínima para validar
            self.client.messages.create(
                model=self.model,
                max_tokens=10,
                messages=[{"role": "user", "content": "test"}],
            )
            return True
        except anthropic.AuthenticationError:
            logger.error("API key inválida")
            return False
        except Exception as e:
            logger.error(f"Erro ao validar API key: {e}")
            return False


class ClaudeClientFactory:
    """Factory para criação de clientes Claude."""

    _instances: dict[str, ClaudeClient] = {}

    @classmethod
    def create(
        cls,
        api_key: str,
        model: str = "claude-sonnet-4-5-20250514",
        config: ChatConfig | None = None,
    ) -> ClaudeClient:
        """
        Cria ou retorna instância existente do cliente.

        Args:
            api_key: Chave da API.
            model: Modelo a usar.
            config: Configurações.

        Returns:
            Instância do ClaudeClient.
        """
        cache_key = f"{api_key[:8]}_{model}"

        if cache_key not in cls._instances:
            cls._instances[cache_key] = ClaudeClient(
                api_key=api_key,
                model=model,
                config=config,
            )

        return cls._instances[cache_key]

    @classmethod
    def clear_cache(cls) -> None:
        """Limpa o cache de instâncias."""
        cls._instances.clear()
