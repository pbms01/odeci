"""
Formatador de respostas do chat.

Formata respostas com citações, follow-up questions e estrutura didática.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from src.chat.models import (
    ChatMessage,
    ChatResponse,
    MessageRole,
    ResponseStyle,
    RetrievalContext,
    SourceReference,
)

logger = logging.getLogger(__name__)


class ResponseFormatter:
    """
    Formatador de respostas do chat.

    Adiciona estrutura, citações e formatação às respostas.
    """

    def __init__(
        self,
        include_sources: bool = True,
        include_follow_up: bool = True,
        max_citation_length: int = 200,
    ) -> None:
        """
        Inicializa o formatador.

        Args:
            include_sources: Incluir seção de fontes.
            include_follow_up: Incluir perguntas de follow-up.
            max_citation_length: Tamanho máximo de citações.
        """
        self._include_sources = include_sources
        self._include_follow_up = include_follow_up
        self._max_citation_length = max_citation_length

    def format_response(
        self,
        content: str,
        context: RetrievalContext | None = None,
        follow_up_questions: list[str] | None = None,
        style: ResponseStyle = ResponseStyle.DETAILED,
        model_used: str | None = None,
        tokens_used: int | None = None,
    ) -> ChatResponse:
        """
        Formata resposta completa.

        Args:
            content: Conteúdo da resposta.
            context: Contexto usado na geração.
            follow_up_questions: Perguntas sugeridas.
            style: Estilo de resposta usado.
            model_used: Modelo utilizado.
            tokens_used: Tokens consumidos.

        Returns:
            Resposta formatada.
        """
        # Extrair fontes do contexto
        sources = context.chunks if context else []

        # Criar mensagem
        message = ChatMessage(
            role=MessageRole.ASSISTANT,
            content=content,
            sources=sources,
            response_style=style,
            model_used=model_used,
            tokens_used=tokens_used,
            context_used=context.format_for_prompt() if context else None,
        )

        return ChatResponse(
            message=message,
            context=context,
            follow_up_questions=follow_up_questions or [],
        )

    def add_sources_section(
        self,
        content: str,
        sources: list[SourceReference],
    ) -> str:
        """
        Adiciona seção de fontes à resposta.

        Args:
            content: Conteúdo original.
            sources: Lista de fontes.

        Returns:
            Conteúdo com seção de fontes.
        """
        if not self._include_sources or not sources:
            return content

        sources_section = "\n\n---\n\n## 📖 Fontes Consultadas\n\n"

        for i, source in enumerate(sources, 1):
            citation = source.format_citation()
            preview = self._truncate_text(source.text, self._max_citation_length)
            sources_section += f"**[Fonte {i}]** {citation}\n"
            sources_section += f"> {preview}\n\n"

        return content + sources_section

    def add_follow_up_section(
        self,
        content: str,
        questions: list[str],
    ) -> str:
        """
        Adiciona seção de perguntas de follow-up.

        Args:
            content: Conteúdo original.
            questions: Lista de perguntas.

        Returns:
            Conteúdo com seção de follow-up.
        """
        if not self._include_follow_up or not questions:
            return content

        follow_up_section = "\n\n---\n\n## 🔗 Perguntas Relacionadas\n\n"

        for question in questions:
            follow_up_section += f"- {question}\n"

        return content + follow_up_section

    def format_for_display(
        self,
        response: ChatResponse,
        include_metadata: bool = False,
    ) -> str:
        """
        Formata resposta para exibição.

        Args:
            response: Resposta do chat.
            include_metadata: Incluir metadados de debug.

        Returns:
            String formatada para display.
        """
        output = response.content

        # Adicionar fontes se configurado
        if self._include_sources and response.sources:
            output = self.add_sources_section(output, response.sources)

        # Adicionar follow-up se configurado
        if self._include_follow_up and response.follow_up_questions:
            output = self.add_follow_up_section(output, response.follow_up_questions)

        # Adicionar metadados se solicitado
        if include_metadata:
            output += self._format_metadata(response)

        return output

    def _format_metadata(self, response: ChatResponse) -> str:
        """
        Formata metadados de debug.

        Args:
            response: Resposta do chat.

        Returns:
            String com metadados.
        """
        metadata = "\n\n---\n\n<details>\n<summary>📊 Metadados</summary>\n\n"

        if response.message.model_used:
            metadata += f"- **Modelo**: {response.message.model_used}\n"

        if response.message.tokens_used:
            metadata += f"- **Tokens**: {response.message.tokens_used}\n"

        if response.processing_time > 0:
            metadata += f"- **Tempo**: {response.processing_time:.2f}s\n"

        if response.context:
            metadata += f"- **Fontes recuperadas**: {len(response.context.chunks)}\n"
            metadata += f"- **Tokens contexto**: ~{response.context.total_tokens}\n"

        metadata += "\n</details>"

        return metadata

    def _truncate_text(self, text: str, max_length: int) -> str:
        """
        Trunca texto preservando palavras.

        Args:
            text: Texto original.
            max_length: Comprimento máximo.

        Returns:
            Texto truncado.
        """
        if len(text) <= max_length:
            return text

        # Truncar no último espaço antes do limite
        truncated = text[:max_length]
        last_space = truncated.rfind(" ")

        if last_space > max_length * 0.7:
            truncated = truncated[:last_space]

        return truncated.strip() + "..."

    def extract_follow_up_questions(self, text: str) -> list[str]:
        """
        Extrai perguntas de follow-up de um texto.

        Args:
            text: Texto contendo perguntas.

        Returns:
            Lista de perguntas extraídas.
        """
        questions = []

        # Padrão: linhas numeradas ou com bullet
        patterns = [
            r"^\d+\.\s*(.+\?)",  # 1. Pergunta?
            r"^[-•]\s*(.+\?)",   # - Pergunta?
            r"^(.+\?)$",         # Linha terminando em ?
        ]

        for line in text.strip().split("\n"):
            line = line.strip()
            for pattern in patterns:
                match = re.match(pattern, line)
                if match:
                    question = match.group(1).strip()
                    if question and len(question) > 10:
                        questions.append(question)
                    break

        return questions[:3]  # Máximo 3 perguntas

    def format_streaming_chunk(
        self,
        text: str,
        is_final: bool = False,
    ) -> dict[str, Any]:
        """
        Formata chunk de streaming para frontend.

        Args:
            text: Texto do chunk.
            is_final: Se é o chunk final.

        Returns:
            Dict formatado para SSE/WebSocket.
        """
        return {
            "type": "chunk" if not is_final else "final",
            "content": text,
            "done": is_final,
        }


class ResponseFormatterFactory:
    """Factory para criação de formatadores."""

    @classmethod
    def create(
        cls,
        style: ResponseStyle = ResponseStyle.DETAILED,
        include_sources: bool = True,
        include_follow_up: bool = True,
    ) -> ResponseFormatter:
        """
        Cria formatador configurado.

        Args:
            style: Estilo de resposta.
            include_sources: Incluir fontes.
            include_follow_up: Incluir follow-up.

        Returns:
            Instância do ResponseFormatter.
        """
        # Ajustar configurações baseado no estilo
        max_citation_length = {
            ResponseStyle.CONCISE: 100,
            ResponseStyle.DETAILED: 200,
            ResponseStyle.TECHNICAL: 300,
        }.get(style, 200)

        return ResponseFormatter(
            include_sources=include_sources,
            include_follow_up=include_follow_up,
            max_citation_length=max_citation_length,
        )
