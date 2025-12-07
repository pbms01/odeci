"""
Gerenciamento de estado para interface web Streamlit.

Centraliza o estado da sessão e configurações.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

import streamlit as st

from src.chat.models import (
    ChatMessage,
    ChatSession,
    MessageRole,
    ResponseStyle,
    SourceReference,
)


@dataclass
class WebState:
    """
    Estado da aplicação web.

    Gerencia sessão de chat, configurações e histórico visual.
    """

    # Sessão de chat
    session: ChatSession | None = None

    # Configurações atuais
    collection: str = "default"
    response_style: ResponseStyle = ResponseStyle.DETAILED
    show_sources: bool = True
    show_metadata: bool = False

    # Estado da UI
    is_processing: bool = False
    error_message: str | None = None

    # Histórico de mensagens para exibição
    display_messages: list[dict[str, Any]] = field(default_factory=list)


def init_state() -> None:
    """Inicializa estado da sessão Streamlit."""
    if "web_state" not in st.session_state:
        st.session_state.web_state = WebState()

    if "chat_service" not in st.session_state:
        st.session_state.chat_service = None

    if "available_collections" not in st.session_state:
        st.session_state.available_collections = ["default"]


def get_state() -> WebState:
    """Obtém estado atual."""
    init_state()
    return st.session_state.web_state


def update_state(**kwargs) -> None:
    """
    Atualiza campos do estado.

    Args:
        **kwargs: Campos a atualizar.
    """
    state = get_state()
    for key, value in kwargs.items():
        if hasattr(state, key):
            setattr(state, key, value)


def add_user_message(content: str) -> None:
    """
    Adiciona mensagem do usuário ao histórico visual.

    Args:
        content: Conteúdo da mensagem.
    """
    state = get_state()
    state.display_messages.append({
        "role": "user",
        "content": content,
        "sources": [],
        "follow_up": [],
    })


def add_assistant_message(
    content: str,
    sources: list[SourceReference] | None = None,
    follow_up: list[str] | None = None,
) -> None:
    """
    Adiciona mensagem do assistente ao histórico visual.

    Args:
        content: Conteúdo da resposta.
        sources: Fontes citadas.
        follow_up: Perguntas de follow-up.
    """
    state = get_state()
    state.display_messages.append({
        "role": "assistant",
        "content": content,
        "sources": sources or [],
        "follow_up": follow_up or [],
    })


def update_last_assistant_message(content: str) -> None:
    """
    Atualiza conteúdo da última mensagem do assistente.

    Usado durante streaming.

    Args:
        content: Novo conteúdo acumulado.
    """
    state = get_state()
    if state.display_messages and state.display_messages[-1]["role"] == "assistant":
        state.display_messages[-1]["content"] = content


def clear_messages() -> None:
    """Limpa histórico de mensagens."""
    state = get_state()
    state.display_messages = []
    if state.session:
        state.session.clear_history()


def get_style_options() -> list[dict[str, str]]:
    """Retorna opções de estilo para select box."""
    return [
        {
            "value": ResponseStyle.CONCISE.value,
            "label": "🎯 Conciso",
            "description": "Respostas diretas e objetivas",
        },
        {
            "value": ResponseStyle.DETAILED.value,
            "label": "📚 Detalhado",
            "description": "Respostas completas e didáticas",
        },
        {
            "value": ResponseStyle.TECHNICAL.value,
            "label": "⚙️ Técnico",
            "description": "Rigor técnico-jurídico",
        },
    ]


def style_from_string(style_str: str) -> ResponseStyle:
    """
    Converte string para ResponseStyle.

    Args:
        style_str: String do estilo.

    Returns:
        ResponseStyle correspondente.
    """
    style_map = {
        "concise": ResponseStyle.CONCISE,
        "detailed": ResponseStyle.DETAILED,
        "technical": ResponseStyle.TECHNICAL,
    }
    return style_map.get(style_str.lower(), ResponseStyle.DETAILED)
