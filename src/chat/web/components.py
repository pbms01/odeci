"""
Componentes reutilizáveis para interface Streamlit.

Fornece componentes visuais para o chat.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from src.chat.models import ResponseStyle, SourceReference
from src.chat.web.state import get_style_options


def render_header(title: str, description: str) -> None:
    """
    Renderiza cabeçalho da aplicação.

    Args:
        title: Título da aplicação.
        description: Descrição/subtítulo.
    """
    st.markdown(
        f"""
        <div style="text-align: center; padding: 1rem 0;">
            <h1 style="margin-bottom: 0.5rem;">{title}</h1>
            <p style="color: #666; font-size: 1.1rem;">{description}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_config(
    collections: list[str],
    current_collection: str,
    current_style: ResponseStyle,
    show_sources: bool,
) -> tuple[str, ResponseStyle, bool]:
    """
    Renderiza configurações na sidebar.

    Args:
        collections: Lista de coleções disponíveis.
        current_collection: Coleção atual.
        current_style: Estilo atual.
        show_sources: Se mostra fontes.

    Returns:
        Tupla (collection, style, show_sources) atualizados.
    """
    with st.sidebar:
        st.header("⚙️ Configurações")

        # Seleção de coleção
        st.subheader("📁 Coleção de Documentos")
        collection = st.selectbox(
            "Selecione a coleção",
            options=collections,
            index=collections.index(current_collection) if current_collection in collections else 0,
            help="Escolha a coleção de documentos para buscar contexto",
        )

        st.divider()

        # Estilo de resposta
        st.subheader("✍️ Estilo de Resposta")
        style_options = get_style_options()

        style_labels = [opt["label"] for opt in style_options]
        style_values = [opt["value"] for opt in style_options]

        current_index = (
            style_values.index(current_style.value)
            if current_style.value in style_values
            else 1
        )

        selected_label = st.radio(
            "Escolha o estilo",
            options=style_labels,
            index=current_index,
            help="Define o formato e profundidade das respostas",
        )

        # Encontrar estilo selecionado
        selected_index = style_labels.index(selected_label)
        selected_style = ResponseStyle(style_values[selected_index])

        # Descrição do estilo
        st.caption(style_options[selected_index]["description"])

        st.divider()

        # Opções de exibição
        st.subheader("👁️ Exibição")
        show_sources_new = st.checkbox(
            "Mostrar fontes citadas",
            value=show_sources,
            help="Exibe os trechos dos documentos usados na resposta",
        )

        st.divider()

        # Botão limpar histórico
        if st.button("🗑️ Limpar Conversa", use_container_width=True):
            return collection, selected_style, show_sources_new, True

        return collection, selected_style, show_sources_new, False


def render_message(message: dict[str, Any], show_sources: bool = True) -> None:
    """
    Renderiza uma mensagem do chat.

    Args:
        message: Dict com role, content, sources, follow_up.
        show_sources: Se deve mostrar fontes.
    """
    role = message.get("role", "user")
    content = message.get("content", "")
    sources = message.get("sources", [])
    follow_up = message.get("follow_up", [])

    # Avatar e nome baseado no role
    if role == "user":
        avatar = "👤"
        name = "Você"
    else:
        avatar = "🤖"
        name = "ODECI"

    with st.chat_message(role, avatar=avatar):
        # Conteúdo principal
        st.markdown(content)

        # Fontes (apenas para assistente)
        if role == "assistant" and show_sources and sources:
            render_sources_expander(sources)

        # Perguntas de follow-up
        if role == "assistant" and follow_up:
            render_follow_up(follow_up)


def render_sources_expander(sources: list[SourceReference | dict]) -> None:
    """
    Renderiza fontes em um expander.

    Args:
        sources: Lista de fontes.
    """
    with st.expander("📖 Fontes Consultadas", expanded=False):
        for i, source in enumerate(sources, 1):
            # Suportar tanto SourceReference quanto dict
            if isinstance(source, dict):
                text = source.get("text", "")[:300]
                doc_name = source.get("document_name", "Documento")
                section = source.get("section", "")
                score = source.get("score", 0)
            else:
                text = source.text[:300] if source.text else ""
                doc_name = source.document_name or "Documento"
                section = source.section or ""
                score = source.score

            st.markdown(f"**[Fonte {i}]** {doc_name}")
            if section:
                st.caption(f"📍 {section}")
            st.markdown(f"> {text}...")
            st.progress(min(score, 1.0), text=f"Relevância: {score:.1%}")
            st.divider()


def render_follow_up(questions: list[str]) -> None:
    """
    Renderiza perguntas de follow-up como botões.

    Args:
        questions: Lista de perguntas sugeridas.
    """
    if not questions:
        return

    st.markdown("**🔗 Perguntas Relacionadas:**")
    cols = st.columns(len(questions))

    for col, question in zip(cols, questions):
        with col:
            if st.button(
                question[:50] + "..." if len(question) > 50 else question,
                key=f"followup_{hash(question)}",
                use_container_width=True,
            ):
                st.session_state.pending_question = question
                st.rerun()


def render_chat_input() -> str | None:
    """
    Renderiza campo de input do chat.

    Returns:
        Texto inserido ou None.
    """
    # Verificar se há pergunta pendente de follow-up
    if "pending_question" in st.session_state:
        question = st.session_state.pending_question
        del st.session_state.pending_question
        return question

    return st.chat_input(
        "Digite sua pergunta sobre os documentos...",
        key="chat_input",
    )


def render_error(message: str) -> None:
    """
    Renderiza mensagem de erro.

    Args:
        message: Mensagem de erro.
    """
    st.error(f"❌ {message}")


def render_warning(message: str) -> None:
    """
    Renderiza mensagem de aviso.

    Args:
        message: Mensagem de aviso.
    """
    st.warning(f"⚠️ {message}")


def render_info(message: str) -> None:
    """
    Renderiza mensagem informativa.

    Args:
        message: Mensagem informativa.
    """
    st.info(f"ℹ️ {message}")


def render_processing_indicator() -> None:
    """Renderiza indicador de processamento."""
    with st.chat_message("assistant", avatar="🤖"):
        st.markdown("*Pensando...*")
        st.spinner()


def render_welcome_message() -> None:
    """Renderiza mensagem de boas-vindas."""
    st.markdown(
        """
        <div style="text-align: center; padding: 2rem; color: #666;">
            <h3>👋 Bem-vindo ao ODECI Chat!</h3>
            <p>
                Faça perguntas sobre os documentos indexados.<br>
                As respostas serão baseadas no conteúdo dos documentos,<br>
                com citação das fontes utilizadas.
            </p>
            <p style="font-size: 0.9rem; margin-top: 1rem;">
                <strong>Dica:</strong> Use a barra lateral para ajustar<br>
                o estilo de resposta e outras configurações.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_api_key_input() -> str | None:
    """
    Renderiza input para API key.

    Returns:
        API key inserida ou None.
    """
    st.warning("⚠️ API Key da Anthropic não configurada")

    api_key = st.text_input(
        "Insira sua API Key da Anthropic",
        type="password",
        help="Sua chave de API da Anthropic (começa com 'sk-ant-')",
    )

    if api_key:
        if api_key.startswith("sk-ant-"):
            return api_key
        else:
            st.error("API Key inválida. Deve começar com 'sk-ant-'")

    st.markdown(
        """
        <small>
        Obtenha sua API key em
        <a href="https://console.anthropic.com/" target="_blank">console.anthropic.com</a>
        </small>
        """,
        unsafe_allow_html=True,
    )

    return None
