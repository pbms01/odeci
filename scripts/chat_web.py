#!/usr/bin/env python3
"""
Interface Web do ODECI Chat.

Aplicação Streamlit para chat RAG com documentos jurídicos e tecnológicos.

Uso:
    streamlit run scripts/chat_web.py

    # ou com porta específica
    streamlit run scripts/chat_web.py --server.port 8502
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Adicionar diretório raiz ao path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

import streamlit as st

from src.chat.models import ChatConfig, ResponseStyle
from src.chat.service import ChatService
from src.chat.web import (
    add_assistant_message,
    add_user_message,
    clear_messages,
    get_state,
    init_state,
    render_api_key_input,
    render_chat_input,
    render_error,
    render_header,
    render_message,
    render_welcome_message,
    style_from_string,
    update_last_assistant_message,
    update_state,
)
from src.chat.web.retriever_factory import (
    create_retriever,
    get_available_collections as get_collections_from_store,
    get_collection_stats,
    retriever_manager,
    RetrieverInitError,
)
from src.config import get_settings

# ==============================================================================
# Configuração da Página
# ==============================================================================

st.set_page_config(
    page_title="ODECI Chat",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS customizado
st.markdown(
    """
    <style>
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .stChatMessage[data-testid="user-message"] {
        background-color: #f0f2f6;
    }
    .stChatMessage[data-testid="assistant-message"] {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
    }
    .source-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 3px solid #4CAF50;
    }
    .follow-up-btn {
        margin: 0.25rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==============================================================================
# Inicialização
# ==============================================================================

def init_chat_service(api_key: str, use_mock: bool = False) -> ChatService | None:
    """
    Inicializa o serviço de chat.

    Args:
        api_key: Chave da API Anthropic.
        use_mock: Se True, usa retriever mock (para testes).

    Returns:
        Instância do ChatService ou None se falhar.
    """
    try:
        settings = get_settings()

        # Criar configuração do chat
        config = ChatConfig(
            model=settings.chat.model,
            max_tokens=settings.chat.max_tokens,
            temperature=settings.chat.temperature,
            max_context_tokens=settings.chat.context.max_tokens,
            max_sources=settings.chat.context.max_sources,
            include_parent_context=settings.chat.context.include_parent,
            response_style=style_from_string(settings.chat.response_style),
            include_sources=settings.chat.include_sources,
            include_follow_up=settings.chat.include_follow_up,
            stream=settings.chat.stream,
            max_history_messages=settings.chat.history.max_messages,
        )

        # Criar retriever (real ou mock)
        if use_mock:
            retriever = _create_mock_retriever()
        else:
            try:
                retriever = retriever_manager.get_retriever(settings)
            except RetrieverInitError as e:
                st.warning(f"⚠️ Retriever real não disponível: {e}")
                st.info("Usando modo de demonstração com dados mock.")
                retriever = _create_mock_retriever()

        return ChatService(
            api_key=api_key,
            retriever=retriever,
            config=config,
        )
    except Exception as e:
        st.error(f"Erro ao inicializar serviço: {e}")
        return None


def _create_mock_retriever():
    """
    Cria um retriever mock para demonstração/fallback.

    Returns:
        MockRetriever para testes e demonstração.
    """
    from dataclasses import dataclass
    from typing import Any

    @dataclass
    class MockRetrievalResult:
        chunk_id: str
        text: str
        score: float
        rerank_score: float | None
        metadata: dict[str, Any]
        parent_text: str | None = None

        @property
        def final_score(self) -> float:
            return self.rerank_score if self.rerank_score else self.score

    class MockRetriever:
        """Retriever mock para demonstração."""

        def search(
            self,
            query: str,
            collection: str,
            namespaces: list[str] | None = None,
            top_k: int = 5,
            filter_metadata: dict | None = None,
            rerank: bool = True,
            include_parent: bool = True,
        ) -> list[MockRetrievalResult]:
            """Retorna resultados mock baseados na query."""
            # Resultados mock com conteúdo relevante
            return [
                MockRetrievalResult(
                    chunk_id="mock-1",
                    text="Os smart contracts são programas autoexecutáveis armazenados "
                         "em blockchain que automatizam acordos entre partes. Eles "
                         "representam uma evolução significativa na forma como contratos "
                         "podem ser implementados e executados.",
                    score=0.92,
                    rerank_score=0.95,
                    metadata={
                        "document_name": "O Direito na Era dos Contratos Inteligentes",
                        "section": "Capítulo 2 - Smart Contracts",
                        "page_number": 15,
                    },
                    parent_text="Os contratos inteligentes surgiram como uma solução "
                                "tecnológica para automatizar e garantir a execução de acordos...",
                ),
                MockRetrievalResult(
                    chunk_id="mock-2",
                    text="A validade jurídica dos contratos inteligentes depende "
                         "do cumprimento dos requisitos legais tradicionais: "
                         "capacidade das partes, objeto lícito, forma adequada e "
                         "manifestação livre de vontade.",
                    score=0.88,
                    rerank_score=0.90,
                    metadata={
                        "document_name": "O Direito na Era dos Contratos Inteligentes",
                        "section": "Capítulo 4 - Validade Jurídica",
                        "page_number": 42,
                    },
                ),
                MockRetrievalResult(
                    chunk_id="mock-3",
                    text="DAOs (Organizações Autônomas Descentralizadas) representam "
                         "um novo paradigma de governança corporativa baseado em "
                         "smart contracts e votação tokenizada, permitindo decisões "
                         "coletivas sem intermediários centralizados.",
                    score=0.85,
                    rerank_score=0.87,
                    metadata={
                        "document_name": "O Direito na Era dos Contratos Inteligentes",
                        "section": "Capítulo 6 - DAOs",
                        "page_number": 78,
                    },
                ),
            ]

    return MockRetriever()


def get_available_collections() -> list[str]:
    """
    Obtém lista de coleções disponíveis.

    Tenta buscar do vector store real, com fallback para valores padrão.

    Returns:
        Lista de nomes de coleções.
    """
    try:
        collections = get_collections_from_store()
        if collections:
            return collections
    except Exception:
        pass

    # Fallback para coleções padrão
    return ["juridico_tech", "smart_contracts", "default"]


# ==============================================================================
# Interface Principal
# ==============================================================================

def main():
    """Função principal da aplicação."""
    # Inicializar estado
    init_state()
    state = get_state()
    settings = get_settings()

    # Header
    render_header(
        title=settings.chat.web.title,
        description=settings.chat.web.description,
    )

    # Sidebar com configurações
    with st.sidebar:
        st.header("⚙️ Configurações")

        # API Key
        api_key = settings.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")

        if not api_key:
            st.warning("API Key não configurada")
            api_key = st.text_input(
                "Anthropic API Key",
                type="password",
                help="Insira sua API key da Anthropic",
            )
            if api_key:
                os.environ["ANTHROPIC_API_KEY"] = api_key
        else:
            st.success("✅ API Key configurada")

        st.divider()

        # Status do Retriever
        st.subheader("🔍 Status do Retriever")
        if retriever_manager.is_initialized:
            st.success("✅ Retriever real ativo")
        else:
            st.info("🎭 Modo demonstração")

        st.divider()

        # Coleção
        st.subheader("📁 Coleção")
        collections = get_available_collections()
        collection = st.selectbox(
            "Selecione a coleção",
            options=collections,
            index=0,
            help="Coleção de documentos para busca",
        )
        update_state(collection=collection)

        # Estatísticas da coleção
        if retriever_manager.is_initialized:
            stats = get_collection_stats(collection)
            if stats.get("exists"):
                st.caption(f"📊 {stats.get('count', 0)} documentos indexados")
            else:
                st.caption("⚠️ Coleção não encontrada")

        st.divider()

        # Estilo de resposta
        st.subheader("✍️ Estilo")
        style_option = st.radio(
            "Estilo de resposta",
            options=["🎯 Conciso", "📚 Detalhado", "⚙️ Técnico"],
            index=1,
            help="Define a profundidade das respostas",
        )

        style_map = {
            "🎯 Conciso": ResponseStyle.CONCISE,
            "📚 Detalhado": ResponseStyle.DETAILED,
            "⚙️ Técnico": ResponseStyle.TECHNICAL,
        }
        update_state(response_style=style_map[style_option])

        st.divider()

        # Opções
        st.subheader("👁️ Exibição")
        show_sources = st.checkbox("Mostrar fontes", value=True)
        update_state(show_sources=show_sources)

        st.divider()

        # Limpar conversa
        if st.button("🗑️ Limpar Conversa", use_container_width=True):
            clear_messages()
            st.rerun()

        # Info
        st.divider()
        with st.expander("ℹ️ Sobre"):
            st.markdown(
                """
                **ODECI Chat** é um assistente de IA que responde
                perguntas baseado em documentos jurídicos e
                tecnológicos indexados.

                As respostas são geradas usando:
                - **Retrieval**: Busca semântica nos documentos
                - **Claude**: Modelo de linguagem da Anthropic

                Todas as respostas incluem citação das fontes.
                """
            )

    # Verificar API key
    if not api_key:
        render_welcome_message()
        st.info("👆 Configure sua API Key na barra lateral para começar.")
        return

    # Inicializar serviço de chat
    if st.session_state.get("chat_service") is None:
        with st.spinner("Inicializando chat..."):
            service = init_chat_service(api_key)
            if service:
                st.session_state.chat_service = service
                # Criar sessão
                session = service.create_session(
                    collection=state.collection,
                    response_style=state.response_style,
                )
                update_state(session=session)

    service = st.session_state.get("chat_service")
    if not service:
        render_error("Não foi possível inicializar o serviço de chat.")
        return

    # Container principal do chat
    chat_container = st.container()

    # Renderizar mensagens existentes
    with chat_container:
        if not state.display_messages:
            render_welcome_message()
        else:
            for message in state.display_messages:
                render_message(message, show_sources=state.show_sources)

    # Input do usuário
    user_input = render_chat_input()

    if user_input:
        # Adicionar mensagem do usuário
        add_user_message(user_input)

        # Renderizar mensagem do usuário
        with chat_container:
            render_message({"role": "user", "content": user_input})

        # Processar resposta
        with chat_container:
            with st.chat_message("assistant", avatar="🤖"):
                message_placeholder = st.empty()

                try:
                    # Obter ou criar sessão
                    session = state.session
                    if session is None:
                        session = service.create_session(
                            collection=state.collection,
                            response_style=state.response_style,
                        )
                        update_state(session=session)

                    # Atualizar estilo se mudou
                    if session.response_style != state.response_style:
                        service.change_response_style(session, state.response_style)

                    # Gerar resposta com streaming
                    full_response = ""
                    sources = []
                    follow_up = []

                    # Usar streaming
                    response_generator = service.chat(
                        query=user_input,
                        session=session,
                        collection=state.collection,
                        style=state.response_style,
                        stream=True,
                    )

                    for chunk in response_generator:
                        if hasattr(chunk, 'is_final') and chunk.is_final:
                            # Chunk final com metadados
                            if chunk.sources:
                                sources = chunk.sources
                            if chunk.follow_up_questions:
                                follow_up = chunk.follow_up_questions
                        else:
                            # Chunk de texto
                            full_response = chunk.accumulated_text
                            message_placeholder.markdown(full_response + "▌")

                    # Remover cursor
                    message_placeholder.markdown(full_response)

                    # Mostrar fontes
                    if state.show_sources and sources:
                        with st.expander("📖 Fontes Consultadas", expanded=False):
                            for i, source in enumerate(sources, 1):
                                if hasattr(source, 'document_name'):
                                    doc_name = source.document_name or "Documento"
                                    section = source.section or ""
                                    text = source.text[:300] if source.text else ""
                                    score = source.score
                                else:
                                    doc_name = source.get("document_name", "Documento")
                                    section = source.get("section", "")
                                    text = source.get("text", "")[:300]
                                    score = source.get("score", 0)

                                st.markdown(f"**[{i}]** {doc_name}")
                                if section:
                                    st.caption(f"📍 {section}")
                                st.markdown(f"> {text}...")
                                st.progress(min(score, 1.0))
                                st.divider()

                    # Mostrar follow-up
                    if follow_up:
                        st.markdown("**🔗 Perguntas Relacionadas:**")
                        for q in follow_up[:3]:
                            if st.button(q, key=f"fu_{hash(q)}"):
                                st.session_state.pending_question = q
                                st.rerun()

                    # Salvar no histórico
                    add_assistant_message(
                        content=full_response,
                        sources=sources,
                        follow_up=follow_up,
                    )

                except Exception as e:
                    error_msg = f"Erro ao processar: {str(e)}"
                    message_placeholder.error(error_msg)
                    add_assistant_message(content=f"❌ {error_msg}")


# ==============================================================================
# Ponto de Entrada
# ==============================================================================

if __name__ == "__main__":
    main()
