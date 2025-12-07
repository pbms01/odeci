#!/usr/bin/env python3
"""
Interface Web do ODECI Chat.

Aplicação Streamlit para chat RAG com documentos jurídicos e tecnológicos.

Uso:
    streamlit run scripts/chat_web.py

    # ou com porta específica
    streamlit run scripts/chat_web.py --server.port 8502

Deploy no Streamlit Cloud:
    1. Conecte o repositório GitHub
    2. Configure os secrets no painel do Streamlit Cloud
    3. Aponte para scripts/chat_web.py como arquivo principal
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Adicionar diretório raiz ao path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

import streamlit as st

# Versão do deploy - atualizar a cada mudança significativa
APP_VERSION = "v1.0.2-fix"
BUILD_ID = "2024-12-07T18:45"  # Timestamp do build

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
from src.config import clear_settings_cache, get_settings


# ==============================================================================
# Helpers para Secrets (Streamlit Cloud + Env Vars)
# ==============================================================================

def get_secret(key: str, default: str | None = None) -> str | None:
    """
    Obtém um secret do Streamlit Cloud ou variáveis de ambiente.

    Args:
        key: Nome da chave.
        default: Valor padrão se não encontrado.

    Returns:
        Valor do secret ou default.
    """
    # Tentar st.secrets primeiro (Streamlit Cloud)
    try:
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass

    # Fallback para variáveis de ambiente
    return os.getenv(key, default)


def load_secrets_to_env() -> None:
    """
    Carrega secrets do Streamlit Cloud para variáveis de ambiente.

    Isso permite que o restante do código use os.getenv normalmente.
    """
    secret_keys = [
        "ANTHROPIC_API_KEY",
        "VOYAGE_API_KEY",
        "COHERE_API_KEY",
        "QDRANT_URL",
        "QDRANT_API_KEY",
        "OPENAI_API_KEY",
    ]

    loaded_keys = []
    for key in secret_keys:
        value = get_secret(key)
        if value and not os.getenv(key):
            os.environ[key] = value
            loaded_keys.append(key)

    # Log para debug (visível nos logs do Streamlit Cloud)
    if loaded_keys:
        print(f"[ODECI] Secrets carregados: {loaded_keys}")
    else:
        print("[ODECI] Nenhum secret novo carregado (já existem ou não configurados)")

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
    Obtém lista de coleções disponíveis no Qdrant.

    Retorna apenas coleções que existem no vector store.
    Se não houver coleções, retorna lista vazia.

    Returns:
        Lista de nomes de coleções existentes.
    """
    try:
        return get_collections_from_store()
    except Exception:
        return []


# ==============================================================================
# Interface Principal
# ==============================================================================

def main():
    """Função principal da aplicação."""
    # Carregar secrets do Streamlit Cloud para env vars
    load_secrets_to_env()

    # IMPORTANTE: Limpar cache de settings após carregar secrets
    # para garantir que as credenciais sejam reconhecidas
    clear_settings_cache()

    # Resetar retriever_manager se já foi inicializado com credenciais erradas
    if retriever_manager.is_initialized:
        retriever_manager.reset()
        # Também resetar chat_service que depende do retriever
        if "chat_service" in st.session_state:
            st.session_state.chat_service = None

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

        # Marcador de versão para debug
        st.caption(f"🏷️ {APP_VERSION} | {BUILD_ID}")

        # API Key (busca de st.secrets, env vars ou settings)
        api_key = get_secret("ANTHROPIC_API_KEY") or settings.anthropic_api_key

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

        # Verificar credenciais
        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_key = os.getenv("QDRANT_API_KEY")
        voyage_key = os.getenv("VOYAGE_API_KEY")

        if qdrant_url and qdrant_key and voyage_key:
            st.success("✅ Credenciais configuradas")
        else:
            missing = []
            if not qdrant_url:
                missing.append("QDRANT_URL")
            if not qdrant_key:
                missing.append("QDRANT_API_KEY")
            if not voyage_key:
                missing.append("VOYAGE_API_KEY")
            st.warning(f"⚠️ Faltam: {', '.join(missing)}")

        if retriever_manager.is_initialized:
            st.success("✅ Retriever ativo")
        else:
            st.info("🔄 Retriever será inicializado na primeira pergunta")

        st.divider()

        # Coleção
        st.subheader("📁 Coleção")

        # Tentar listar coleções com diagnóstico
        try:
            collections = get_available_collections()
            if collections:
                st.caption(f"Encontradas: {len(collections)} coleção(ões)")

                # Mostrar estatísticas detalhadas
                for coll_name in collections:
                    stats = get_collection_stats(coll_name)
                    if stats.get("exists"):
                        st.caption(f"📊 {coll_name}: {stats.get('count', 0)} vetores")
        except Exception as e:
            st.error(f"Erro ao conectar: {e}")
            collections = []

        if not collections:
            if qdrant_url and qdrant_key:
                st.warning("⚠️ Nenhuma coleção encontrada no Qdrant Cloud")
                st.caption("Verifique se os documentos foram ingeridos corretamente.")
            else:
                st.warning("⚠️ Credenciais do Qdrant não configuradas")
                st.caption("Configure QDRANT_URL e QDRANT_API_KEY nos secrets.")
            collection = None
            update_state(collection=None)
        else:
            collection = st.selectbox(
                "Selecione a coleção",
                options=collections,
                index=0,
                help="Coleção de documentos para busca",
            )
            update_state(collection=collection)

            # Estatísticas da coleção
            if retriever_manager.is_initialized and collection:
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

        # Diagnóstico de Busca
        st.divider()
        with st.expander("🔧 Diagnóstico de Busca"):
            if st.button("🧪 Testar Busca Direta"):
                try:
                    from src.chat.web.retriever_factory import _create_vector_store, _create_embedder
                    test_settings = get_settings()

                    # 1. Criar vector store e embedder
                    vs = _create_vector_store(test_settings)
                    embedder = _create_embedder(test_settings)

                    # 2. Testar query
                    test_query = "O que são contratos inteligentes?"
                    st.write(f"**Query de teste:** {test_query}")

                    # 3. Gerar embedding
                    query_vector = embedder.embed_query(test_query)
                    st.write(f"**Dimensão do embedding:** {len(query_vector)}")
                    st.write(f"**Primeiros 5 valores:** {query_vector[:5]}")

                    # 4. Buscar diretamente no Qdrant
                    coll = state.collection if state.collection else "odeci_docs"
                    results = vs.search(
                        collection=coll,
                        query_vector=query_vector,
                        top_k=10,
                    )

                    st.write(f"**Resultados encontrados:** {len(results)}")

                    if results:
                        for i, r in enumerate(results[:5]):
                            st.write(f"**{i+1}.** Score: {r.score:.4f}")
                            st.caption(r.text[:150] + "...")
                    else:
                        st.error("Nenhum resultado encontrado!")

                        # Verificar info da coleção
                        if hasattr(vs, '_client'):
                            try:
                                info = vs._client.get_collection(coll)
                                st.write(f"**Info da coleção:**")
                                st.write(f"- Pontos: {info.points_count}")
                                st.write(f"- Dimensão: {info.config.params.vectors.size}")
                                st.write(f"- Distância: {info.config.params.vectors.distance}")
                            except Exception as e:
                                st.error(f"Erro ao obter info: {e}")

                except Exception as e:
                    st.error(f"Erro no teste: {e}")
                    import traceback
                    st.code(traceback.format_exc())

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

    # Verificar se há coleções disponíveis
    if not state.collection:
        render_welcome_message()
        st.warning(
            "⚠️ Nenhuma coleção de documentos disponível. "
            "É necessário ingerir documentos no Qdrant para usar o chat RAG."
        )
        st.info(
            "Para ingerir documentos, execute o pipeline de ingestão localmente:\n\n"
            "```bash\n"
            "python scripts/ingest.py --document seu_documento.docx\n"
            "```"
        )
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

                    # Debug: mostrar info de busca
                    with st.expander("🔍 Debug: Informações de Busca", expanded=True):
                        st.write(f"**Coleção:** {state.collection}")
                        st.write(f"**Query:** {user_input}")
                        st.write(f"**Retriever inicializado:** {retriever_manager.is_initialized}")

                        # Mostrar tipo do retriever
                        if retriever_manager.is_initialized:
                            retriever = retriever_manager._retriever
                            st.write(f"**Tipo do retriever:** {type(retriever).__name__}")

                            # Testar busca diretamente pelo retriever
                            try:
                                test_results = retriever.search(
                                    query=user_input,
                                    collection=state.collection,
                                    top_k=5,
                                    rerank=False,  # Sem rerank para teste rápido
                                    include_parent=False,
                                )
                                st.write(f"**Teste retriever (sem rerank):** {len(test_results)} resultados")
                                if test_results:
                                    for i, r in enumerate(test_results[:3]):
                                        st.caption(f"{i+1}. Score {r.score:.4f}: {r.text[:80]}...")
                            except Exception as e:
                                st.error(f"Erro no teste do retriever: {e}")

                        # Testar vector store diretamente
                        try:
                            from src.chat.web.retriever_factory import _create_vector_store
                            settings = get_settings()
                            vs = _create_vector_store(settings)
                            stats = vs.count(state.collection)
                            st.write(f"**Vetores na coleção:** {stats}")
                        except Exception as e:
                            st.error(f"Erro ao verificar coleção: {e}")

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

                    # Debug: mostrar resultado da busca
                    with st.expander("🔍 Debug: Resultado da Busca", expanded=True):
                        st.write(f"**Fontes encontradas:** {len(sources)}")
                        if sources:
                            for i, src in enumerate(sources[:3]):
                                if hasattr(src, 'text'):
                                    st.write(f"Fonte {i+1}: {src.text[:100]}...")
                                else:
                                    st.write(f"Fonte {i+1}: {str(src)[:100]}...")
                        else:
                            st.warning("Nenhuma fonte retornada pelo retriever!")

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
