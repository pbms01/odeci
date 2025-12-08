"""
ODECI - Interface Educativa com Streamlit

Demonstra visualmente o pipeline RAG (Retrieval-Augmented Generation):
1. Carregamento e parsing de documentos
2. Chunking hierárquico (Parent/Child/Atomic)
3. Classificação de domínio
4. Geração de embeddings multi-modelo
5. Armazenamento vetorial
6. Busca semântica com reranking
7. Geração de resposta com contexto
"""

# Versão da aplicação
__version__ = "0.2.0"

import streamlit as st
import tempfile
import time
import json
import numpy as np
from pathlib import Path
from uuid import UUID

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from sklearn.decomposition import PCA
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# Configuração da página
st.set_page_config(
    page_title="ODECI - Pipeline RAG Educativo",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS customizado
st.markdown("""
<style>
    .stProgress > div > div > div > div {
        background-color: #4CAF50;
    }
    .step-box {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        border-left: 5px solid #4CAF50;
    }
    .metric-box {
        background-color: #e8f4ea;
        border-radius: 5px;
        padding: 10px;
        text-align: center;
    }
    .chunk-parent { border-left: 4px solid #2196F3; padding-left: 10px; }
    .chunk-child { border-left: 4px solid #4CAF50; padding-left: 10px; }
    .chunk-atomic { border-left: 4px solid #FF9800; padding-left: 10px; }
    .domain-legal { background-color: #e3f2fd; }
    .domain-code { background-color: #f3e5f5; }
    .domain-tech { background-color: #e8f5e9; }
    .domain-general { background-color: #fff3e0; }
</style>
""", unsafe_allow_html=True)


def get_api_key(key_name: str) -> str | None:
    """Obtém API key do st.secrets ou variáveis de ambiente."""
    import os

    # Primeiro tenta st.secrets (Streamlit Cloud)
    try:
        if key_name in st.secrets:
            return st.secrets[key_name]
    except Exception:
        pass

    # Fallback para variáveis de ambiente (local)
    return os.getenv(key_name)


def check_api_status(api_name: str, key_name: str) -> dict:
    """Verifica o status de uma API."""
    api_key = get_api_key(key_name)

    if not api_key:
        return {"status": "missing", "message": "Chave não configurada"}

    # Verificar se a chave tem formato válido (verificação básica)
    if len(api_key) < 10:
        return {"status": "invalid", "message": "Chave muito curta"}

    # Teste de conexão real (opcional, pode ser lento)
    try:
        if api_name == "voyage" and api_key.startswith("pa-"):
            return {"status": "configured", "message": "Chave configurada"}
        elif api_name == "openai" and api_key.startswith("sk-"):
            return {"status": "configured", "message": "Chave configurada"}
        elif api_name == "anthropic" and ("sk-ant" in api_key or len(api_key) > 50):
            return {"status": "configured", "message": "Chave configurada"}
        elif api_name == "cohere":
            return {"status": "configured", "message": "Chave configurada"}
        else:
            return {"status": "configured", "message": "Chave configurada"}
    except Exception as e:
        return {"status": "error", "message": str(e)[:30]}


def render_api_diagnostics():
    """Renderiza diagnóstico de conexão com APIs."""
    st.subheader("🔌 Status das APIs")

    apis = [
        ("Voyage AI", "VOYAGE_API_KEY", "voyage", "Embeddings"),
        ("OpenAI", "OPENAI_API_KEY", "openai", "LLM"),
        ("Anthropic", "ANTHROPIC_API_KEY", "anthropic", "LLM"),
        ("Cohere", "COHERE_API_KEY", "cohere", "Reranking"),
    ]

    status_icons = {
        "configured": "🟢",
        "missing": "🔴",
        "invalid": "🟡",
        "error": "🟠",
    }

    for display_name, key_name, api_name, purpose in apis:
        status = check_api_status(api_name, key_name)
        icon = status_icons.get(status["status"], "⚪")

        with st.container():
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"{icon} **{display_name}** ({purpose})")
            with col2:
                if status["status"] == "configured":
                    st.markdown("✓")
                elif status["status"] == "missing":
                    st.markdown("✗")
                else:
                    st.markdown("?")

    # Resumo
    configured_count = sum(
        1 for _, key_name, api_name, _ in apis
        if check_api_status(api_name, key_name)["status"] == "configured"
    )

    if configured_count >= 2:
        st.success(f"{configured_count}/4 APIs configuradas")
    elif configured_count == 1:
        st.warning(f"{configured_count}/4 APIs configuradas")
    else:
        st.error("Nenhuma API configurada")


def init_session_state():
    """Inicializa o estado da sessão."""
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = None
    if "document" not in st.session_state:
        st.session_state.document = None
    if "chunks" not in st.session_state:
        st.session_state.chunks = None
    if "collection_name" not in st.session_state:
        st.session_state.collection_name = None
    if "search_history" not in st.session_state:
        st.session_state.search_history = []


def render_header():
    """Renderiza o cabeçalho."""
    st.title("🔍 ODECI - Pipeline RAG Educativo")
    st.markdown("""
    > **O**ptimized **D**ocument **E**mbedding and **C**hunking **I**ntelligence

    Esta interface demonstra **passo a passo** como funciona um sistema RAG
    (Retrieval-Augmented Generation) para busca semântica em documentos.
    """)


def render_sidebar():
    """Renderiza a barra lateral com configurações."""
    with st.sidebar:
        # Versão no topo
        st.markdown(f"""
        <div style="text-align: center; padding: 10px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    border-radius: 10px; margin-bottom: 20px;">
            <h2 style="margin: 0; color: white;">🔍 ODECI</h2>
            <p style="margin: 5px 0 0 0; color: rgba(255,255,255,0.9); font-size: 0.9em;">v{__version__}</p>
        </div>
        """, unsafe_allow_html=True)

        st.header("⚙️ Configurações")

        st.subheader("📊 Parâmetros de Chunking")
        parent_size = st.slider("Tamanho Parent (tokens)", 1000, 4000, 2000, 100)
        child_size = st.slider("Tamanho Child (tokens)", 200, 800, 400, 50)
        atomic_size = st.slider("Tamanho Atomic (tokens)", 50, 200, 100, 10)

        st.subheader("🔎 Parâmetros de Busca")
        top_k = st.slider("Resultados (top_k)", 1, 20, 5)
        use_reranking = st.checkbox("Usar Reranking", value=True)
        include_parent = st.checkbox("Incluir contexto Parent", value=True)

        st.markdown("---")

        # Diagnóstico de APIs
        render_api_diagnostics()

        st.markdown("---")

        st.subheader("📚 Sobre o Pipeline")
        with st.expander("O que é RAG?"):
            st.markdown("""
            **RAG** (Retrieval-Augmented Generation) é uma técnica que:

            1. **Recupera** informações relevantes de uma base de conhecimento
            2. **Aumenta** o prompt com esse contexto
            3. **Gera** respostas mais precisas e fundamentadas

            Isso permite que LLMs respondam sobre dados específicos
            sem precisar de fine-tuning.
            """)

        # Info de versão no rodapé
        st.markdown("---")
        st.caption(f"ODECI v{__version__} | Pipeline RAG Educativo")

        return {
            "parent_size": parent_size,
            "child_size": child_size,
            "atomic_size": atomic_size,
            "top_k": top_k,
            "use_reranking": use_reranking,
            "include_parent": include_parent,
        }


def render_pipeline_diagram():
    """Renderiza um diagrama do pipeline."""
    st.subheader("📈 Arquitetura do Pipeline")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.markdown("""
        <div style="text-align: center; padding: 20px; background: #e3f2fd; border-radius: 10px; color: #1a1a1a;">
            <h3 style="margin: 0; color: #1565c0;">📄</h3>
            <b>1. Documento</b>
            <br><small>PDF, DOCX, TXT</small>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 20px; background: #e8f5e9; border-radius: 10px; color: #1a1a1a;">
            <h3 style="margin: 0; color: #2e7d32;">✂️</h3>
            <b>2. Chunking</b>
            <br><small>Hierárquico</small>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div style="text-align: center; padding: 20px; background: #fff3e0; border-radius: 10px; color: #1a1a1a;">
            <h3 style="margin: 0; color: #e65100;">🧮</h3>
            <b>3. Embedding</b>
            <br><small>Modelo único</small>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div style="text-align: center; padding: 20px; background: #f3e5f5; border-radius: 10px; color: #1a1a1a;">
            <h3 style="margin: 0; color: #7b1fa2;">🗄️</h3>
            <b>4. Vector DB</b>
            <br><small>Qdrant/Chroma</small>
        </div>
        """, unsafe_allow_html=True)

    with col5:
        st.markdown("""
        <div style="text-align: center; padding: 20px; background: #ffebee; border-radius: 10px; color: #1a1a1a;">
            <h3 style="margin: 0; color: #c62828;">🔍</h3>
            <b>5. Retrieval</b>
            <br><small>+ Reranking</small>
        </div>
        """, unsafe_allow_html=True)


def render_step_1_upload():
    """Etapa 1: Upload e carregamento de documento."""
    st.header("📄 Etapa 1: Carregamento do Documento")

    with st.expander("ℹ️ O que acontece nesta etapa?", expanded=False):
        st.markdown("""
        **Parsing de Documentos:**
        - Extração de texto de diferentes formatos (PDF, DOCX, TXT, MD)
        - Identificação de estrutura (headings, seções, capítulos)
        - Extração de metadados (autor, data, título)
        - Detecção de elementos especiais (tabelas, código, notas de rodapé)

        **Por que isso importa?**
        A qualidade do parsing afeta diretamente a qualidade do chunking
        e, consequentemente, da busca.
        """)

    uploaded_file = st.file_uploader(
        "Faça upload de um documento",
        type=["pdf", "docx", "txt", "md"],
        help="Formatos suportados: PDF, Word, Texto, Markdown"
    )

    if uploaded_file:
        # Salvar arquivo temporário
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name

        with st.spinner("Carregando e parseando documento..."):
            try:
                from src.main import DocumentLoader

                # Simular progresso
                progress = st.progress(0)
                for i in range(100):
                    time.sleep(0.01)
                    progress.progress(i + 1)

                document = DocumentLoader.load(tmp_path)
                st.session_state.document = document

                # Exibir resultados
                st.success(f"✅ Documento carregado: **{document.name}**")

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Caracteres", f"{len(document.text):,}")
                with col2:
                    st.metric("Palavras", f"{len(document.text.split()):,}")
                with col3:
                    st.metric("Seções", len(document.sections))
                with col4:
                    st.metric("Tipo", document.metadata.file_type.upper())

                # Preview do texto
                with st.expander("👁️ Preview do Documento"):
                    st.text_area(
                        "Primeiros 2000 caracteres:",
                        document.text[:2000] + "..." if len(document.text) > 2000 else document.text,
                        height=300,
                        disabled=True
                    )

                # Estrutura de seções
                if document.sections:
                    with st.expander("📑 Estrutura de Seções"):
                        for section in document.sections:
                            st.markdown(f"{'  ' * (section.level - 1)}**{section.title}** ({section.char_count} chars)")

            except Exception as e:
                st.error(f"Erro ao carregar documento: {e}")
                import traceback
                st.code(traceback.format_exc())

        # Limpar arquivo temporário
        Path(tmp_path).unlink(missing_ok=True)

    return st.session_state.document


def render_step_2_chunking(document, config):
    """Etapa 2: Chunking hierárquico."""
    st.header("✂️ Etapa 2: Chunking Hierárquico")

    with st.expander("ℹ️ O que é Chunking Hierárquico?", expanded=False):
        st.markdown("""
        **Estratégia de 3 Níveis:**

        | Nível | Tamanho | Propósito |
        |-------|---------|-----------|
        | 🔵 **Parent** | ~2000 tokens | Contexto amplo para reranking |
        | 🟢 **Child** | ~400 tokens | Unidade principal de retrieval |
        | 🟠 **Atomic** | ~100 tokens | Definições e dados pontuais |

        **Vantagens:**
        - Melhor precisão na busca (chunks menores)
        - Contexto preservado (referência ao parent)
        - Captura de definições importantes (atomic)

        **Técnicas aplicadas:**
        - Preservação de blocos de código
        - Overlap semântico entre chunks
        - Corte em separadores naturais (parágrafos, sentenças)
        """)

    if document is None:
        st.warning("⚠️ Carregue um documento primeiro.")
        return None

    if st.button("🔪 Executar Chunking", type="primary"):
        with st.spinner("Processando chunking hierárquico..."):
            try:
                from src.chunking.hierarchical import HierarchicalChunker
                from src.chunking.domain_classifier import DomainClassifier

                # Inicializar chunker
                classifier = DomainClassifier()
                chunker = HierarchicalChunker(classifier=classifier)

                # Executar chunking com progresso
                progress = st.progress(0, text="Analisando estrutura...")
                time.sleep(0.3)
                progress.progress(25, text="Criando parent chunks...")
                time.sleep(0.3)
                progress.progress(50, text="Dividindo em child chunks...")
                time.sleep(0.3)
                progress.progress(75, text="Extraindo atomic chunks...")

                collection = chunker.chunk(document)

                progress.progress(100, text="Chunking concluído!")
                time.sleep(0.3)

                st.session_state.chunks = collection

                # Estatísticas
                st.success(f"✅ Chunking concluído: **{collection.total_chunks}** chunks gerados")

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown("""
                    <div style="background: #e3f2fd; padding: 15px; border-radius: 10px; text-align: center; color: #1a1a1a;">
                        <h2 style="color: #1565c0; margin: 0;">🔵 {}</h2>
                        <p style="margin: 0; color: #1a1a1a;">Parent Chunks</p>
                    </div>
                    """.format(len(collection.parent_chunks)), unsafe_allow_html=True)

                with col2:
                    st.markdown("""
                    <div style="background: #e8f5e9; padding: 15px; border-radius: 10px; text-align: center; color: #1a1a1a;">
                        <h2 style="color: #2e7d32; margin: 0;">🟢 {}</h2>
                        <p style="margin: 0; color: #1a1a1a;">Child Chunks</p>
                    </div>
                    """.format(len(collection.child_chunks)), unsafe_allow_html=True)

                with col3:
                    st.markdown("""
                    <div style="background: #fff3e0; padding: 15px; border-radius: 10px; text-align: center; color: #1a1a1a;">
                        <h2 style="color: #e65100; margin: 0;">🟠 {}</h2>
                        <p style="margin: 0; color: #1a1a1a;">Atomic Chunks</p>
                    </div>
                    """.format(len(collection.atomic_chunks)), unsafe_allow_html=True)

                # Distribuição por domínio
                st.subheader("🏷️ Classificação por Domínio")

                domain_colors = {
                    "legal": "#e3f2fd",
                    "code": "#f3e5f5",
                    "tech": "#e8f5e9",
                    "general": "#fff3e0"
                }

                domain_icons = {
                    "legal": "⚖️",
                    "code": "💻",
                    "tech": "🔧",
                    "general": "📄"
                }

                cols = st.columns(len(collection.chunks_by_domain))
                for i, (domain, count) in enumerate(collection.chunks_by_domain.items()):
                    with cols[i]:
                        st.markdown(f"""
                        <div style="background: {domain_colors.get(domain, '#f5f5f5')};
                                    padding: 15px; border-radius: 10px; text-align: center; color: #1a1a1a;">
                            <h3 style="margin: 0; color: #1a1a1a;">{domain_icons.get(domain, '📄')} {count}</h3>
                            <p style="margin: 0; text-transform: uppercase; color: #1a1a1a;">{domain}</p>
                        </div>
                        """, unsafe_allow_html=True)

                # Visualização de chunks
                st.subheader("👁️ Visualização de Chunks")

                tab1, tab2, tab3 = st.tabs(["🔵 Parents", "🟢 Children", "🟠 Atomics"])

                with tab1:
                    for i, chunk in enumerate(collection.parent_chunks[:5]):
                        with st.expander(f"Parent #{i+1} - {chunk.metadata.section or 'Sem seção'} ({chunk.metadata.token_count} tokens)"):
                            st.markdown(f"**Domínio:** {chunk.metadata.domain}")
                            st.markdown(f"**Filhos:** {len(chunk.children_ids)}")
                            st.text_area("Conteúdo:", chunk.text[:500] + "..." if len(chunk.text) > 500 else chunk.text, height=150, key=f"parent_{i}", disabled=True)

                with tab2:
                    for i, chunk in enumerate(collection.child_chunks[:10]):
                        with st.expander(f"Child #{i+1} - {chunk.metadata.section or 'Sem seção'} ({chunk.metadata.token_count} tokens)"):
                            st.markdown(f"**Domínio:** {chunk.metadata.domain}")
                            st.markdown(f"**Parent ID:** `{chunk.parent_id}`")
                            st.text_area("Conteúdo:", chunk.text, height=150, key=f"child_{i}", disabled=True)

                with tab3:
                    if collection.atomic_chunks:
                        for i, chunk in enumerate(collection.atomic_chunks[:10]):
                            with st.expander(f"Atomic #{i+1} ({chunk.metadata.token_count} tokens)"):
                                st.markdown(f"**Tipo:** {chunk.metadata.content_type or 'N/A'}")
                                st.code(chunk.text)
                    else:
                        st.info("Nenhum chunk atômico extraído deste documento.")

            except Exception as e:
                st.error(f"Erro no chunking: {e}")
                import traceback
                st.code(traceback.format_exc())

    return st.session_state.chunks


def render_step_3_embedding(chunks):
    """Etapa 3: Geração de embeddings."""
    st.header("🧮 Etapa 3: Geração de Embeddings")

    with st.expander("ℹ️ O que são Embeddings?", expanded=False):
        st.markdown("""
        **Embeddings** são representações vetoriais de texto em um espaço de alta dimensão.

        **Como funcionam:**
        ```
        "O contrato foi assinado" → [0.123, -0.456, 0.789, ..., 0.321]
                                     (vetor de 1024 dimensões)
        ```

        **Propriedades importantes:**
        - Textos semanticamente similares ficam próximos no espaço vetorial
        - Permitem busca por similaridade usando distância de cosseno
        - Capturam nuances de significado além de palavras-chave

        **Modelo utilizado: `voyage-3-large`**

        | Característica | Valor |
        |----------------|-------|
        | Modelo | voyage-3-large |
        | Dimensões | 1024 |
        | Tipo | Generalista de alta qualidade |
        | Vantagem | Um único espaço vetorial para todos os domínios |
        """)

    if chunks is None:
        st.warning("⚠️ Execute o chunking primeiro.")
        return

    # Verificar se API key está disponível
    voyage_api_key = get_api_key("VOYAGE_API_KEY")
    has_api_key = voyage_api_key is not None and len(voyage_api_key) > 10

    if has_api_key:
        st.success("✅ API Voyage AI configurada - embeddings reais disponíveis")
    else:
        st.warning("""
        ⚠️ **API Voyage AI não configurada** - usando embeddings simulados para demonstração.
        Configure `VOYAGE_API_KEY` para gerar embeddings reais.
        """)

    # Processo de Embedding
    st.subheader("📊 Processo de Embedding")

    all_chunks = chunks.get_all_chunks()

    # Mostrar modelo único
    st.markdown("**Modelo:** `voyage-3-large` (1024 dimensões)")

    # Mostrar distribuição por domínio
    st.markdown("**Classificação por Domínio (metadados):**")

    domain_counts = {}
    for chunk in all_chunks:
        domain = chunk.metadata.domain
        domain_counts[domain] = domain_counts.get(domain, 0) + 1

    for domain, count in sorted(domain_counts.items(), key=lambda x: x[1], reverse=True):
        col1, col2 = st.columns([3, 1])
        with col1:
            st.progress(count / len(all_chunks), text=f"{domain}")
        with col2:
            st.write(f"{count} chunks")

    # Botão para gerar embeddings
    st.markdown("---")

    if "embeddings_generated" not in st.session_state:
        st.session_state.embeddings_generated = False
        st.session_state.embeddings_data = None

    if st.button("🧮 Gerar Embeddings", type="primary"):
        if has_api_key:
            # Gerar embeddings reais com Voyage AI
            with st.spinner("Gerando embeddings com Voyage AI..."):
                try:
                    import voyageai

                    client = voyageai.Client(api_key=voyage_api_key)

                    # Preparar textos
                    texts = [chunk.text for chunk in all_chunks]

                    # Gerar embeddings em batches
                    progress = st.progress(0, text="Iniciando...")
                    embeddings = []
                    batch_size = 8  # Voyage AI recomenda batches pequenos

                    for i in range(0, len(texts), batch_size):
                        batch = texts[i:i + batch_size]
                        result = client.embed(
                            batch,
                            model="voyage-3-large",
                            input_type="document"
                        )
                        embeddings.extend(result.embeddings)

                        progress_pct = min((i + batch_size) / len(texts), 1.0)
                        progress.progress(progress_pct, text=f"Processando {min(i + batch_size, len(texts))}/{len(texts)} chunks...")

                    progress.progress(1.0, text="Embeddings gerados!")

                    # Armazenar no session state
                    st.session_state.embeddings_data = {
                        "embeddings": embeddings,
                        "chunks": all_chunks,
                        "model": "voyage-3-large",
                        "dimensions": len(embeddings[0]) if embeddings else 1024,
                        "is_real": True
                    }
                    st.session_state.embeddings_generated = True

                    st.success(f"✅ {len(embeddings)} embeddings gerados com sucesso!")

                    # Mostrar métricas
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Embeddings", len(embeddings))
                    with col2:
                        st.metric("Dimensões", len(embeddings[0]))
                    with col3:
                        st.metric("Modelo", "voyage-3-large")

                except Exception as e:
                    st.error(f"Erro ao gerar embeddings: {e}")
                    import traceback
                    with st.expander("Ver detalhes do erro"):
                        st.code(traceback.format_exc())
        else:
            # Gerar embeddings simulados
            with st.spinner("Gerando embeddings simulados..."):
                time.sleep(1)

                # Criar embeddings simulados com clusters baseados em domínio
                np.random.seed(42)

                domain_centroids = {
                    "legal": np.random.randn(1024) * 0.5,
                    "code": np.random.randn(1024) * 0.5 + 2,
                    "tech": np.random.randn(1024) * 0.5 + 4,
                    "general": np.random.randn(1024) * 0.5 + 6,
                }

                embeddings = []
                for chunk in all_chunks:
                    domain = chunk.metadata.domain
                    centroid = domain_centroids.get(domain, domain_centroids["general"])
                    noise = np.random.randn(1024) * 0.3
                    embedding = (centroid + noise).tolist()
                    embeddings.append(embedding)

                st.session_state.embeddings_data = {
                    "embeddings": embeddings,
                    "chunks": all_chunks,
                    "model": "simulado",
                    "dimensions": 1024,
                    "is_real": False
                }
                st.session_state.embeddings_generated = True

                st.info(f"💡 {len(embeddings)} embeddings simulados gerados para demonstração")

    # Mostrar exemplo de embedding se já foram gerados
    if st.session_state.embeddings_generated and st.session_state.embeddings_data:
        st.subheader("🔢 Exemplo de Embedding")

        data = st.session_state.embeddings_data
        sample_idx = 0
        sample_chunk = data["chunks"][sample_idx]
        sample_embedding = data["embeddings"][sample_idx]

        st.markdown(f"**Texto:** _{sample_chunk.text[:200]}..._")
        st.markdown(f"**Tipo:** {'Real (Voyage AI)' if data['is_real'] else 'Simulado'}")

        st.markdown(f"**Vetor (primeiras 10 de {data['dimensions']} dimensões):**")
        embedding_preview = [round(v, 4) for v in sample_embedding[:10]]
        st.code(f"[{', '.join(map(str, embedding_preview))}, ...]")

        st.markdown("""
        **Propriedades do vetor:**
        - Dimensões: {}
        - Tipo: float32
        - Normalizado: Sim (para distância de cosseno)
        """.format(data['dimensions']))

    # Renderizar visualização do espaço de embeddings
    render_embedding_visualization(chunks)


def render_embedding_visualization(chunks):
    """Renderiza visualização interativa do espaço de embeddings."""
    if chunks is None:
        return

    # Verificar se embeddings foram gerados
    if not st.session_state.get("embeddings_generated") or not st.session_state.get("embeddings_data"):
        st.info("💡 Clique em **Gerar Embeddings** acima para visualizar o espaço vetorial.")
        return

    st.markdown("---")
    st.subheader("🌐 Visualização do Espaço de Embeddings")

    if not PLOTLY_AVAILABLE:
        st.warning("""
        ⚠️ Para visualizar o espaço de embeddings, instale as dependências:
        ```bash
        pip install plotly scikit-learn
        ```
        """)
        return

    # Obter dados dos embeddings
    data = st.session_state.embeddings_data
    is_real = data.get("is_real", False)

    if is_real:
        st.success("🎯 Visualizando embeddings **reais** gerados pelo Voyage AI")
    else:
        st.info("💡 Visualizando embeddings **simulados** para demonstração")

    with st.expander("ℹ️ O que é esta visualização?", expanded=False):
        st.markdown("""
        **Redução de Dimensionalidade com PCA:**

        Os embeddings originais têm 1024 dimensões, impossíveis de visualizar diretamente.
        Usamos **PCA** (Principal Component Analysis) para reduzir a 2D ou 3D, preservando
        a maior quantidade possível de variância (informação).

        **O que observar:**
        - **Clusters**: Chunks similares ficam próximos no espaço
        - **Separação por domínio**: Cores diferentes mostram domínios semânticos
        - **Outliers**: Pontos isolados podem indicar conteúdo único ou erros

        **Limitações:**
        - A redução de 1024 para 2/3D inevitavelmente perde informação
        - A visualização é uma aproximação do espaço real
        """)

    all_chunks = data["chunks"]
    embeddings = data["embeddings"]

    if len(all_chunks) < 3:
        st.warning("⚠️ Mínimo de 3 chunks necessário para visualização.")
        return

    # Preparar metadados para visualização
    metadata_list = []
    for chunk in all_chunks:
        metadata_list.append({
            "domain": chunk.metadata.domain,
            "level": chunk.level.value,
            "tokens": chunk.metadata.token_count,
            "section": chunk.metadata.section or "Sem seção",
            "text_preview": chunk.text[:100] + "..." if len(chunk.text) > 100 else chunk.text,
        })

    embeddings_array = np.array(embeddings)

    # Configurações de visualização
    col1, col2, col3 = st.columns(3)

    with col1:
        n_dimensions = st.radio(
            "Dimensões:",
            [2, 3],
            index=1,
            horizontal=True,
            help="2D é mais fácil de interpretar, 3D mostra mais estrutura"
        )

    with col2:
        color_by = st.selectbox(
            "Colorir por:",
            ["domain", "level", "tokens"],
            format_func=lambda x: {"domain": "Domínio", "level": "Nível", "tokens": "Tokens"}[x]
        )

    with col3:
        point_size = st.slider("Tamanho dos pontos:", 3, 15, 8)

    # Aplicar PCA
    n_components = min(n_dimensions, len(embeddings_array) - 1, 3)
    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(embeddings_array)

    # Variância explicada
    variance_explained = pca.explained_variance_ratio_

    var_cols = st.columns(n_components + 1)
    with var_cols[0]:
        st.metric("Total Variância", f"{sum(variance_explained)*100:.1f}%")
    for i in range(n_components):
        with var_cols[i + 1]:
            st.metric(f"PC{i+1}", f"{variance_explained[i]*100:.1f}%")

    # Preparar dados para plotly
    import pandas as pd
    df = pd.DataFrame(metadata_list)
    df["PC1"] = reduced[:, 0]
    df["PC2"] = reduced[:, 1]
    if n_components >= 3:
        df["PC3"] = reduced[:, 2]

    # Definir cores
    color_map_domain = {
        "legal": "#2196F3",
        "code": "#9C27B0",
        "tech": "#4CAF50",
        "general": "#FF9800"
    }

    color_map_level = {
        "parent": "#2196F3",
        "child": "#4CAF50",
        "atomic": "#FF9800"
    }

    # Criar gráfico
    if n_dimensions == 3 and n_components >= 3:
        fig = px.scatter_3d(
            df,
            x="PC1",
            y="PC2",
            z="PC3",
            color=color_by,
            color_discrete_map=color_map_domain if color_by == "domain" else (color_map_level if color_by == "level" else None),
            hover_data=["section", "domain", "level", "tokens", "text_preview"],
            title="Espaço de Embeddings (PCA 3D)",
            labels={
                "PC1": f"PC1 ({variance_explained[0]*100:.1f}%)",
                "PC2": f"PC2 ({variance_explained[1]*100:.1f}%)",
                "PC3": f"PC3 ({variance_explained[2]*100:.1f}%)",
                "domain": "Domínio",
                "level": "Nível",
                "tokens": "Tokens",
            }
        )

        fig.update_traces(marker=dict(size=point_size))
        fig.update_layout(
            height=600,
            scene=dict(
                xaxis_title=f"PC1 ({variance_explained[0]*100:.1f}%)",
                yaxis_title=f"PC2 ({variance_explained[1]*100:.1f}%)",
                zaxis_title=f"PC3 ({variance_explained[2]*100:.1f}%)",
            )
        )
    else:
        fig = px.scatter(
            df,
            x="PC1",
            y="PC2",
            color=color_by,
            color_discrete_map=color_map_domain if color_by == "domain" else (color_map_level if color_by == "level" else None),
            hover_data=["section", "domain", "level", "tokens", "text_preview"],
            title="Espaço de Embeddings (PCA 2D)",
            labels={
                "PC1": f"PC1 ({variance_explained[0]*100:.1f}%)",
                "PC2": f"PC2 ({variance_explained[1]*100:.1f}%)",
                "domain": "Domínio",
                "level": "Nível",
                "tokens": "Tokens",
            }
        )

        fig.update_traces(marker=dict(size=point_size))
        fig.update_layout(height=500)

    st.plotly_chart(fig, use_container_width=True)

    # Estatísticas dos clusters
    st.markdown("**📊 Estatísticas por Cluster:**")

    stats_cols = st.columns(len(df["domain"].unique()))
    for i, domain in enumerate(df["domain"].unique()):
        domain_df = df[df["domain"] == domain]
        with stats_cols[i]:
            st.markdown(f"""
            <div style="background: {color_map_domain.get(domain, '#gray')}22;
                        padding: 10px; border-radius: 8px;
                        border-left: 4px solid {color_map_domain.get(domain, '#gray')};">
                <h4 style="margin: 0; color: {color_map_domain.get(domain, '#333')};">{domain.upper()}</h4>
                <p style="margin: 5px 0; color: #1a1a1a;">Chunks: {len(domain_df)}</p>
                <p style="margin: 5px 0; color: #1a1a1a;">Tokens médio: {domain_df['tokens'].mean():.0f}</p>
            </div>
            """, unsafe_allow_html=True)


def render_attention_path(query: str, results: list, chunks):
    """
    Renderiza visualização 3D do caminho de atenção entre query e chunks recuperados.

    Mostra:
    - Todos os chunks no espaço vetorial (PCA 3D)
    - Chunks recuperados destacados
    - Linhas conectando o centro da query aos chunks selecionados
    - Estatísticas de recuperação
    """
    if not PLOTLY_AVAILABLE:
        st.warning("Plotly não disponível para visualização de atenção.")
        return

    # Verificar se temos embeddings
    if not st.session_state.get("embeddings_data"):
        st.info("💡 Gere os embeddings na aba 'Embedding' para visualizar o caminho de atenção.")
        return

    st.markdown("---")
    st.subheader("🔗 Visualização do Caminho de Atenção")

    with st.expander("ℹ️ O que é esta visualização?", expanded=False):
        st.markdown("""
        **Caminho de Atenção (Attention Path):**

        Esta visualização mostra como o sistema RAG "presta atenção" aos diferentes
        chunks do documento para responder à sua pergunta.

        **Elementos visuais:**
        - **Pontos coloridos:** Todos os chunks no espaço vetorial
        - **Pontos destacados (maiores):** Chunks selecionados como contexto
        - **Linhas:** Conexões entre a query e os chunks recuperados
        - **Centro (★):** Posição média da query no espaço

        **O que observar:**
        - Chunks próximos tendem a ser semanticamente similares
        - As linhas mostram quais regiões do espaço foram "consultadas"
        - A distribuição dos chunks selecionados indica a cobertura temática
        """)

    data = st.session_state.embeddings_data
    all_chunks = data["chunks"]
    embeddings = data["embeddings"]

    if len(all_chunks) < 3:
        st.warning("Poucos chunks para visualização.")
        return

    # IDs dos chunks recuperados
    retrieved_ids = set()
    for r in results:
        chunk = r.get("chunk")
        if chunk:
            retrieved_ids.add(id(chunk))

    # Aplicar PCA para 3D
    embeddings_array = np.array(embeddings)
    n_components = min(3, len(embeddings_array) - 1)
    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(embeddings_array)

    # Preparar dados
    import pandas as pd

    metadata_list = []
    is_retrieved = []
    scores = []

    for i, chunk in enumerate(all_chunks):
        # Verificar se este chunk foi recuperado
        chunk_retrieved = id(chunk) in retrieved_ids
        is_retrieved.append(chunk_retrieved)

        # Encontrar score se recuperado
        score = 0.0
        for r in results:
            if r.get("chunk") and id(r["chunk"]) == id(chunk):
                score = r.get("rerank_score") or r.get("score", 0)
                break
        scores.append(score)

        metadata_list.append({
            "domain": chunk.metadata.domain,
            "level": chunk.level.value,
            "tokens": chunk.metadata.token_count,
            "section": chunk.metadata.section or "Sem seção",
            "text_preview": chunk.text[:80] + "..." if len(chunk.text) > 80 else chunk.text,
            "retrieved": "Recuperado" if chunk_retrieved else "Não usado",
            "score": score,
        })

    df = pd.DataFrame(metadata_list)
    df["PC1"] = reduced[:, 0]
    df["PC2"] = reduced[:, 1]
    df["PC3"] = reduced[:, 2] if n_components >= 3 else 0
    df["is_retrieved"] = is_retrieved
    df["size"] = [12 if r else 5 for r in is_retrieved]

    # Calcular centro da query (média dos chunks recuperados)
    retrieved_indices = [i for i, r in enumerate(is_retrieved) if r]
    if retrieved_indices:
        query_center = reduced[retrieved_indices].mean(axis=0)
    else:
        query_center = reduced.mean(axis=0)

    # Criar traces
    traces = []

    # 1. Todos os chunks (não recuperados) - cinza
    df_not_retrieved = df[~df["is_retrieved"]]
    if len(df_not_retrieved) > 0:
        traces.append(go.Scatter3d(
            x=df_not_retrieved["PC1"],
            y=df_not_retrieved["PC2"],
            z=df_not_retrieved["PC3"],
            mode="markers",
            marker=dict(
                size=5,
                color="#e5e7eb",
                opacity=0.5,
            ),
            text=df_not_retrieved["text_preview"],
            hovertemplate="<b>%{text}</b><br>Domínio: %{customdata[0]}<br>Nível: %{customdata[1]}<extra></extra>",
            customdata=df_not_retrieved[["domain", "level"]].values,
            name="Corpus",
            showlegend=True,
        ))

    # 2. Chunks recuperados - coloridos por domínio
    df_retrieved = df[df["is_retrieved"]]
    domain_colors = {
        "legal": "#3b82f6",
        "code": "#8b5cf6",
        "tech": "#10b981",
        "general": "#f59e0b"
    }

    if len(df_retrieved) > 0:
        colors = [domain_colors.get(d, "#6b7280") for d in df_retrieved["domain"]]
        traces.append(go.Scatter3d(
            x=df_retrieved["PC1"],
            y=df_retrieved["PC2"],
            z=df_retrieved["PC3"],
            mode="markers",
            marker=dict(
                size=12,
                color=colors,
                opacity=0.9,
                line=dict(width=2, color="white"),
            ),
            text=df_retrieved.apply(
                lambda r: f"Score: {r['score']:.3f}<br>{r['text_preview']}", axis=1
            ),
            hovertemplate="<b>%{text}</b><extra></extra>",
            name="Recuperados",
            showlegend=True,
        ))

    # 3. Centro da query
    traces.append(go.Scatter3d(
        x=[query_center[0]],
        y=[query_center[1]],
        z=[query_center[2] if len(query_center) > 2 else 0],
        mode="markers+text",
        marker=dict(
            size=15,
            color="#dc2626",
            symbol="diamond",
            line=dict(width=2, color="white"),
        ),
        text=["Query"],
        textposition="top center",
        name="Query",
        showlegend=True,
    ))

    # 4. Linhas de atenção (do centro para os chunks recuperados)
    for i, (idx, row) in enumerate(df_retrieved.iterrows()):
        opacity = 0.8 - (i * 0.1)  # Diminui opacidade por ranking
        traces.append(go.Scatter3d(
            x=[query_center[0], row["PC1"]],
            y=[query_center[1], row["PC2"]],
            z=[query_center[2] if len(query_center) > 2 else 0, row["PC3"]],
            mode="lines",
            line=dict(
                color=domain_colors.get(row["domain"], "#6b7280"),
                width=4,
            ),
            opacity=max(0.3, opacity),
            hoverinfo="skip",
            showlegend=False,
        ))

    # Layout
    fig = go.Figure(data=traces)
    fig.update_layout(
        scene=dict(
            xaxis_title="PC1",
            yaxis_title="PC2",
            zaxis_title="PC3",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2)),
        ),
        height=550,
        margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255,255,255,0.8)",
        ),
        title=dict(
            text=f"Caminho de Atenção: \"{query[:50]}...\"" if len(query) > 50 else f"Caminho de Atenção: \"{query}\"",
            font=dict(size=14),
        ),
    )

    st.plotly_chart(fig, use_container_width=True)

    # Estatísticas
    st.markdown("**📊 Estatísticas de Atenção:**")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Chunks Usados", len(df_retrieved))
    with col2:
        st.metric("Total Corpus", len(df))
    with col3:
        coverage = len(df_retrieved) / len(df) * 100 if len(df) > 0 else 0
        st.metric("Cobertura", f"{coverage:.1f}%")
    with col4:
        avg_score = df_retrieved["score"].mean() if len(df_retrieved) > 0 else 0
        st.metric("Score Médio", f"{avg_score:.3f}")

    # Distribuição por domínio dos recuperados
    if len(df_retrieved) > 0:
        st.markdown("**Distribuição dos chunks recuperados:**")
        domain_dist = df_retrieved["domain"].value_counts()
        cols = st.columns(len(domain_dist))
        for i, (domain, count) in enumerate(domain_dist.items()):
            with cols[i]:
                color = domain_colors.get(domain, "#6b7280")
                st.markdown(f"""
                <div style="background: {color}22; padding: 8px; border-radius: 6px;
                            border-left: 3px solid {color}; text-align: center;">
                    <strong style="color: {color};">{domain.upper()}</strong><br>
                    <span style="font-size: 1.2em; color: #1a1a1a;">{count}</span>
                </div>
                """, unsafe_allow_html=True)


def render_step_4_storage(chunks):
    """Etapa 4: Armazenamento vetorial."""
    st.header("🗄️ Etapa 4: Armazenamento Vetorial")

    with st.expander("ℹ️ Como funciona o Vector Store?", expanded=False):
        st.markdown("""
        **Vector Databases** são otimizados para:
        - Armazenar vetores de alta dimensão
        - Busca eficiente por similaridade (ANN - Approximate Nearest Neighbors)
        - Filtros por metadados

        **Backends suportados:**

        | Backend | Tipo | Uso Recomendado |
        |---------|------|-----------------|
        | **Qdrant** | Local/Cloud | Produção, escalabilidade |
        | **ChromaDB** | Local | Desenvolvimento, prototipagem |

        **Estrutura de um registro:**
        ```json
        {
            "id": "uuid-do-chunk",
            "vector": [0.123, -0.456, ...],  // 1024 dims
            "payload": {
                "text": "conteúdo do chunk",
                "level": "child",
                "domain": "legal",
                "parent_id": "uuid-do-parent",
                "section": "Capítulo 1",
                ...
            }
        }
        ```
        """)

    if chunks is None:
        st.warning("⚠️ Execute o chunking primeiro.")
        return

    # Configuração de coleção
    collection_name = st.text_input(
        "Nome da coleção:",
        value=f"odeci_{st.session_state.document.name.replace('.', '_').lower()}" if st.session_state.document else "odeci_demo",
        help="Nome único para identificar esta coleção no vector store"
    )

    col1, col2 = st.columns(2)
    with col1:
        backend = st.selectbox("Backend:", ["qdrant", "chroma"])
    with col2:
        st.metric("Chunks a armazenar", chunks.total_chunks)

    st.session_state.collection_name = collection_name

    # Visualização da estrutura
    st.subheader("📐 Estrutura de Armazenamento")

    sample_payload = {
        "id": "550e8400-e29b-41d4-a716-446655440000",
        "text": chunks.child_chunks[0].text[:100] + "..." if chunks.child_chunks else "exemplo",
        "level": "child",
        "domain": chunks.child_chunks[0].metadata.domain if chunks.child_chunks else "general",
        "parent_id": str(chunks.child_chunks[0].parent_id) if chunks.child_chunks and chunks.child_chunks[0].parent_id else None,
        "section": chunks.child_chunks[0].metadata.section if chunks.child_chunks else None,
        "document_name": st.session_state.document.name if st.session_state.document else "documento.pdf",
        "token_count": chunks.child_chunks[0].metadata.token_count if chunks.child_chunks else 0,
    }

    st.json(sample_payload)


def render_step_5_retrieval(chunks, config):
    """Etapa 5: Busca e retrieval."""
    st.header("🔍 Etapa 5: Busca Semântica e Retrieval")

    with st.expander("ℹ️ Como funciona o Retrieval?", expanded=False):
        st.markdown("""
        **Pipeline de Retrieval:**

        ```
        Query do usuário
              ↓
        1. Embedding da query (mesmo modelo)
              ↓
        2. Busca por similaridade (cosine distance)
              ↓
        3. Recuperação de top-K candidatos
              ↓
        4. Reranking com Cross-Encoder (opcional)
              ↓
        5. Expansão com contexto do Parent (opcional)
              ↓
        Resultados ordenados por relevância
        ```

        **Métricas de Similaridade:**
        - **Cosine Similarity:** Mede o ângulo entre vetores (0 a 1)
        - **Dot Product:** Produto escalar, considera magnitude
        - **Euclidean:** Distância geométrica no espaço

        **Reranking:**
        O reranker (Cohere) usa um modelo cross-encoder que analisa
        query + documento juntos, sendo mais preciso que embeddings
        separados, mas mais lento.
        """)

    if chunks is None:
        st.warning("⚠️ Execute o chunking primeiro.")
        return

    # Interface de busca
    query = st.text_input(
        "🔎 Digite sua pergunta:",
        placeholder="Ex: Quais são as cláusulas de rescisão do contrato?",
        help="A busca será feita por similaridade semântica, não por palavras-chave"
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        top_k = st.number_input("Resultados (top_k):", 1, 20, config["top_k"])
    with col2:
        use_rerank = st.checkbox("Reranking", config["use_reranking"])
    with col3:
        include_parent = st.checkbox("Contexto Parent", config["include_parent"])

    if query and st.button("🚀 Buscar", type="primary"):
        render_search_process(query, chunks, top_k, use_rerank, include_parent)


def render_search_process(query, chunks, top_k, use_rerank, include_parent):
    """Renderiza o processo de busca passo a passo."""

    st.markdown("---")
    st.subheader("📋 Processo de Retrieval Detalhado")

    # Etapa 1: Embedding da query
    with st.container():
        st.markdown("### 1️⃣ Embedding da Query")

        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown(f"**Query:** _{query}_")

            # Simular embedding
            with st.spinner("Gerando embedding da query..."):
                time.sleep(0.5)

            import random
            query_embedding = [round(random.uniform(-1, 1), 4) for _ in range(8)]
            st.code(f"Query Vector: [{', '.join(map(str, query_embedding))}, ...] (1024 dims)")

        with col2:
            st.info("💡 A query usa o mesmo modelo de embedding para garantir compatibilidade no espaço vetorial.")

    # Etapa 2: Busca por similaridade
    with st.container():
        st.markdown("### 2️⃣ Busca por Similaridade")

        with st.spinner("Calculando similaridades..."):
            time.sleep(0.5)

        # Simular resultados
        all_chunks = chunks.child_chunks[:20]  # Usar child chunks para busca

        import random
        results = []
        for chunk in all_chunks:
            score = round(random.uniform(0.5, 0.95), 4)
            results.append({
                "chunk": chunk,
                "score": score,
                "rerank_score": None
            })

        # Ordenar por score
        results.sort(key=lambda x: x["score"], reverse=True)
        results = results[:top_k * 2]  # Pegar mais para reranking

        st.markdown(f"**Encontrados:** {len(results)} candidatos iniciais")

        # Visualização de scores
        st.markdown("**Distribuição de Scores (Cosine Similarity):**")

        score_data = {f"Chunk {i+1}": r["score"] for i, r in enumerate(results[:10])}
        st.bar_chart(score_data)

    # Etapa 3: Reranking (opcional)
    if use_rerank:
        with st.container():
            st.markdown("### 3️⃣ Reranking com Cross-Encoder")

            with st.spinner("Aplicando reranking..."):
                time.sleep(0.7)

            st.markdown("""
            O **Cross-Encoder** processa query + documento juntos:
            ```
            Input: [CLS] query [SEP] documento [SEP]
            Output: relevance_score (0 a 1)
            ```
            """)

            # Simular reranking
            for r in results:
                r["rerank_score"] = round(random.uniform(0.3, 0.98), 4)

            results.sort(key=lambda x: x["rerank_score"], reverse=True)

            # Mostrar mudança de ranking
            st.markdown("**Comparação de Rankings:**")

            comparison_data = []
            for i, r in enumerate(results[:5]):
                comparison_data.append({
                    "Posição Final": i + 1,
                    "Score Original": r["score"],
                    "Score Rerank": r["rerank_score"],
                    "Chunk Preview": r["chunk"].text[:50] + "..."
                })

            st.dataframe(comparison_data, use_container_width=True)

    # Etapa 4: Resultados finais
    with st.container():
        st.markdown("### 4️⃣ Resultados Finais")

        final_results = results[:top_k]

        for i, r in enumerate(final_results):
            chunk = r["chunk"]
            score = r["rerank_score"] if use_rerank else r["score"]

            # Determinar cor baseada no score
            if score >= 0.8:
                score_color = "🟢"
            elif score >= 0.6:
                score_color = "🟡"
            else:
                score_color = "🔴"

            with st.expander(f"{score_color} **Resultado #{i+1}** - Score: {score:.4f} | Seção: {chunk.metadata.section or 'N/A'}"):
                col1, col2 = st.columns([3, 1])

                with col1:
                    st.markdown("**Conteúdo:**")
                    st.markdown(f"_{chunk.text}_")

                with col2:
                    st.markdown("**Metadados:**")
                    st.markdown(f"- **Domínio:** {chunk.metadata.domain}")
                    st.markdown(f"- **Tokens:** {chunk.metadata.token_count}")
                    st.markdown(f"- **Nível:** {chunk.level.value}")

                # Contexto do parent
                if include_parent and chunk.parent_id:
                    st.markdown("---")
                    st.markdown("**📎 Contexto do Parent:**")

                    # Encontrar parent
                    parent = next((p for p in chunks.parent_chunks if p.id == chunk.parent_id), None)
                    if parent:
                        st.text_area(
                            "Contexto expandido:",
                            parent.text[:500] + "..." if len(parent.text) > 500 else parent.text,
                            height=100,
                            disabled=True,
                            key=f"parent_context_{i}"
                        )

    # Etapa 5: Geração de resposta com LLM
    with st.container():
        st.markdown("### 5️⃣ Geração de Resposta (RAG)")

        st.markdown("""
        Com os chunks recuperados, um LLM gera uma resposta fundamentada:
        """)

        # Montar contexto
        context_texts = [r['chunk'].text for r in final_results[:5]]
        context_preview = "\n\n".join([f"[{i+1}] {text[:200]}..." for i, text in enumerate(context_texts[:3])])

        prompt_template = f"""
**System Prompt:**
Você é um assistente especializado. Use APENAS o contexto fornecido para responder.
Cite as fontes usando [1], [2], etc.

**Contexto:**
{context_preview}

**Pergunta:** {query}

**Resposta:**
"""

        with st.expander("📝 Ver Prompt Completo"):
            st.code(prompt_template, language="markdown")

        # Configuração do LLM (apenas Anthropic)
        st.markdown("**Configuração do LLM (Anthropic Claude):**")

        llm_model = st.selectbox(
            "Modelo:",
            [
                "claude-opus-4-5-20251101",
                "claude-sonnet-4-5-20250929",
                "claude-3-5-sonnet-20241022",
            ],
            format_func=lambda x: {
                "claude-opus-4-5-20251101": "Claude Opus 4.5 (mais capaz)",
                "claude-sonnet-4-5-20250929": "Claude Sonnet 4.5 (equilibrado)",
                "claude-3-5-sonnet-20241022": "Claude 3.5 Sonnet (rápido)",
            }.get(x, x),
            key="llm_model"
        )

        # Botão para gerar resposta
        if st.button("🤖 Gerar Resposta com LLM", type="secondary"):
            try:
                from src.generation import create_generator

                # Verificar API key Anthropic
                api_key = get_api_key("ANTHROPIC_API_KEY")
                if not api_key:
                    st.error("⚠️ Configure ANTHROPIC_API_KEY nos Secrets ou variáveis de ambiente")
                    st.code("# Local: export ANTHROPIC_API_KEY='sua-chave'\n# Streamlit Cloud: adicione em Settings > Secrets")
                    st.stop()

                with st.spinner(f"Gerando resposta com {llm_model}..."):
                    # Criar gerador Anthropic
                    generator = create_generator(
                        provider="anthropic",
                        model=llm_model,
                        api_key=api_key,
                        temperature=0.1,
                        max_tokens=1024
                    )

                    # Preparar metadados das fontes
                    sources_metadata = [
                        {
                            "section": r['chunk'].metadata.section or "N/A",
                            "domain": r['chunk'].metadata.domain,
                            "score": r['rerank_score'] if use_rerank else r['score']
                        }
                        for r in final_results[:5]
                    ]

                    # Gerar resposta
                    result = generator.generate(
                        query=query,
                        context=context_texts,
                        sources_metadata=sources_metadata
                    )

                    # Exibir resposta
                    st.markdown("---")
                    st.markdown("**🤖 Resposta Gerada:**")
                    st.markdown(result.answer)

                    # Métricas
                    st.markdown("---")
                    st.markdown("**📊 Métricas de Geração:**")

                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Modelo", result.model)
                    with col2:
                        st.metric("Tokens Prompt", f"{result.prompt_tokens:,}")
                    with col3:
                        st.metric("Tokens Resposta", f"{result.completion_tokens:,}")
                    with col4:
                        st.metric("Custo Est.", f"${result.cost_estimate:.4f}")

                    # Fontes utilizadas
                    if result.sources:
                        st.markdown("**📚 Fontes Citadas:**")
                        for src in result.sources:
                            st.markdown(f"- [{src['index']}] Seção: {src['section']} | Domínio: {src['domain']} | Score: {src['score']:.4f}")

                    # Visualização do caminho de atenção
                    render_attention_path(query, final_results, chunks)

            except ImportError as e:
                st.error(f"Erro de importação: {e}")
                st.info("Certifique-se de que os pacotes 'openai' ou 'anthropic' estão instalados.")
            except Exception as e:
                st.error(f"Erro na geração: {e}")
                import traceback
                with st.expander("Ver detalhes do erro"):
                    st.code(traceback.format_exc())
        else:
            st.info("""
            💡 **Clique no botão acima para gerar uma resposta real com LLM.**

            O contexto recuperado será enviado junto com a pergunta para gerar uma resposta
            fundamentada nos documentos, reduzindo alucinações e permitindo citação de fontes.
            """)

    # Salvar no histórico
    st.session_state.search_history.append({
        "query": query,
        "results": len(final_results),
        "top_score": final_results[0]["rerank_score"] if use_rerank else final_results[0]["score"] if final_results else 0
    })


def render_history():
    """Renderiza o histórico de buscas."""
    if st.session_state.search_history:
        st.sidebar.markdown("---")
        st.sidebar.subheader("📜 Histórico de Buscas")

        for i, search in enumerate(reversed(st.session_state.search_history[-5:])):
            st.sidebar.markdown(f"""
            **{i+1}.** _{search['query'][:30]}..._
            Resultados: {search['results']} | Score: {search['top_score']:.2f}
            """)


def main():
    """Função principal."""
    init_session_state()
    render_header()
    config = render_sidebar()

    # Abas principais
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📄 1. Upload",
        "✂️ 2. Chunking",
        "🧮 3. Embedding",
        "🗄️ 4. Storage",
        "🔍 5. Retrieval"
    ])

    # Diagrama do pipeline
    render_pipeline_diagram()

    st.markdown("---")

    with tab1:
        document = render_step_1_upload()

    with tab2:
        chunks = render_step_2_chunking(document, config)

    with tab3:
        render_step_3_embedding(chunks)

    with tab4:
        render_step_4_storage(chunks)

    with tab5:
        render_step_5_retrieval(chunks, config)

    # Histórico na sidebar
    render_history()

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>ODECI - Optimized Document Embedding and Chunking Intelligence</p>
        <p><small>Pipeline RAG educativo para documentos jurídico-tecnológicos</small></p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
