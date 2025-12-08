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

import streamlit as st
import tempfile
import time
import json
from pathlib import Path
from uuid import UUID

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
        st.header("⚙️ Configurações")

        st.subheader("📊 Parâmetros de Chunking")
        parent_size = st.slider("Tamanho Parent (tokens)", 1000, 4000, 2000, 100)
        child_size = st.slider("Tamanho Child (tokens)", 200, 800, 400, 50)
        atomic_size = st.slider("Tamanho Atomic (tokens)", 50, 200, 100, 10)

        st.subheader("🔎 Parâmetros de Busca")
        top_k = st.slider("Resultados (top_k)", 1, 20, 5)
        use_reranking = st.checkbox("Usar Reranking", value=True)
        include_parent = st.checkbox("Incluir contexto Parent", value=True)

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

    st.info("""
    💡 **Nota:** A geração real de embeddings requer a chave da API Voyage AI.
    Nesta demonstração, mostramos como o processo funcionaria.
    """)

    # Simulação do processo
    st.subheader("📊 Processo de Embedding")

    all_chunks = chunks.get_all_chunks()

    # Mostrar modelo único
    st.markdown("**Modelo Único:** `voyage-3-large`")
    st.progress(1.0, text=f"Todos os {len(all_chunks)} chunks usam o mesmo modelo")

    # Mostrar distribuição por domínio (classificação, não modelo)
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

    # Demonstração visual
    st.subheader("🔢 Exemplo de Embedding")

    if all_chunks:
        sample_chunk = all_chunks[0]
        st.markdown(f"**Texto:** _{sample_chunk.text[:200]}..._")

        # Simular embedding
        import random
        fake_embedding = [round(random.uniform(-1, 1), 4) for _ in range(10)]

        st.markdown("**Vetor (primeiras 10 de 1024 dimensões):**")
        st.code(f"[{', '.join(map(str, fake_embedding))}, ...]")

        st.markdown("""
        **Propriedades do vetor:**
        - Dimensões: 1024
        - Tipo: float32
        - Normalizado: Sim (para distância de cosseno)
        """)


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

        # Configuração do LLM
        st.markdown("**Configuração do LLM:**")

        col1, col2 = st.columns(2)
        with col1:
            llm_provider = st.selectbox(
                "Provedor:",
                ["openai", "anthropic"],
                key="llm_provider"
            )
        with col2:
            if llm_provider == "openai":
                llm_model = st.selectbox(
                    "Modelo:",
                    ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo"],
                    key="llm_model"
                )
            else:
                llm_model = st.selectbox(
                    "Modelo:",
                    ["claude-3-5-sonnet-20241022", "claude-3-haiku-20240307"],
                    key="llm_model"
                )

        # Botão para gerar resposta
        if st.button("🤖 Gerar Resposta com LLM", type="secondary"):
            try:
                from src.generation import create_generator
                import os

                # Verificar API key
                if llm_provider == "openai":
                    api_key = os.getenv("OPENAI_API_KEY")
                    if not api_key:
                        st.error("⚠️ Configure a variável de ambiente OPENAI_API_KEY")
                        st.code("export OPENAI_API_KEY='sua-chave-aqui'")
                        st.stop()
                else:
                    api_key = os.getenv("ANTHROPIC_API_KEY")
                    if not api_key:
                        st.error("⚠️ Configure a variável de ambiente ANTHROPIC_API_KEY")
                        st.code("export ANTHROPIC_API_KEY='sua-chave-aqui'")
                        st.stop()

                with st.spinner(f"Gerando resposta com {llm_model}..."):
                    # Criar gerador
                    generator = create_generator(
                        provider=llm_provider,
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
