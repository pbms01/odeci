"""
ODECI - Pipeline Principal

Integra todos os componentes para processamento completo:
1. Carregamento de documentos
2. Chunking hierárquico
3. Classificação de domínio
4. Embedding multi-modelo
5. Armazenamento vetorial
6. Retrieval híbrido com reranking
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, TYPE_CHECKING
from uuid import UUID

from tqdm import tqdm

if TYPE_CHECKING:
    from src.config import Settings

from src.config import Settings, get_settings, ensure_directories
from src.models.document import Document, DocumentMetadata, DocumentSection
from src.models.chunk import Chunk, ChunkCollection, ChunkLevel
from src.chunking.hierarchical import HierarchicalChunker
from src.chunking.domain_classifier import DomainClassifier
from src.embedding.unified_embedder import UnifiedEmbedder
from src.embedding.voyage_embedder import VoyageEmbedder
from src.storage.vector_store import QdrantVectorStore, create_vector_store
from src.retrieval.retriever import HybridRetriever, RetrievalResult
from src.retrieval.reranker import CohereReranker, create_reranker
from src.utils.logging_config import setup_logging, get_logger
from src.utils.text_processing import clean_text, calculate_text_statistics

logger = get_logger(__name__)


class DocumentLoader:
    """
    Carregador de documentos.
    
    Suporta .docx, .pdf, .txt, .md
    """
    
    @staticmethod
    def load(file_path: str | Path) -> Document:
        """
        Carrega documento de arquivo.
        
        Args:
            file_path: Caminho do arquivo.
            
        Returns:
            Documento carregado.
            
        Raises:
            ValueError: Se formato não suportado.
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {path}")
        
        suffix = path.suffix.lower()
        
        if suffix == ".docx":
            return DocumentLoader._load_docx(path)
        elif suffix == ".pdf":
            return DocumentLoader._load_pdf(path)
        elif suffix in [".txt", ".md"]:
            return DocumentLoader._load_text(path)
        else:
            raise ValueError(f"Formato não suportado: {suffix}")
    
    @staticmethod
    def _load_docx(path: Path) -> Document:
        """Carrega documento Word."""
        try:
            from docx import Document as DocxDocument
        except ImportError:
            raise ImportError(
                "Pacote 'python-docx' não instalado. "
                "Execute: pip install python-docx"
            )
        
        doc = DocxDocument(str(path))
        
        # Extrair texto e estrutura
        text_parts = []
        sections = []
        current_section = None
        current_content = []
        
        for para in doc.paragraphs:
            text = para.text.strip()
            if not text:
                continue
            
            # Detectar headings
            if para.style.name.startswith("Heading"):
                # Salvar seção anterior
                if current_section:
                    current_section_obj = DocumentSection(
                        level=current_section["level"],
                        title=current_section["title"],
                        content="\n\n".join(current_content),
                        start_char=current_section["start"],
                        end_char=len("\n\n".join(text_parts)),
                    )
                    sections.append(current_section_obj)
                
                # Iniciar nova seção
                level = int(para.style.name.replace("Heading ", "")) if para.style.name != "Heading" else 1
                current_section = {
                    "level": level,
                    "title": text,
                    "start": len("\n\n".join(text_parts)),
                }
                current_content = []
            else:
                current_content.append(text)
            
            text_parts.append(text)
        
        # Última seção
        if current_section and current_content:
            sections.append(DocumentSection(
                level=current_section["level"],
                title=current_section["title"],
                content="\n\n".join(current_content),
                start_char=current_section["start"],
                end_char=len("\n\n".join(text_parts)),
            ))
        
        full_text = "\n\n".join(text_parts)
        
        # Metadados
        core_props = doc.core_properties
        metadata = DocumentMetadata(
            file_name=path.name,
            file_path=str(path.absolute()),
            file_size_bytes=path.stat().st_size,
            file_type="docx",
            title=core_props.title or path.stem,
            author=core_props.author,
            subject=core_props.subject,
            created_date=core_props.created,
            modified_date=core_props.modified,
            char_count=len(full_text),
            word_count=len(full_text.split()),
        )
        
        return Document(
            text=full_text,
            sections=sections,
            metadata=metadata,
        )
    
    @staticmethod
    def _load_pdf(path: Path) -> Document:
        """Carrega documento PDF."""
        try:
            from pypdf import PdfReader
        except ImportError:
            raise ImportError(
                "Pacote 'pypdf' não instalado. "
                "Execute: pip install pypdf"
            )
        
        reader = PdfReader(str(path))
        
        text_parts = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                text_parts.append(text)
        
        full_text = "\n\n".join(text_parts)
        
        metadata = DocumentMetadata(
            file_name=path.name,
            file_path=str(path.absolute()),
            file_size_bytes=path.stat().st_size,
            file_type="pdf",
            title=path.stem,
            page_count=len(reader.pages),
            char_count=len(full_text),
            word_count=len(full_text.split()),
        )
        
        return Document(
            text=full_text,
            metadata=metadata,
        )
    
    @staticmethod
    def _load_text(path: Path) -> Document:
        """Carrega arquivo de texto."""
        text = path.read_text(encoding="utf-8")
        
        metadata = DocumentMetadata(
            file_name=path.name,
            file_path=str(path.absolute()),
            file_size_bytes=path.stat().st_size,
            file_type=path.suffix.lstrip("."),
            title=path.stem,
            char_count=len(text),
            word_count=len(text.split()),
        )
        
        return Document(
            text=text,
            metadata=metadata,
        )


class ODECIPipeline:
    """
    Pipeline principal do ODECI.
    
    Orquestra todo o fluxo de processamento:
    - Carregamento de documentos
    - Chunking hierárquico
    - Embedding multi-modelo
    - Armazenamento vetorial
    - Retrieval com reranking
    
    Attributes:
        settings: Configurações do pipeline.
        chunker: Chunker hierárquico.
        embedder: Embedder híbrido multi-modelo.
        vector_store: Store de vetores.
        retriever: Retriever híbrido.
    """
    
    def __init__(
        self,
        settings: Settings | None = None,
        config_path: str | None = None,
    ) -> None:
        """
        Inicializa o pipeline.
        
        Args:
            settings: Configurações. Se None, carrega do arquivo.
            config_path: Caminho para arquivo de configuração.
        """
        # Carregar configurações
        self.settings = settings or get_settings(config_path)
        
        # Garantir diretórios
        ensure_directories()
        
        # Configurar logging
        setup_logging(
            level=self.settings.log_level,
            log_file=self.settings.logging.file,
        )
        
        logger.info("Inicializando ODECI Pipeline...")
        
        # Inicializar componentes
        self._init_components()
        
        logger.info("Pipeline inicializado com sucesso")
    
    def _init_components(self) -> None:
        """Inicializa componentes do pipeline."""
        # Classificador de domínio
        self._classifier = DomainClassifier(
            config=self.settings.domain_classifier
        )
        
        # Chunker
        self._chunker = HierarchicalChunker(
            config=self.settings.chunking,
            classifier=self._classifier,
        )
        
        # Embedder (modelo único para todos os domínios)
        if self.settings.is_voyage_configured():
            self._embedder = UnifiedEmbedder(
                api_key=self.settings.voyage_api_key,
                model="voyage-3-large",  # Modelo único generalista
                dimensions=self.settings.embedding.dimensions,
            )
            logger.info("Usando Voyage AI (voyage-3-large) para embeddings")
        else:
            logger.warning(
                "Voyage AI não configurado. "
                "Configure VOYAGE_API_KEY para usar embeddings."
            )
            self._embedder = None
        
        # Vector Store
        self._vector_store = create_vector_store(
            backend=self.settings.vector_store.backend,
            path=self.settings.vector_store.qdrant.path,
        )
        logger.info(f"Vector store: {self.settings.vector_store.backend}")
        
        # Reranker
        if self.settings.is_cohere_configured():
            self._reranker = CohereReranker(
                api_key=self.settings.cohere_api_key,
                model=self.settings.retrieval.reranking.model,
            )
            logger.info("Usando Cohere para reranking")
        else:
            logger.warning(
                "Cohere não configurado. "
                "Reranking desabilitado."
            )
            self._reranker = None
        
        # Retriever
        if self._embedder:
            self._retriever = HybridRetriever(
                vector_store=self._vector_store,
                embedder=self._embedder,
                reranker=self._reranker,
                config=self.settings.retrieval,
            )
        else:
            self._retriever = None
    
    def load_document(self, file_path: str | Path) -> Document:
        """
        Carrega documento de arquivo.
        
        Args:
            file_path: Caminho do arquivo.
            
        Returns:
            Documento carregado.
        """
        logger.info(f"Carregando documento: {file_path}")
        document = DocumentLoader.load(file_path)
        
        stats = calculate_text_statistics(document.text)
        logger.info(
            f"Documento carregado: {document.name} "
            f"({stats['word_count']} palavras, {stats['token_count']} tokens)"
        )
        
        return document
    
    def chunk_document(self, document: Document) -> ChunkCollection:
        """
        Realiza chunking do documento.
        
        Args:
            document: Documento a processar.
            
        Returns:
            Coleção de chunks.
        """
        logger.info(f"Iniciando chunking de: {document.name}")
        collection = self._chunker.chunk(document)
        
        logger.info(
            f"Chunking concluído: {collection.total_chunks} chunks "
            f"(P:{len(collection.parent_chunks)}, "
            f"C:{len(collection.child_chunks)}, "
            f"A:{len(collection.atomic_chunks)})"
        )
        logger.info(f"Distribuição por domínio: {collection.chunks_by_domain}")
        
        return collection
    
    def embed_chunks(
        self,
        chunks: list[Chunk],
        show_progress: bool = True
    ) -> list[Chunk]:
        """
        Gera embeddings para chunks.
        
        Args:
            chunks: Lista de chunks.
            show_progress: Mostrar barra de progresso.
            
        Returns:
            Chunks com embeddings.
        """
        if not self._embedder:
            raise RuntimeError(
                "Embedder não configurado. "
                "Configure VOYAGE_API_KEY."
            )
        
        logger.info(f"Gerando embeddings para {len(chunks)} chunks...")
        
        embedded_chunks = self._embedder.embed_chunks(
            chunks,
            show_progress=show_progress
        )
        
        logger.info("Embeddings gerados com sucesso")
        
        return embedded_chunks
    
    def store_chunks(
        self,
        chunks: list[Chunk],
        collection_name: str,
        create_if_not_exists: bool = True
    ) -> None:
        """
        Armazena chunks no vector store.
        
        Args:
            chunks: Chunks com embeddings.
            collection_name: Nome da coleção.
            create_if_not_exists: Criar coleção se não existir.
        """
        # Criar coleção se necessário
        if create_if_not_exists and not self._vector_store.collection_exists(collection_name):
            self._vector_store.create_collection(
                name=collection_name,
                dimensions=self.settings.embedding.dimensions,
            )
            logger.info(f"Coleção criada: {collection_name}")
        
        # Inserir chunks
        logger.info(f"Armazenando {len(chunks)} chunks em '{collection_name}'...")
        
        self._vector_store.insert_chunks(
            collection=collection_name,
            chunks=chunks,
        )
        
        count = self._vector_store.count(collection_name)
        logger.info(f"Armazenamento concluído. Total na coleção: {count}")
    
    def ingest_document(
        self,
        file_path: str | Path,
        collection_name: str | None = None,
        chunk_levels: list[ChunkLevel] | None = None
    ) -> dict[str, Any]:
        """
        Pipeline completo de ingestão.
        
        1. Carrega documento
        2. Gera chunks
        3. Gera embeddings
        4. Armazena no vector store
        
        Args:
            file_path: Caminho do arquivo.
            collection_name: Nome da coleção. Se None, usa nome do arquivo.
            chunk_levels: Níveis de chunk a processar. Se None, todos.
            
        Returns:
            Estatísticas do processamento.
        """
        path = Path(file_path)
        collection_name = collection_name or f"odeci_{path.stem}"
        
        logger.info(f"=== Iniciando ingestão: {path.name} ===")
        
        # 1. Carregar
        document = self.load_document(path)
        
        # 2. Chunking
        chunk_collection = self.chunk_document(document)
        
        # 3. Selecionar níveis
        if chunk_levels is None:
            chunks = chunk_collection.get_all_chunks()
        else:
            chunks = []
            for level in chunk_levels:
                chunks.extend(chunk_collection.get_chunks_by_level(level))
        
        logger.info(f"Processando {len(chunks)} chunks")
        
        # 4. Embedding
        embedded_chunks = self.embed_chunks(chunks)
        
        # 5. Armazenamento
        self.store_chunks(embedded_chunks, collection_name)
        
        # Estatísticas
        stats = {
            "document_name": document.name,
            "collection_name": collection_name,
            "total_chunks": len(chunks),
            "chunks_by_level": {
                "parent": len(chunk_collection.parent_chunks),
                "child": len(chunk_collection.child_chunks),
                "atomic": len(chunk_collection.atomic_chunks),
            },
            "chunks_by_domain": chunk_collection.chunks_by_domain,
            "document_stats": calculate_text_statistics(document.text),
        }
        
        logger.info(f"=== Ingestão concluída: {collection_name} ===")
        
        return stats
    
    def search(
        self,
        query: str,
        collection_name: str,
        top_k: int = 10,
        rerank: bool = True,
        include_parent: bool = True,
        filter_metadata: dict[str, Any] | None = None
    ) -> list[RetrievalResult]:
        """
        Busca semântica na coleção.
        
        Args:
            query: Texto da busca.
            collection_name: Nome da coleção.
            top_k: Número de resultados.
            rerank: Aplicar reranking.
            include_parent: Incluir contexto do parent.
            filter_metadata: Filtros de metadados.
            
        Returns:
            Lista de resultados ordenados por relevância.
        """
        if not self._retriever:
            raise RuntimeError(
                "Retriever não configurado. "
                "Configure VOYAGE_API_KEY."
            )
        
        logger.info(f"Buscando: '{query}' em '{collection_name}'")
        
        results = self._retriever.search(
            query=query,
            collection=collection_name,
            top_k=top_k,
            rerank=rerank and self._reranker is not None,
            include_parent=include_parent,
            filter_metadata=filter_metadata,
        )
        
        logger.info(f"Encontrados {len(results)} resultados")
        
        return results
    
    def delete_collection(self, collection_name: str) -> None:
        """
        Remove uma coleção.
        
        Args:
            collection_name: Nome da coleção.
        """
        if self._vector_store.collection_exists(collection_name):
            self._vector_store.delete_collection(collection_name)
            logger.info(f"Coleção removida: {collection_name}")
        else:
            logger.warning(f"Coleção não existe: {collection_name}")
    
    def get_collection_stats(self, collection_name: str) -> dict[str, Any]:
        """
        Retorna estatísticas da coleção.
        
        Args:
            collection_name: Nome da coleção.
            
        Returns:
            Dicionário com estatísticas.
        """
        exists = self._vector_store.collection_exists(collection_name)
        
        if not exists:
            return {"exists": False, "name": collection_name}
        
        count = self._vector_store.count(collection_name)
        
        return {
            "exists": True,
            "name": collection_name,
            "vector_count": count,
            "dimensions": self.settings.embedding.dimensions,
            "backend": self.settings.vector_store.backend,
        }


# Função de conveniência para uso rápido
def quick_ingest(
    file_path: str,
    collection_name: str | None = None
) -> dict[str, Any]:
    """
    Ingestão rápida de documento.
    
    Args:
        file_path: Caminho do arquivo.
        collection_name: Nome da coleção opcional.
        
    Returns:
        Estatísticas do processamento.
    """
    pipeline = ODECIPipeline()
    return pipeline.ingest_document(file_path, collection_name)


def quick_search(
    query: str,
    collection_name: str,
    top_k: int = 5
) -> list[RetrievalResult]:
    """
    Busca rápida em coleção.
    
    Args:
        query: Texto da busca.
        collection_name: Nome da coleção.
        top_k: Número de resultados.
        
    Returns:
        Lista de resultados.
    """
    pipeline = ODECIPipeline()
    return pipeline.search(query, collection_name, top_k)
