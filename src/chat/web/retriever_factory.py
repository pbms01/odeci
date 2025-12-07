"""
Factory para inicialização do retriever e componentes relacionados.

Centraliza a criação do HybridRetriever com todas as suas dependências.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.config import Settings

logger = logging.getLogger(__name__)


class RetrieverInitError(Exception):
    """Erro na inicialização do retriever."""
    pass


def create_retriever(settings: Settings | None = None):
    """
    Cria instância completa do HybridRetriever.

    Inicializa:
    1. Vector Store (Qdrant ou ChromaDB)
    2. Embedder (VoyageAI)
    3. Reranker (Cohere) - opcional
    4. HybridRetriever

    Args:
        settings: Configurações da aplicação.

    Returns:
        Instância do HybridRetriever configurada.

    Raises:
        RetrieverInitError: Se não for possível inicializar.
    """
    from src.config import get_settings
    from src.retrieval.retriever import HybridRetriever

    settings = settings or get_settings()

    logger.info("Inicializando retriever com componentes reais...")

    # 1. Criar Vector Store
    vector_store = _create_vector_store(settings)

    # 2. Criar Embedder
    embedder = _create_embedder(settings)

    # 3. Criar Reranker (opcional)
    reranker = _create_reranker(settings)

    # 4. Criar HybridRetriever
    retriever = HybridRetriever(
        vector_store=vector_store,
        embedder=embedder,
        reranker=reranker,
        config=settings.retrieval,
    )

    logger.info("HybridRetriever inicializado com sucesso")
    return retriever


def _create_vector_store(settings: Settings):
    """
    Cria vector store baseado na configuração.

    Args:
        settings: Configurações.

    Returns:
        Instância do vector store.
    """
    from src.storage.vector_store import create_vector_store

    backend = settings.vector_store.backend
    logger.info(f"Criando vector store: {backend}")

    if backend == "qdrant":
        return create_vector_store(
            backend="qdrant",
            path=settings.vector_store.qdrant.path,
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
        )
    elif backend == "chroma":
        chroma_path = getattr(settings.vector_store, 'chroma', None)
        path = chroma_path.path if chroma_path else "./data/chroma"
        return create_vector_store(
            backend="chroma",
            path=path,
        )
    else:
        raise RetrieverInitError(f"Backend não suportado: {backend}")


def _create_embedder(settings: Settings):
    """
    Cria embedder baseado na configuração.

    Args:
        settings: Configurações.

    Returns:
        Instância do embedder.
    """
    provider = settings.embedding.provider
    logger.info(f"Criando embedder: {provider}")

    if provider == "voyage":
        if not settings.voyage_api_key:
            raise RetrieverInitError(
                "VOYAGE_API_KEY não configurada. "
                "Configure a variável de ambiente ou no .env"
            )

        from src.embedding.voyage_embedder import VoyageEmbedder

        return VoyageEmbedder(
            model=settings.embedding.models.general,
            dimensions=settings.embedding.dimensions,
            output_dtype=settings.embedding.output_dtype,
            batch_size=settings.embedding.batch_size,
            api_key=settings.voyage_api_key,
        )
    elif provider == "openai":
        if not settings.openai_api_key:
            raise RetrieverInitError("OPENAI_API_KEY não configurada")

        # Implementar OpenAI embedder se necessário
        raise RetrieverInitError("OpenAI embedder não implementado ainda")
    else:
        raise RetrieverInitError(f"Provider de embedding não suportado: {provider}")


def _create_reranker(settings: Settings):
    """
    Cria reranker se configurado.

    Args:
        settings: Configurações.

    Returns:
        Instância do reranker ou None.
    """
    if not settings.retrieval.reranking.enabled:
        logger.info("Reranking desabilitado")
        return None

    provider = settings.retrieval.reranking.provider
    logger.info(f"Criando reranker: {provider}")

    if provider == "cohere":
        if not settings.cohere_api_key:
            logger.warning(
                "COHERE_API_KEY não configurada. Reranking desabilitado."
            )
            return None

        from src.retrieval.reranker import CohereReranker

        return CohereReranker(
            api_key=settings.cohere_api_key,
            model=settings.retrieval.reranking.model,
            top_n=settings.retrieval.reranking.top_n,
        )
    else:
        logger.warning(f"Provider de reranking não suportado: {provider}")
        return None


def get_available_collections(settings: Settings | None = None) -> list[str]:
    """
    Lista coleções disponíveis no vector store.

    Args:
        settings: Configurações.

    Returns:
        Lista de nomes de coleções.
    """
    from src.config import get_settings

    settings = settings or get_settings()

    try:
        vector_store = _create_vector_store(settings)

        # Tentar listar coleções do Qdrant
        if hasattr(vector_store, '_client'):
            try:
                collections = vector_store._client.get_collections().collections
                names = [c.name for c in collections]
                if names:
                    return names
            except Exception:
                pass

        # Fallback para coleções padrão da configuração
        prefix = settings.vector_store.collections.prefix
        namespaces = settings.vector_store.collections.namespaces
        return [f"{prefix}{ns}" for ns in namespaces]

    except Exception as e:
        logger.warning(f"Erro ao listar coleções: {e}")
        return ["default", "juridico_tech"]


def check_collection_exists(
    collection: str,
    settings: Settings | None = None
) -> bool:
    """
    Verifica se uma coleção existe.

    Args:
        collection: Nome da coleção.
        settings: Configurações.

    Returns:
        True se a coleção existe.
    """
    from src.config import get_settings

    settings = settings or get_settings()

    try:
        vector_store = _create_vector_store(settings)
        return vector_store.collection_exists(collection)
    except Exception:
        return False


def get_collection_stats(
    collection: str,
    settings: Settings | None = None
) -> dict:
    """
    Obtém estatísticas de uma coleção.

    Args:
        collection: Nome da coleção.
        settings: Configurações.

    Returns:
        Dict com estatísticas (count, etc).
    """
    from src.config import get_settings

    settings = settings or get_settings()

    try:
        vector_store = _create_vector_store(settings)

        if not vector_store.collection_exists(collection):
            return {"exists": False, "count": 0}

        count = vector_store.count(collection)
        return {
            "exists": True,
            "count": count,
            "collection": collection,
        }
    except Exception as e:
        logger.error(f"Erro ao obter stats da coleção: {e}")
        return {"exists": False, "count": 0, "error": str(e)}


class RetrieverManager:
    """
    Gerenciador singleton do retriever.

    Mantém uma única instância do retriever para reutilização.
    """

    _instance: 'RetrieverManager | None' = None
    _retriever = None
    _settings = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def get_retriever(self, settings: Settings | None = None):
        """
        Obtém ou cria instância do retriever.

        Args:
            settings: Configurações.

        Returns:
            Instância do HybridRetriever.
        """
        if self._retriever is None:
            self._retriever = create_retriever(settings)
            self._settings = settings
        return self._retriever

    def reset(self):
        """Reseta a instância do retriever."""
        self._retriever = None
        self._settings = None

    @property
    def is_initialized(self) -> bool:
        """Verifica se o retriever está inicializado."""
        return self._retriever is not None


# Singleton global
retriever_manager = RetrieverManager()
