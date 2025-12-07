"""
Embedder unificado com modelo único.

Usa voyage-3-large para todos os domínios, simplificando
a arquitetura e garantindo compatibilidade vetorial.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from tqdm import tqdm

if TYPE_CHECKING:
    from src.config import EmbeddingConfig

from src.embedding.base import BaseEmbedder, EmbeddingResult
from src.embedding.voyage_embedder import VoyageEmbedder
from src.models.chunk import Chunk

logger = logging.getLogger(__name__)


class UnifiedEmbedder(BaseEmbedder):
    """
    Embedder unificado que usa um único modelo para todos os domínios.

    Vantagens sobre multi-modelo:
    - Vetores sempre comparáveis (mesmo espaço vetorial)
    - 1/3 do custo de API
    - 1/3 da latência
    - Código mais simples
    - Reranking compensa diferenças de precisão

    Attributes:
        embedder: Instância do VoyageEmbedder.
        model: Nome do modelo (voyage-3-large por padrão).
    """

    # Modelo padrão - generalista de alta qualidade
    DEFAULT_MODEL = "voyage-3-large"

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        dimensions: int = 1024,
        config: EmbeddingConfig | None = None,
    ) -> None:
        """
        Inicializa o embedder unificado.

        Args:
            api_key: Chave da API Voyage.
            model: Nome do modelo (default: voyage-3-large).
            dimensions: Dimensões do embedding.
            config: Configuração de embedding.
        """
        self._model = model or self.DEFAULT_MODEL
        self._dimensions = dimensions
        self._api_key = api_key

        # Usar modelo da config se disponível
        if config and hasattr(config, 'models') and hasattr(config.models, 'general'):
            self._model = config.models.general

        if config and hasattr(config, 'dimensions'):
            self._dimensions = config.dimensions

        # Inicializar embedder único
        self._embedder = VoyageEmbedder(
            model=self._model,
            dimensions=self._dimensions,
            api_key=self._api_key,
        )

        logger.info(
            f"UnifiedEmbedder inicializado: model={self._model}, "
            f"dims={self._dimensions}"
        )

    @property
    def model_name(self) -> str:
        """Nome do modelo."""
        return self._model

    @property
    def dimensions(self) -> int:
        """Dimensões do embedding."""
        return self._dimensions

    def embed_text(
        self,
        text: str,
        input_type: str = "document"
    ) -> EmbeddingResult:
        """
        Gera embedding para um texto.

        Args:
            text: Texto para embedding.
            input_type: Tipo de input (document/query).

        Returns:
            Resultado do embedding.
        """
        return self._embedder.embed_text(text, input_type)

    def embed_texts(
        self,
        texts: list[str],
        input_type: str = "document"
    ) -> list[EmbeddingResult]:
        """
        Gera embeddings para múltiplos textos.

        Args:
            texts: Lista de textos.
            input_type: Tipo de input.

        Returns:
            Lista de resultados.
        """
        return self._embedder.embed_texts(texts, input_type)

    def embed_chunk(self, chunk: Chunk) -> Chunk:
        """
        Gera embedding para um chunk.

        Args:
            chunk: Chunk para embedding.

        Returns:
            Chunk com embedding preenchido.
        """
        return self._embedder.embed_chunk(chunk)

    def embed_chunks(
        self,
        chunks: list[Chunk],
        show_progress: bool = True
    ) -> list[Chunk]:
        """
        Gera embeddings para múltiplos chunks.

        Todos os chunks usam o mesmo modelo, garantindo
        compatibilidade vetorial para busca.

        Args:
            chunks: Lista de chunks.
            show_progress: Mostrar barra de progresso.

        Returns:
            Lista de chunks com embeddings.
        """
        logger.info(
            f"Gerando embeddings para {len(chunks)} chunks "
            f"com {self._model}"
        )

        return self._embedder.embed_chunks(chunks, show_progress)

    def embed_query(self, query: str) -> list[float]:
        """
        Gera embedding para query de busca.

        Args:
            query: Texto da query.

        Returns:
            Vetor de embedding.
        """
        return self._embedder.embed_query(query)


def create_embedder(
    mode: str = "unified",
    **kwargs
) -> BaseEmbedder:
    """
    Factory para criar embedder.

    Args:
        mode: Modo de operação:
            - "unified": Modelo único (recomendado)
            - "hybrid": Múltiplos modelos por domínio
        **kwargs: Argumentos do embedder.

    Returns:
        Instância do embedder.
    """
    if mode == "unified":
        return UnifiedEmbedder(**kwargs)
    elif mode == "hybrid":
        from src.embedding.hybrid_embedder import HybridEmbedder
        return HybridEmbedder(**kwargs)
    else:
        raise ValueError(f"Modo '{mode}' não suportado. Use 'unified' ou 'hybrid'.")
