"""
Embedder híbrido multi-modelo.

Roteia chunks para modelos especializados baseado no domínio:
- Legal → voyage-law-2
- Code → voyage-code-3
- Tech/General → voyage-3-large
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import TYPE_CHECKING

from tqdm import tqdm

if TYPE_CHECKING:
    from src.config import EmbeddingConfig

from src.embedding.base import BaseEmbedder, EmbeddingResult
from src.embedding.voyage_embedder import VoyageEmbedder, VoyageEmbedderFactory
from src.models.chunk import Chunk, Domain

logger = logging.getLogger(__name__)


class HybridEmbedder(BaseEmbedder):
    """
    Embedder híbrido que usa modelos especializados por domínio.
    
    Estratégia:
    1. Agrupa chunks por domínio
    2. Processa cada grupo com modelo especializado
    3. Combina resultados
    
    Modelos:
    - LEGAL → voyage-law-2
    - CODE → voyage-code-3
    - TECH → voyage-3-large
    - GENERAL → voyage-3-large
    
    Attributes:
        embedders: Dicionário de embedders por domínio.
        default_model: Modelo padrão para domínios não mapeados.
    """
    
    def __init__(
        self,
        api_key: str | None = None,
        dimensions: int = 1024,
        config: EmbeddingConfig | None = None,
    ) -> None:
        """
        Inicializa o embedder híbrido.
        
        Args:
            api_key: Chave da API Voyage.
            dimensions: Dimensões do embedding.
            config: Configuração completa.
        """
        self._dimensions = dimensions
        self._api_key = api_key
        
        # Modelos por domínio
        if config:
            self._model_map = {
                Domain.LEGAL: config.models.legal,
                Domain.CODE: config.models.code,
                Domain.TECH: config.models.general,
                Domain.GENERAL: config.models.general,
            }
            dimensions = config.dimensions
        else:
            self._model_map = {
                Domain.LEGAL: "voyage-law-2",
                Domain.CODE: "voyage-code-3",
                Domain.TECH: "voyage-3-large",
                Domain.GENERAL: "voyage-3-large",
            }
        
        # Inicializar embedders (lazy loading)
        self._embedders: dict[str, VoyageEmbedder] = {}

        # Modelo padrão para queries - usar modelo legal para documentos jurídicos
        # Isso garante compatibilidade com chunks embedados com voyage-law-2
        if config:
            self._default_model = config.models.legal  # voyage-law-2
        else:
            self._default_model = "voyage-law-2"
        
        logger.info(
            f"HybridEmbedder inicializado com mapeamento: {self._model_map}"
        )
    
    def _get_embedder(self, model: str) -> VoyageEmbedder:
        """
        Obtém ou cria embedder para um modelo.
        
        Args:
            model: Nome do modelo.
            
        Returns:
            Instância do embedder.
        """
        if model not in self._embedders:
            self._embedders[model] = VoyageEmbedder(
                model=model,
                dimensions=self._dimensions,
                api_key=self._api_key,
            )
        
        return self._embedders[model]
    
    def _get_model_for_domain(self, domain: Domain) -> str:
        """
        Retorna modelo apropriado para o domínio.
        
        Args:
            domain: Domínio do chunk.
            
        Returns:
            Nome do modelo.
        """
        return self._model_map.get(domain, self._default_model)
    
    @property
    def model_name(self) -> str:
        """Nome do modelo (híbrido)."""
        return "hybrid-multi-model"
    
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
        Gera embedding para um texto usando modelo padrão.
        
        Args:
            text: Texto para embedding.
            input_type: Tipo de input.
            
        Returns:
            Resultado do embedding.
        """
        embedder = self._get_embedder(self._default_model)
        return embedder.embed_text(text, input_type)
    
    def embed_texts(
        self,
        texts: list[str],
        input_type: str = "document"
    ) -> list[EmbeddingResult]:
        """
        Gera embeddings para múltiplos textos usando modelo padrão.
        
        Args:
            texts: Lista de textos.
            input_type: Tipo de input.
            
        Returns:
            Lista de resultados.
        """
        embedder = self._get_embedder(self._default_model)
        return embedder.embed_texts(texts, input_type)
    
    def embed_chunk(self, chunk: Chunk) -> Chunk:
        """
        Gera embedding para um chunk usando modelo apropriado.
        
        Args:
            chunk: Chunk para embedding.
            
        Returns:
            Chunk com embedding preenchido.
        """
        domain = Domain(chunk.metadata.domain)
        model = self._get_model_for_domain(domain)
        embedder = self._get_embedder(model)
        
        return embedder.embed_chunk(chunk)
    
    def embed_chunks(
        self,
        chunks: list[Chunk],
        show_progress: bool = True
    ) -> list[Chunk]:
        """
        Gera embeddings para múltiplos chunks.
        
        Agrupa por domínio e processa com modelo especializado.
        
        Args:
            chunks: Lista de chunks.
            show_progress: Mostrar progresso.
            
        Returns:
            Lista de chunks com embeddings.
        """
        # Agrupar chunks por modelo
        chunks_by_model: dict[str, list[tuple[int, Chunk]]] = defaultdict(list)
        
        for idx, chunk in enumerate(chunks):
            domain = Domain(chunk.metadata.domain)
            model = self._get_model_for_domain(domain)
            chunks_by_model[model].append((idx, chunk))
        
        logger.info(f"Distribuição por modelo: {
            {k: len(v) for k, v in chunks_by_model.items()}
        }")
        
        # Processar cada grupo
        results: dict[int, Chunk] = {}
        
        for model, indexed_chunks in chunks_by_model.items():
            logger.info(f"Processando {len(indexed_chunks)} chunks com {model}")
            
            embedder = self._get_embedder(model)
            
            # Extrair chunks mantendo índices
            indices = [ic[0] for ic in indexed_chunks]
            model_chunks = [ic[1] for ic in indexed_chunks]
            
            # Processar
            embedded_chunks = embedder.embed_chunks(
                model_chunks,
                show_progress=show_progress
            )
            
            # Mapear resultados
            for idx, chunk in zip(indices, embedded_chunks):
                results[idx] = chunk
        
        # Ordenar por índice original
        return [results[i] for i in range(len(chunks))]
    
    def embed_query(
        self,
        query: str,
        domain: Domain | None = None
    ) -> list[float]:
        """
        Gera embedding para query de busca.
        
        Args:
            query: Texto da query.
            domain: Domínio para seleção de modelo (opcional).
            
        Returns:
            Vetor de embedding.
        """
        if domain:
            model = self._get_model_for_domain(domain)
        else:
            # Usar modelo geral para queries
            model = self._default_model
        
        embedder = self._get_embedder(model)
        return embedder.embed_query(query)
    
    def get_statistics(self) -> dict[str, int]:
        """
        Retorna estatísticas de uso dos modelos.
        
        Returns:
            Dicionário com contagens por modelo.
        """
        return {
            model: embedder._batch_size  # Placeholder
            for model, embedder in self._embedders.items()
        }
