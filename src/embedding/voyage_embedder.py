"""
Embedder usando Voyage AI.

Implementação de embedding usando modelos Voyage AI
(voyage-3-large, voyage-law-2, voyage-code-3).
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Literal

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from tqdm import tqdm

if TYPE_CHECKING:
    from src.config import EmbeddingConfig

from src.embedding.base import BaseEmbedder, EmbeddingResult
from src.models.chunk import Chunk

logger = logging.getLogger(__name__)


class VoyageAPIError(Exception):
    """Erro na API Voyage."""
    pass


class VoyageEmbedder(BaseEmbedder):
    """
    Embedder usando Voyage AI API.
    
    Suporta modelos:
    - voyage-3-large: Geral e multilíngue
    - voyage-law-2: Domínio jurídico
    - voyage-code-3: Código e smart contracts
    - voyage-multilingual-2: Multilíngue especializado
    
    Features:
    - Matryoshka embeddings (dimensões configuráveis)
    - Quantização (float, int8, binary)
    - Batch processing com rate limiting
    - Retry automático em erros transitórios
    
    Attributes:
        model: Nome do modelo Voyage AI.
        dimensions: Dimensões do embedding.
        batch_size: Tamanho do batch para processamento.
    """
    
    # Modelos disponíveis
    AVAILABLE_MODELS = {
        "voyage-3-large": {"max_dims": 2048, "default_dims": 1024},
        "voyage-3": {"max_dims": 1024, "default_dims": 1024},
        "voyage-3-lite": {"max_dims": 512, "default_dims": 512},
        "voyage-law-2": {"max_dims": 1024, "default_dims": 1024},
        "voyage-code-3": {"max_dims": 2048, "default_dims": 1024},
        "voyage-code-2": {"max_dims": 1536, "default_dims": 1536},
        "voyage-multilingual-2": {"max_dims": 1024, "default_dims": 1024},
        "voyage-finance-2": {"max_dims": 1024, "default_dims": 1024},
    }
    
    def __init__(
        self,
        model: str = "voyage-3-large",
        dimensions: int | None = None,
        output_dtype: Literal["float", "int8", "binary"] = "float",
        batch_size: int = 32,
        api_key: str | None = None,
        config: EmbeddingConfig | None = None,
    ) -> None:
        """
        Inicializa o embedder Voyage AI.
        
        Args:
            model: Nome do modelo.
            dimensions: Dimensões do embedding (Matryoshka).
            output_dtype: Tipo de output.
            batch_size: Tamanho do batch.
            api_key: Chave da API (ou usar env var).
            config: Configuração completa.
        """
        # Configuração
        if config:
            model = config.models.general
            dimensions = config.dimensions
            output_dtype = config.output_dtype
            batch_size = config.batch_size
        
        # Validar modelo
        if model not in self.AVAILABLE_MODELS:
            raise ValueError(
                f"Modelo '{model}' não suportado. "
                f"Disponíveis: {list(self.AVAILABLE_MODELS.keys())}"
            )
        
        self._model = model
        model_config = self.AVAILABLE_MODELS[model]
        
        # Dimensões
        self._dimensions = dimensions or model_config["default_dims"]
        if self._dimensions > model_config["max_dims"]:
            logger.warning(
                f"Dimensões {self._dimensions} excedem máximo "
                f"{model_config['max_dims']} para {model}. Usando máximo."
            )
            self._dimensions = model_config["max_dims"]
        
        self._output_dtype = output_dtype
        self._batch_size = batch_size
        
        # API Key
        self._api_key = api_key or os.getenv("VOYAGE_API_KEY")
        if not self._api_key:
            raise ValueError(
                "API key Voyage não configurada. "
                "Defina VOYAGE_API_KEY ou passe api_key."
            )
        
        # Inicializar cliente
        self._client = self._init_client()
        
        logger.info(
            f"VoyageEmbedder inicializado: model={model}, "
            f"dims={self._dimensions}, dtype={output_dtype}"
        )
    
    def _init_client(self):
        """Inicializa cliente Voyage AI."""
        try:
            import voyageai
            return voyageai.Client(api_key=self._api_key)
        except ImportError:
            raise ImportError(
                "Pacote 'voyageai' não instalado. "
                "Execute: pip install voyageai"
            )
    
    @property
    def model_name(self) -> str:
        """Nome do modelo."""
        return self._model
    
    @property
    def dimensions(self) -> int:
        """Dimensões do embedding."""
        return self._dimensions
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((Exception,)),
        reraise=True,
    )
    def _call_api(
        self,
        texts: list[str],
        input_type: str = "document"
    ) -> list[list[float]]:
        """
        Chama API Voyage com retry automático.
        
        Args:
            texts: Textos para embedding.
            input_type: Tipo de input.
            
        Returns:
            Lista de embeddings.
        """
        try:
            result = self._client.embed(
                texts=texts,
                model=self._model,
                input_type=input_type,
                output_dimension=self._dimensions,
                output_dtype=self._output_dtype,
            )
            return result.embeddings
        except Exception as e:
            logger.error(f"Erro na API Voyage: {e}")
            raise VoyageAPIError(f"Falha na API: {e}") from e
    
    def embed_text(
        self,
        text: str,
        input_type: str = "document"
    ) -> EmbeddingResult:
        """
        Gera embedding para um texto.
        
        Args:
            text: Texto para embedding.
            input_type: Tipo de input (document, query).
            
        Returns:
            Resultado do embedding.
        """
        embeddings = self._call_api([text], input_type)
        
        return EmbeddingResult(
            text=text,
            embedding=embeddings[0],
            model=self._model,
            dimensions=len(embeddings[0]),
            tokens_used=len(text) // 4,  # Estimativa
        )
    
    def embed_texts(
        self,
        texts: list[str],
        input_type: str = "document",
        show_progress: bool = True
    ) -> list[EmbeddingResult]:
        """
        Gera embeddings para múltiplos textos em batches.
        
        Args:
            texts: Lista de textos.
            input_type: Tipo de input.
            show_progress: Mostrar barra de progresso.
            
        Returns:
            Lista de resultados.
        """
        results = []
        
        # Processar em batches
        batches = [
            texts[i:i + self._batch_size]
            for i in range(0, len(texts), self._batch_size)
        ]
        
        iterator = tqdm(batches, desc="Embedding") if show_progress else batches
        
        for batch in iterator:
            embeddings = self._call_api(batch, input_type)
            
            for text, embedding in zip(batch, embeddings):
                results.append(EmbeddingResult(
                    text=text,
                    embedding=embedding,
                    model=self._model,
                    dimensions=len(embedding),
                    tokens_used=len(text) // 4,
                ))
        
        return results
    
    def embed_chunk(self, chunk: Chunk) -> Chunk:
        """
        Gera embedding para um chunk.
        
        Args:
            chunk: Chunk para embedding.
            
        Returns:
            Chunk com embedding preenchido.
        """
        result = self.embed_text(chunk.text, "document")
        
        chunk.embedding = result.embedding
        chunk.embedding_model = self._model
        
        return chunk
    
    def embed_chunks(
        self,
        chunks: list[Chunk],
        show_progress: bool = True
    ) -> list[Chunk]:
        """
        Gera embeddings para múltiplos chunks.
        
        Args:
            chunks: Lista de chunks.
            show_progress: Mostrar progresso.
            
        Returns:
            Lista de chunks com embeddings.
        """
        texts = [chunk.text for chunk in chunks]
        results = self.embed_texts(texts, "document", show_progress)
        
        for chunk, result in zip(chunks, results):
            chunk.embedding = result.embedding
            chunk.embedding_model = self._model
        
        return chunks
    
    def embed_query(self, query: str) -> list[float]:
        """
        Gera embedding para query de busca.
        
        Args:
            query: Texto da query.
            
        Returns:
            Vetor de embedding.
        """
        result = self.embed_text(query, "query")
        return result.embedding


class VoyageEmbedderFactory:
    """
    Factory para criar embedders especializados por domínio.
    
    Facilita criação de embedders com modelo apropriado.
    """
    
    @staticmethod
    def create_legal_embedder(
        api_key: str | None = None,
        dimensions: int = 1024
    ) -> VoyageEmbedder:
        """Cria embedder para domínio jurídico."""
        return VoyageEmbedder(
            model="voyage-law-2",
            dimensions=dimensions,
            api_key=api_key,
        )
    
    @staticmethod
    def create_code_embedder(
        api_key: str | None = None,
        dimensions: int = 1024
    ) -> VoyageEmbedder:
        """Cria embedder para código."""
        return VoyageEmbedder(
            model="voyage-code-3",
            dimensions=dimensions,
            api_key=api_key,
        )
    
    @staticmethod
    def create_general_embedder(
        api_key: str | None = None,
        dimensions: int = 1024
    ) -> VoyageEmbedder:
        """Cria embedder geral."""
        return VoyageEmbedder(
            model="voyage-3-large",
            dimensions=dimensions,
            api_key=api_key,
        )
    
    @staticmethod
    def create_multilingual_embedder(
        api_key: str | None = None,
        dimensions: int = 1024
    ) -> VoyageEmbedder:
        """Cria embedder multilíngue."""
        return VoyageEmbedder(
            model="voyage-multilingual-2",
            dimensions=dimensions,
            api_key=api_key,
        )
