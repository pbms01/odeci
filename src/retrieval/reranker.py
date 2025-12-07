"""
Reranker para refinamento de resultados de busca.

Usa Cohere Rerank para reordenar resultados por relevância.
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)


@dataclass
class RerankResult:
    """Resultado de reranking."""
    
    index: int
    text: str
    relevance_score: float
    
    def to_dict(self) -> dict[str, Any]:
        """Converte para dicionário."""
        return {
            "index": self.index,
            "text": self.text,
            "relevance_score": self.relevance_score,
        }


class BaseReranker(ABC):
    """Interface base para rerankers."""
    
    @abstractmethod
    def rerank(
        self,
        query: str,
        documents: list[str],
        top_n: int = 10
    ) -> list[dict[str, Any]]:
        """
        Reordena documentos por relevância à query.
        
        Args:
            query: Texto da query.
            documents: Lista de documentos.
            top_n: Número de resultados.
            
        Returns:
            Lista de resultados reordenados.
        """
        pass


class CohereReranker(BaseReranker):
    """
    Reranker usando Cohere Rerank API.
    
    Modelos disponíveis:
    - rerank-multilingual-v3.0: Multilíngue (recomendado para pt-br)
    - rerank-english-v3.0: Inglês otimizado
    - rerank-v3.5: Mais recente
    
    Attributes:
        model: Nome do modelo Cohere.
        top_n: Número padrão de resultados.
    """
    
    AVAILABLE_MODELS = [
        "rerank-multilingual-v3.0",
        "rerank-english-v3.0", 
        "rerank-v3.5",
        "rerank-english-v2.0",
        "rerank-multilingual-v2.0",
    ]
    
    def __init__(
        self,
        model: str = "rerank-multilingual-v3.0",
        api_key: str | None = None,
        top_n: int = 10,
    ) -> None:
        """
        Inicializa o reranker Cohere.
        
        Args:
            model: Nome do modelo.
            api_key: Chave da API (ou usar env var).
            top_n: Número padrão de resultados.
        """
        if model not in self.AVAILABLE_MODELS:
            logger.warning(
                f"Modelo '{model}' pode não estar disponível. "
                f"Recomendados: {self.AVAILABLE_MODELS}"
            )
        
        self._model = model
        self._top_n = top_n
        self._api_key = api_key or os.getenv("COHERE_API_KEY")
        
        if not self._api_key:
            raise ValueError(
                "API key Cohere não configurada. "
                "Defina COHERE_API_KEY ou passe api_key."
            )
        
        # Inicializar cliente
        self._client = self._init_client()
        
        logger.info(f"CohereReranker inicializado: model={model}")
    
    def _init_client(self):
        """Inicializa cliente Cohere."""
        try:
            import cohere
            return cohere.Client(api_key=self._api_key)
        except ImportError:
            raise ImportError(
                "Pacote 'cohere' não instalado. "
                "Execute: pip install cohere"
            )
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    def rerank(
        self,
        query: str,
        documents: list[str],
        top_n: int | None = None
    ) -> list[dict[str, Any]]:
        """
        Reordena documentos por relevância à query.
        
        Args:
            query: Texto da query.
            documents: Lista de documentos (máx 1000).
            top_n: Número de resultados (default: self._top_n).
            
        Returns:
            Lista de dicionários com index, text, relevance_score.
        """
        if not documents:
            return []
        
        top_n = top_n or self._top_n
        top_n = min(top_n, len(documents))
        
        try:
            response = self._client.rerank(
                model=self._model,
                query=query,
                documents=documents,
                top_n=top_n,
                return_documents=True,
            )
            
            results = []
            for item in response.results:
                results.append({
                    "index": item.index,
                    "text": documents[item.index],
                    "relevance_score": item.relevance_score,
                })
            
            logger.debug(
                f"Rerank: {len(documents)} docs → {len(results)} resultados"
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Erro no rerank Cohere: {e}")
            raise
    
    def rerank_with_metadata(
        self,
        query: str,
        documents: list[dict[str, Any]],
        text_key: str = "text",
        top_n: int | None = None
    ) -> list[dict[str, Any]]:
        """
        Reordena documentos preservando metadados.
        
        Args:
            query: Texto da query.
            documents: Lista de dicts com texto e metadados.
            text_key: Chave do texto no dict.
            top_n: Número de resultados.
            
        Returns:
            Documentos reordenados com metadados preservados.
        """
        if not documents:
            return []
        
        # Extrair textos
        texts = [doc[text_key] for doc in documents]
        
        # Reranking
        reranked = self.rerank(query, texts, top_n)
        
        # Reconstruir com metadados
        results = []
        for item in reranked:
            idx = item["index"]
            result = documents[idx].copy()
            result["relevance_score"] = item["relevance_score"]
            result["original_index"] = idx
            results.append(result)
        
        return results


class CrossEncoderReranker(BaseReranker):
    """
    Reranker usando Cross-Encoder local.
    
    Alternativa gratuita ao Cohere usando sentence-transformers.
    Mais lento mas sem custos de API.
    """
    
    DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-12-v2"
    MULTILINGUAL_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    
    def __init__(
        self,
        model: str | None = None,
        device: str = "cpu",
        multilingual: bool = True,
    ) -> None:
        """
        Inicializa o Cross-Encoder.
        
        Args:
            model: Nome do modelo HuggingFace.
            device: Dispositivo (cpu/cuda).
            multilingual: Usar modelo multilíngue.
        """
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            raise ImportError(
                "Pacote 'sentence-transformers' não instalado. "
                "Execute: pip install sentence-transformers"
            )
        
        if model is None:
            model = self.MULTILINGUAL_MODEL if multilingual else self.DEFAULT_MODEL
        
        self._model_name = model
        self._model = CrossEncoder(model, device=device)
        
        logger.info(f"CrossEncoderReranker inicializado: {model}")
    
    def rerank(
        self,
        query: str,
        documents: list[str],
        top_n: int = 10
    ) -> list[dict[str, Any]]:
        """
        Reordena documentos usando Cross-Encoder.
        
        Args:
            query: Texto da query.
            documents: Lista de documentos.
            top_n: Número de resultados.
            
        Returns:
            Lista de resultados reordenados.
        """
        if not documents:
            return []
        
        # Criar pares query-documento
        pairs = [(query, doc) for doc in documents]
        
        # Calcular scores
        scores = self._model.predict(pairs)
        
        # Ordenar por score
        indexed_scores = list(enumerate(scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Retornar top_n
        results = []
        for idx, score in indexed_scores[:top_n]:
            results.append({
                "index": idx,
                "text": documents[idx],
                "relevance_score": float(score),
            })
        
        return results


def create_reranker(
    provider: str = "cohere",
    **kwargs: Any
) -> BaseReranker:
    """
    Factory para criar reranker.
    
    Args:
        provider: Provedor (cohere, cross-encoder).
        **kwargs: Argumentos do reranker.
        
    Returns:
        Instância do reranker.
    """
    providers = {
        "cohere": CohereReranker,
        "cross-encoder": CrossEncoderReranker,
    }
    
    if provider not in providers:
        raise ValueError(
            f"Provedor '{provider}' não suportado. "
            f"Disponíveis: {list(providers.keys())}"
        )
    
    return providers[provider](**kwargs)
