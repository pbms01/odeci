"""
Retriever híbrido para busca semântica.

Combina busca densa (embeddings) com reranking.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.config import RetrievalConfig
    from src.embedding.base import BaseEmbedder
    from src.storage.base import BaseVectorStore

from src.storage.base import SearchResult

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Resultado final de retrieval com contexto."""
    
    chunk_id: str
    text: str
    score: float
    rerank_score: float | None
    metadata: dict[str, Any]
    
    # Contexto do parent
    parent_text: str | None = None
    
    @property
    def final_score(self) -> float:
        """Score final (rerank se disponível, senão original)."""
        return self.rerank_score if self.rerank_score is not None else self.score
    
    def __repr__(self) -> str:
        preview = self.text[:50] + "..." if len(self.text) > 50 else self.text
        return f"RetrievalResult(score={self.final_score:.4f}, text='{preview}')"


class HybridRetriever:
    """
    Retriever híbrido com suporte a múltiplos namespaces.
    
    Estratégia:
    1. Busca em múltiplos namespaces (domínios)
    2. Combina resultados
    3. Opcional: Reranking
    4. Opcional: Inclui contexto do parent chunk
    
    Attributes:
        vector_store: Store de vetores.
        embedder: Modelo de embedding.
        reranker: Reranker opcional.
    """
    
    def __init__(
        self,
        vector_store: BaseVectorStore,
        embedder: BaseEmbedder,
        reranker: Any | None = None,
        config: RetrievalConfig | None = None,
    ) -> None:
        """
        Inicializa o retriever.
        
        Args:
            vector_store: Store de vetores.
            embedder: Modelo de embedding para queries.
            reranker: Reranker opcional.
            config: Configuração de retrieval.
        """
        self._store = vector_store
        self._embedder = embedder
        self._reranker = reranker
        self._config = config
        
        # Configurações default
        self._top_k_per_namespace = 20 if config is None else config.top_k_per_namespace
        self._final_top_k = 10 if config is None else config.final_top_k
        self._include_parent = True if config is None else config.include_parent_context
        
        logger.info("HybridRetriever inicializado")
    
    def _embed_query(self, query: str) -> list[float]:
        """
        Gera embedding para query.

        Args:
            query: Texto da query.

        Returns:
            Vetor de embedding.
        """
        # Usar método específico para query se disponível
        if hasattr(self._embedder, 'embed_query'):
            return self._embedder.embed_query(query)

        result = self._embedder.embed_text(query, input_type="query")
        return result.embedding

    def _embed_query_multi_model(self, query: str) -> dict[str, list[float]]:
        """
        Gera embeddings para query usando múltiplos modelos.

        Args:
            query: Texto da query.

        Returns:
            Dict mapeando modelo -> vetor de embedding.
        """
        if hasattr(self._embedder, 'embed_query_multi_model'):
            return self._embedder.embed_query_multi_model(query)

        # Fallback: usar método padrão
        return {"default": self._embed_query(query)}
    
    def _search_namespace(
        self,
        collection: str,
        query_vector: list[float],
        namespace: str,
        top_k: int,
        filter_metadata: dict[str, Any] | None = None
    ) -> list[SearchResult]:
        """
        Busca em um namespace específico.
        
        Args:
            collection: Nome da coleção.
            query_vector: Vetor da query.
            namespace: Namespace para buscar.
            top_k: Número de resultados.
            filter_metadata: Filtros adicionais.
            
        Returns:
            Lista de resultados.
        """
        return self._store.search(
            collection=collection,
            query_vector=query_vector,
            top_k=top_k,
            namespace=namespace,
            filter_metadata=filter_metadata,
        )
    
    def _combine_results(
        self,
        results_by_namespace: dict[str, list[SearchResult]]
    ) -> list[SearchResult]:
        """
        Combina resultados de múltiplos namespaces.
        
        Usa RRF (Reciprocal Rank Fusion) simplificado.
        
        Args:
            results_by_namespace: Resultados por namespace.
            
        Returns:
            Lista combinada e ordenada.
        """
        # Coletar todos os resultados com scores
        all_results: dict[str, tuple[SearchResult, float]] = {}
        
        k = 60  # Constante RRF
        
        for namespace, results in results_by_namespace.items():
            for rank, result in enumerate(results):
                rrf_score = 1 / (k + rank + 1)
                
                if result.chunk_id in all_results:
                    # Combinar scores
                    existing = all_results[result.chunk_id]
                    combined_score = existing[1] + rrf_score
                    all_results[result.chunk_id] = (result, combined_score)
                else:
                    all_results[result.chunk_id] = (result, rrf_score)
        
        # Ordenar por score combinado
        sorted_results = sorted(
            all_results.values(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return [r[0] for r in sorted_results]
    
    def _fetch_parent_context(
        self,
        collection: str,
        results: list[SearchResult]
    ) -> list[SearchResult]:
        """
        Adiciona contexto do parent chunk aos resultados.
        
        Args:
            collection: Nome da coleção.
            results: Resultados da busca.
            
        Returns:
            Resultados com parent_text preenchido.
        """
        for result in results:
            parent_id = result.metadata.get("parent_id")
            if parent_id:
                parent_chunk = self._store.get_chunk_by_id(collection, parent_id)
                if parent_chunk:
                    result.parent_text = parent_chunk.text
        
        return results
    
    def search(
        self,
        query: str,
        collection: str,
        namespaces: list[str] | None = None,
        top_k: int | None = None,
        filter_metadata: dict[str, Any] | None = None,
        rerank: bool = True,
        include_parent: bool | None = None
    ) -> list[RetrievalResult]:
        """
        Executa busca semântica.
        
        Args:
            query: Texto da query.
            collection: Nome da coleção.
            namespaces: Namespaces para buscar (None = todos).
            top_k: Número final de resultados.
            filter_metadata: Filtros de metadados.
            rerank: Aplicar reranking.
            include_parent: Incluir contexto do parent.
            
        Returns:
            Lista de resultados ordenados por relevância.
        """
        top_k = top_k or self._final_top_k
        include_parent = include_parent if include_parent is not None else self._include_parent
        
        logger.info(f"Buscando: '{query}' em {collection}")
        print(f"[ODECI Search] Query: '{query[:50]}...' em collection={collection}")

        # 1. Gerar embeddings da query com múltiplos modelos
        # Isso é necessário porque chunks foram embedados com diferentes modelos
        # baseado no domínio detectado durante a ingestão
        try:
            query_embeddings = self._embed_query_multi_model(query)
            print(f"[ODECI Search] Embeddings gerados com {len(query_embeddings)} modelo(s): {list(query_embeddings.keys())}")
            logger.info(f"Query embedada com {len(query_embeddings)} modelo(s): {list(query_embeddings.keys())}")
        except Exception as e:
            print(f"[ODECI Search] ERRO ao gerar embeddings: {e}")
            logger.error(f"Erro ao gerar embeddings: {e}")
            return []

        # 2. Buscar com cada modelo e combinar resultados
        all_results: dict[str, SearchResult] = {}

        search_debug = []  # Para diagnóstico
        for model, query_vector in query_embeddings.items():
            logger.info(f"Buscando com modelo {model}, vetor dim={len(query_vector)}")
            print(f"[ODECI Search] Buscando com {model}, dim={len(query_vector)}, top_k={self._top_k_per_namespace}")
            try:
                if namespaces is None:
                    # Busca geral sem namespace
                    model_results = self._store.search(
                        collection=collection,
                        query_vector=query_vector,
                        top_k=self._top_k_per_namespace,
                        filter_metadata=filter_metadata,
                    )
                else:
                    # Buscar em cada namespace
                    results_by_namespace = {}
                    for namespace in namespaces:
                        ns_results = self._search_namespace(
                            collection=collection,
                            query_vector=query_vector,
                            namespace=namespace,
                            top_k=self._top_k_per_namespace,
                            filter_metadata=filter_metadata,
                        )
                        results_by_namespace[namespace] = ns_results

                    # Combinar resultados dos namespaces
                    model_results = self._combine_results(results_by_namespace)

                # Diagnóstico detalhado
                scores = [r.score for r in model_results[:5]] if model_results else []
                debug_msg = f"{model}: {len(model_results)} resultados, scores={scores}"
                search_debug.append(debug_msg)
                print(f"[ODECI Search] {debug_msg}")
                logger.info(f"Modelo {model}: {len(model_results)} resultados encontrados")

                # Adicionar resultados, mantendo o melhor score se duplicado
                for result in model_results:
                    if result.chunk_id not in all_results:
                        all_results[result.chunk_id] = result
                    elif result.score > all_results[result.chunk_id].score:
                        all_results[result.chunk_id] = result

            except Exception as e:
                logger.error(f"Erro ao buscar com modelo {model}: {e}")
                import traceback
                logger.error(traceback.format_exc())

        # Converter para lista ordenada por score
        results = sorted(all_results.values(), key=lambda r: r.score, reverse=True)

        # Salvar debug info para acesso externo
        self._last_search_debug = {
            "models": list(query_embeddings.keys()),
            "results_per_model": search_debug,
            "total_results": len(results),
        }

        print(f"[ODECI Search] Total de resultados combinados: {len(results)}")
        print(f"[ODECI Search] Debug por modelo: {search_debug}")
        logger.info(f"Encontrados {len(results)} resultados iniciais")
        
        # 3. Reranking
        rerank_scores: dict[str, float] = {}
        
        if rerank and self._reranker and results:
            try:
                reranked = self._reranker.rerank(
                    query=query,
                    documents=[r.text for r in results],
                    top_n=top_k,
                )
                
                # Mapear scores de rerank
                for item in reranked:
                    idx = item.get("index", 0)
                    score = item.get("relevance_score", 0)
                    if idx < len(results):
                        rerank_scores[results[idx].chunk_id] = score
                
                # Reordenar por score de rerank
                results = sorted(
                    results,
                    key=lambda r: rerank_scores.get(r.chunk_id, 0),
                    reverse=True
                )[:top_k]
                
                logger.debug("Reranking aplicado")
            except Exception as e:
                logger.warning(f"Erro no reranking: {e}. Usando ordem original.")
        else:
            results = results[:top_k]
        
        # 4. Contexto do parent
        if include_parent:
            results = self._fetch_parent_context(collection, results)
        
        # 5. Converter para RetrievalResult
        final_results = [
            RetrievalResult(
                chunk_id=r.chunk_id,
                text=r.text,
                score=r.score,
                rerank_score=rerank_scores.get(r.chunk_id),
                metadata=r.metadata,
                parent_text=r.parent_text,
            )
            for r in results
        ]
        
        logger.info(f"Retornando {len(final_results)} resultados")
        
        return final_results
    
    def search_similar(
        self,
        chunk_id: str,
        collection: str,
        top_k: int = 5,
        exclude_self: bool = True
    ) -> list[RetrievalResult]:
        """
        Busca chunks similares a um chunk dado.
        
        Args:
            chunk_id: ID do chunk de referência.
            collection: Nome da coleção.
            top_k: Número de resultados.
            exclude_self: Excluir o próprio chunk.
            
        Returns:
            Lista de chunks similares.
        """
        # Recuperar chunk
        chunk = self._store.get_chunk_by_id(collection, chunk_id)
        if not chunk or not chunk.embedding:
            logger.warning(f"Chunk {chunk_id} não encontrado ou sem embedding")
            return []
        
        # Buscar similares
        results = self._store.search(
            collection=collection,
            query_vector=chunk.embedding,
            top_k=top_k + (1 if exclude_self else 0),
        )
        
        # Filtrar self se necessário
        if exclude_self:
            results = [r for r in results if r.chunk_id != chunk_id][:top_k]
        
        return [
            RetrievalResult(
                chunk_id=r.chunk_id,
                text=r.text,
                score=r.score,
                rerank_score=None,
                metadata=r.metadata,
            )
            for r in results
        ]
