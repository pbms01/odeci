"""
Retriever multi-modelo para busca em coleções com embeddings heterogêneos.

Resolve o problema de chunks embedados com modelos diferentes (voyage-law-2,
voyage-code-3, voyage-3-large) fazendo busca paralela por domínio.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.config import RetrievalConfig
    from src.embedding.hybrid_embedder import HybridEmbedder
    from src.storage.base import BaseVectorStore

from src.models.chunk import Domain
from src.storage.base import SearchResult
from src.retrieval.retriever import RetrievalResult

logger = logging.getLogger(__name__)


@dataclass
class DomainSearchResult:
    """Resultado de busca por domínio."""
    domain: Domain
    results: list[SearchResult]
    query_model: str


class MultiModelRetriever:
    """
    Retriever que busca em múltiplos domínios com modelos especializados.

    Estratégia:
    1. Detecta domínio provável da query (ou usa todos)
    2. Gera embedding da query com cada modelo relevante
    3. Busca em paralelo filtrando por domínio
    4. Combina resultados com RRF (Reciprocal Rank Fusion)
    5. Aplica reranking opcional

    Isso resolve o problema de chunks embedados com modelos diferentes
    não serem comparáveis diretamente.

    Attributes:
        vector_store: Store de vetores.
        embedder: HybridEmbedder com múltiplos modelos.
        reranker: Reranker opcional.
    """

    # Mapeamento domínio → modelo
    DOMAIN_MODELS = {
        Domain.LEGAL: "voyage-law-2",
        Domain.CODE: "voyage-code-3",
        Domain.TECH: "voyage-3-large",
        Domain.GENERAL: "voyage-3-large",
    }

    def __init__(
        self,
        vector_store: BaseVectorStore,
        embedder: HybridEmbedder,
        reranker: Any | None = None,
        config: RetrievalConfig | None = None,
    ) -> None:
        """
        Inicializa o retriever multi-modelo.

        Args:
            vector_store: Store de vetores.
            embedder: HybridEmbedder com acesso aos modelos especializados.
            reranker: Reranker opcional (Cohere, Cross-Encoder).
            config: Configuração de retrieval.
        """
        self._store = vector_store
        self._embedder = embedder
        self._reranker = reranker
        self._config = config

        # Configurações
        self._top_k_per_domain = 10 if config is None else config.top_k_per_namespace
        self._final_top_k = 10 if config is None else config.final_top_k
        self._include_parent = True if config is None else config.include_parent_context

        logger.info("MultiModelRetriever inicializado")

    def _detect_query_domains(self, query: str) -> list[Domain]:
        """
        Detecta domínios relevantes para a query.

        Usa heurísticas simples para identificar se a query
        é sobre código, jurídico, etc.

        Args:
            query: Texto da query.

        Returns:
            Lista de domínios relevantes.
        """
        query_lower = query.lower()

        domains = []

        # Detectar código
        code_keywords = [
            "solidity", "smart contract", "contrato inteligente",
            "function", "require", "mapping", "uint", "address",
            "ethereum", "evm", "bytecode", "abi", "deploy"
        ]
        if any(kw in query_lower for kw in code_keywords):
            domains.append(Domain.CODE)

        # Detectar jurídico
        legal_keywords = [
            "lei", "artigo", "jurisprudência", "contrato",
            "cláusula", "jurisdição", "tribunal", "direito",
            "obrigação", "rescisão", "inadimplemento", "dano",
            "indenização", "responsabilidade", "stf", "stj"
        ]
        if any(kw in query_lower for kw in legal_keywords):
            domains.append(Domain.LEGAL)

        # Detectar técnico
        tech_keywords = [
            "blockchain", "descentralizado", "protocolo",
            "algoritmo", "criptografia", "hash", "merkle",
            "consenso", "nó", "rede", "p2p", "defi", "dao"
        ]
        if any(kw in query_lower for kw in tech_keywords):
            domains.append(Domain.TECH)

        # Se nenhum detectado, buscar em todos
        if not domains:
            domains = [Domain.LEGAL, Domain.CODE, Domain.TECH, Domain.GENERAL]

        return domains

    def _embed_query_for_domain(self, query: str, domain: Domain) -> list[float]:
        """
        Gera embedding da query usando modelo do domínio.

        Args:
            query: Texto da query.
            domain: Domínio alvo.

        Returns:
            Vetor de embedding.
        """
        return self._embedder.embed_query(query, domain=domain)

    def _search_domain(
        self,
        collection: str,
        query: str,
        domain: Domain,
        top_k: int,
        filter_metadata: dict[str, Any] | None = None
    ) -> DomainSearchResult:
        """
        Busca em um domínio específico com modelo apropriado.

        Args:
            collection: Nome da coleção.
            query: Texto da query.
            domain: Domínio para buscar.
            top_k: Número de resultados.
            filter_metadata: Filtros adicionais.

        Returns:
            Resultado da busca no domínio.
        """
        # Gerar embedding com modelo do domínio
        model = self.DOMAIN_MODELS[domain]
        query_vector = self._embed_query_for_domain(query, domain)

        # Filtrar por domínio
        domain_filter = {"domain": domain.value}
        if filter_metadata:
            domain_filter.update(filter_metadata)

        # Buscar
        results = self._store.search(
            collection=collection,
            query_vector=query_vector,
            top_k=top_k,
            filter_metadata=domain_filter,
        )

        logger.debug(f"Domínio {domain.value}: {len(results)} resultados com {model}")

        return DomainSearchResult(
            domain=domain,
            results=results,
            query_model=model,
        )

    def _combine_domain_results(
        self,
        domain_results: list[DomainSearchResult]
    ) -> list[SearchResult]:
        """
        Combina resultados de múltiplos domínios usando RRF.

        Reciprocal Rank Fusion (RRF) é robusto para combinar
        rankings de diferentes fontes/modelos.

        Args:
            domain_results: Resultados por domínio.

        Returns:
            Lista combinada ordenada por score RRF.
        """
        k = 60  # Constante RRF

        # Acumular scores RRF
        rrf_scores: dict[str, tuple[SearchResult, float]] = {}

        for dr in domain_results:
            for rank, result in enumerate(dr.results):
                rrf_score = 1 / (k + rank + 1)

                if result.chunk_id in rrf_scores:
                    existing = rrf_scores[result.chunk_id]
                    rrf_scores[result.chunk_id] = (result, existing[1] + rrf_score)
                else:
                    rrf_scores[result.chunk_id] = (result, rrf_score)

        # Ordenar por score RRF
        sorted_results = sorted(
            rrf_scores.values(),
            key=lambda x: x[1],
            reverse=True
        )

        return [r[0] for r in sorted_results]

    def _fetch_parent_context(
        self,
        collection: str,
        results: list[SearchResult]
    ) -> list[SearchResult]:
        """Adiciona contexto do parent chunk aos resultados."""
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
        domains: list[Domain] | None = None,
        top_k: int | None = None,
        filter_metadata: dict[str, Any] | None = None,
        rerank: bool = True,
        include_parent: bool | None = None,
        auto_detect_domains: bool = True
    ) -> list[RetrievalResult]:
        """
        Busca semântica multi-modelo.

        Args:
            query: Texto da busca.
            collection: Nome da coleção.
            domains: Domínios para buscar (None = auto-detectar ou todos).
            top_k: Número final de resultados.
            filter_metadata: Filtros de metadados.
            rerank: Aplicar reranking.
            include_parent: Incluir contexto do parent.
            auto_detect_domains: Auto-detectar domínios da query.

        Returns:
            Lista de resultados ordenados por relevância.
        """
        top_k = top_k or self._final_top_k
        include_parent = include_parent if include_parent is not None else self._include_parent

        # Determinar domínios
        if domains is None:
            if auto_detect_domains:
                domains = self._detect_query_domains(query)
                logger.info(f"Domínios detectados: {[d.value for d in domains]}")
            else:
                domains = list(Domain)

        logger.info(f"Buscando: '{query}' em {len(domains)} domínios")

        # Buscar em cada domínio com modelo apropriado
        domain_results: list[DomainSearchResult] = []

        for domain in domains:
            try:
                dr = self._search_domain(
                    collection=collection,
                    query=query,
                    domain=domain,
                    top_k=self._top_k_per_domain,
                    filter_metadata=filter_metadata,
                )
                domain_results.append(dr)
            except Exception as e:
                logger.warning(f"Erro na busca do domínio {domain.value}: {e}")

        # Combinar resultados
        results = self._combine_domain_results(domain_results)

        logger.debug(f"Combinados {len(results)} resultados únicos")

        # Reranking
        rerank_scores: dict[str, float] = {}

        if rerank and self._reranker and results:
            try:
                reranked = self._reranker.rerank(
                    query=query,
                    documents=[r.text for r in results],
                    top_n=top_k,
                )

                for item in reranked:
                    idx = item.get("index", 0)
                    score = item.get("relevance_score", 0)
                    if idx < len(results):
                        rerank_scores[results[idx].chunk_id] = score

                results = sorted(
                    results,
                    key=lambda r: rerank_scores.get(r.chunk_id, 0),
                    reverse=True
                )[:top_k]

                logger.debug("Reranking aplicado")
            except Exception as e:
                logger.warning(f"Erro no reranking: {e}")
                results = results[:top_k]
        else:
            results = results[:top_k]

        # Contexto do parent
        if include_parent:
            results = self._fetch_parent_context(collection, results)

        # Converter para RetrievalResult
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

    def search_single_domain(
        self,
        query: str,
        collection: str,
        domain: Domain,
        top_k: int = 10,
        **kwargs: Any
    ) -> list[RetrievalResult]:
        """
        Busca em um único domínio (atalho).

        Args:
            query: Texto da busca.
            collection: Nome da coleção.
            domain: Domínio específico.
            top_k: Número de resultados.
            **kwargs: Argumentos adicionais.

        Returns:
            Lista de resultados.
        """
        return self.search(
            query=query,
            collection=collection,
            domains=[domain],
            top_k=top_k,
            auto_detect_domains=False,
            **kwargs
        )
