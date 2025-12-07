"""
Construtor de contexto para RAG.

Integra com o sistema de retrieval para construir contexto relevante.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from src.chat.models import ChatConfig, RetrievalContext, SourceReference

if TYPE_CHECKING:
    from src.retrieval.retriever import HybridRetriever, RetrievalResult

logger = logging.getLogger(__name__)


class ContextBuilder:
    """
    Construtor de contexto para geração de respostas.

    Integra com o HybridRetriever existente para recuperar
    e formatar contexto relevante.

    Attributes:
        retriever: Instância do retriever híbrido.
        config: Configurações do chat.
    """

    def __init__(
        self,
        retriever: HybridRetriever,
        config: ChatConfig | None = None,
    ) -> None:
        """
        Inicializa o construtor de contexto.

        Args:
            retriever: Instância do HybridRetriever.
            config: Configurações do chat.
        """
        self._retriever = retriever
        self._config = config or ChatConfig()

        logger.info("ContextBuilder inicializado")

    def build(
        self,
        query: str,
        collection: str,
        max_sources: int | None = None,
        max_tokens: int | None = None,
        namespaces: list[str] | None = None,
        filter_metadata: dict[str, Any] | None = None,
        include_parent: bool | None = None,
    ) -> RetrievalContext:
        """
        Constrói contexto a partir de uma query.

        Args:
            query: Pergunta do usuário.
            collection: Nome da coleção para buscar.
            max_sources: Número máximo de fontes.
            max_tokens: Limite de tokens no contexto.
            namespaces: Namespaces específicos para buscar.
            filter_metadata: Filtros de metadados.
            include_parent: Incluir contexto do parent chunk.

        Returns:
            Contexto recuperado formatado.
        """
        max_sources = max_sources or self._config.max_sources
        max_tokens = max_tokens or self._config.max_context_tokens
        include_parent = (
            include_parent
            if include_parent is not None
            else self._config.include_parent_context
        )

        logger.info(f"Construindo contexto para: '{query[:50]}...'")
        print(f"[ODECI ContextBuilder] Buscando em {collection}, max_sources={max_sources}")

        # Buscar documentos relevantes
        results = self._retriever.search(
            query=query,
            collection=collection,
            namespaces=namespaces,
            top_k=max_sources,
            filter_metadata=filter_metadata,
            rerank=True,
            include_parent=include_parent,
        )

        print(f"[ODECI ContextBuilder] Retriever retornou {len(results)} resultados")
        if results:
            for i, r in enumerate(results[:3]):
                print(f"[ODECI ContextBuilder] Resultado {i+1}: score={r.final_score:.4f}, text={r.text[:50]}...")

        if not results:
            logger.warning("Nenhum resultado encontrado no retrieval")
            return RetrievalContext(
                chunks=[],
                query=query,
                total_tokens=0,
            )

        # Converter resultados para SourceReference
        sources = self._convert_results_to_sources(results)

        # Calcular tokens
        total_tokens = self._calculate_tokens(sources)

        # Truncar se necessário
        if total_tokens > max_tokens:
            sources = self._truncate_sources(sources, max_tokens)
            total_tokens = self._calculate_tokens(sources)

        print(f"[ODECI ContextBuilder] Contexto final: {len(sources)} fontes, ~{total_tokens} tokens")
        logger.info(f"Contexto construído: {len(sources)} fontes, ~{total_tokens} tokens")

        return RetrievalContext(
            chunks=sources,
            query=query,
            total_tokens=total_tokens,
        )

    def _convert_results_to_sources(
        self,
        results: list[RetrievalResult],
    ) -> list[SourceReference]:
        """
        Converte resultados do retrieval para SourceReference.

        Args:
            results: Resultados do retrieval.

        Returns:
            Lista de referências de fonte.
        """
        sources = []

        for result in results:
            # Usar parent_text se disponível para contexto mais rico
            text = result.parent_text or result.text

            source = SourceReference(
                chunk_id=result.chunk_id,
                text=text,
                section=result.metadata.get("section"),
                document_name=result.metadata.get("document_name"),
                score=result.final_score,
                page_number=result.metadata.get("page_number"),
            )
            sources.append(source)

        return sources

    def _calculate_tokens(self, sources: list[SourceReference]) -> int:
        """
        Calcula número aproximado de tokens.

        Args:
            sources: Lista de fontes.

        Returns:
            Número aproximado de tokens.
        """
        total_chars = sum(len(s.text) for s in sources)
        # Aproximação: 1 token ≈ 4 caracteres
        return total_chars // 4

    def _truncate_sources(
        self,
        sources: list[SourceReference],
        max_tokens: int,
    ) -> list[SourceReference]:
        """
        Trunca fontes para respeitar limite de tokens.

        Prioriza fontes com maior score.

        Args:
            sources: Lista de fontes.
            max_tokens: Limite de tokens.

        Returns:
            Lista truncada de fontes.
        """
        # Ordenar por score (deve já estar ordenado, mas garantir)
        sorted_sources = sorted(sources, key=lambda s: s.score, reverse=True)

        truncated = []
        current_tokens = 0

        for source in sorted_sources:
            source_tokens = len(source.text) // 4

            if current_tokens + source_tokens > max_tokens:
                # Tentar truncar o texto da fonte
                remaining_tokens = max_tokens - current_tokens
                if remaining_tokens > 100:  # Mínimo útil
                    max_chars = remaining_tokens * 4
                    truncated_source = SourceReference(
                        chunk_id=source.chunk_id,
                        text=source.text[:max_chars] + "...",
                        section=source.section,
                        document_name=source.document_name,
                        score=source.score,
                        page_number=source.page_number,
                    )
                    truncated.append(truncated_source)
                break

            truncated.append(source)
            current_tokens += source_tokens

        return truncated

    def build_with_reformulation(
        self,
        query: str,
        collection: str,
        reformulate_fn: callable | None = None,
        **kwargs,
    ) -> RetrievalContext:
        """
        Constrói contexto com reformulação de query.

        Args:
            query: Pergunta original.
            collection: Nome da coleção.
            reformulate_fn: Função para reformular query.
            **kwargs: Argumentos adicionais para build().

        Returns:
            Contexto recuperado.
        """
        reformulated_query = None

        if reformulate_fn:
            try:
                reformulated_query = reformulate_fn(query)
                logger.info(f"Query reformulada: '{reformulated_query[:50]}...'")
            except Exception as e:
                logger.warning(f"Erro na reformulação: {e}. Usando query original.")

        # Usar query reformulada ou original
        search_query = reformulated_query or query

        context = self.build(
            query=search_query,
            collection=collection,
            **kwargs,
        )

        # Registrar query reformulada no contexto
        if reformulated_query:
            context.reformulated_query = reformulated_query

        return context

    def merge_contexts(
        self,
        contexts: list[RetrievalContext],
        max_tokens: int | None = None,
    ) -> RetrievalContext:
        """
        Mescla múltiplos contextos em um único.

        Útil para queries multi-parte ou busca em múltiplas coleções.

        Args:
            contexts: Lista de contextos a mesclar.
            max_tokens: Limite de tokens do resultado.

        Returns:
            Contexto mesclado.
        """
        max_tokens = max_tokens or self._config.max_context_tokens

        # Coletar todos os chunks únicos
        seen_ids: set[str] = set()
        all_chunks: list[SourceReference] = []

        for ctx in contexts:
            for chunk in ctx.chunks:
                if chunk.chunk_id not in seen_ids:
                    seen_ids.add(chunk.chunk_id)
                    all_chunks.append(chunk)

        # Ordenar por score e truncar
        all_chunks.sort(key=lambda c: c.score, reverse=True)
        truncated = self._truncate_sources(all_chunks, max_tokens)

        # Query combinada
        queries = [ctx.query for ctx in contexts]
        combined_query = " | ".join(queries)

        return RetrievalContext(
            chunks=truncated,
            query=combined_query,
            total_tokens=self._calculate_tokens(truncated),
        )


class ContextBuilderFactory:
    """Factory para criação de ContextBuilder."""

    @classmethod
    def create(
        cls,
        retriever: HybridRetriever,
        config: ChatConfig | None = None,
    ) -> ContextBuilder:
        """
        Cria instância do ContextBuilder.

        Args:
            retriever: Instância do retriever.
            config: Configurações.

        Returns:
            Instância do ContextBuilder.
        """
        return ContextBuilder(retriever=retriever, config=config)
