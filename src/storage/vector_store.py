"""
Implementações de vector stores.

Suporta Qdrant (local e cloud) e ChromaDB.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, TYPE_CHECKING
from uuid import UUID

if TYPE_CHECKING:
    from src.config import VectorStoreConfig

from src.storage.base import BaseVectorStore, SearchResult
from src.models.chunk import Chunk, ChunkLevel, ChunkMetadata, Domain

logger = logging.getLogger(__name__)


class QdrantVectorStore(BaseVectorStore):
    """
    Vector store usando Qdrant.
    
    Suporta modo local (arquivo) e cloud (API).
    
    Attributes:
        client: Cliente Qdrant.
        mode: Modo de operação (local/cloud).
    """
    
    def __init__(
        self,
        path: str | None = None,
        url: str | None = None,
        api_key: str | None = None,
        config: VectorStoreConfig | None = None,
    ) -> None:
        """
        Inicializa o Qdrant vector store.
        
        Args:
            path: Caminho para armazenamento local.
            url: URL do servidor Qdrant.
            api_key: Chave da API (para cloud).
            config: Configuração completa.
        """
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams
        except ImportError:
            raise ImportError(
                "Pacote 'qdrant-client' não instalado. "
                "Execute: pip install qdrant-client"
            )
        
        self._Distance = Distance
        self._VectorParams = VectorParams
        
        # Configuração
        if config:
            path = config.qdrant.path
            url = config.qdrant.url
            api_key = config.qdrant.api_key
        
        # Determinar modo
        if url:
            self._mode = "cloud"
            self._client = QdrantClient(
                url=url,
                api_key=api_key or os.getenv("QDRANT_API_KEY"),
            )
        else:
            self._mode = "local"
            storage_path = path or "./data/qdrant"
            Path(storage_path).mkdir(parents=True, exist_ok=True)
            self._client = QdrantClient(path=storage_path)
        
        logger.info(f"QdrantVectorStore inicializado em modo: {self._mode}")
    
    def create_collection(
        self,
        name: str,
        dimensions: int,
        metric: str = "cosine",
        **kwargs: Any
    ) -> None:
        """
        Cria uma nova coleção.
        
        Args:
            name: Nome da coleção.
            dimensions: Dimensões dos vetores.
            metric: Métrica de similaridade.
            **kwargs: Configurações HNSW.
        """
        from qdrant_client.models import Distance, VectorParams
        
        # Mapear métrica
        distance_map = {
            "cosine": Distance.COSINE,
            "dot": Distance.DOT,
            "euclidean": Distance.EUCLID,
        }
        distance = distance_map.get(metric, Distance.COSINE)
        
        # Criar coleção
        self._client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(
                size=dimensions,
                distance=distance,
            ),
        )
        
        logger.info(f"Coleção criada: {name} (dims={dimensions}, metric={metric})")
    
    def delete_collection(self, name: str) -> None:
        """Remove uma coleção."""
        self._client.delete_collection(collection_name=name)
        logger.info(f"Coleção removida: {name}")
    
    def collection_exists(self, name: str) -> bool:
        """Verifica se coleção existe."""
        try:
            collections = self._client.get_collections().collections
            return any(c.name == name for c in collections)
        except Exception:
            return False
    
    def insert_chunk(
        self,
        collection: str,
        chunk: Chunk,
        namespace: str | None = None
    ) -> None:
        """Insere um chunk na coleção."""
        self.insert_chunks(collection, [chunk], namespace)
    
    def insert_chunks(
        self,
        collection: str,
        chunks: list[Chunk],
        namespace: str | None = None
    ) -> None:
        """
        Insere múltiplos chunks.
        
        Args:
            collection: Nome da coleção.
            chunks: Lista de chunks (devem ter embeddings).
            namespace: Namespace opcional (adicionado aos metadados).
        """
        from qdrant_client.models import PointStruct
        
        points = []
        
        for chunk in chunks:
            if not chunk.has_embedding:
                logger.warning(f"Chunk {chunk.id} sem embedding, ignorando")
                continue
            
            # Preparar payload
            payload = chunk.to_vector_payload()
            if namespace:
                payload["namespace"] = namespace
            
            points.append(PointStruct(
                id=str(chunk.id),
                vector=chunk.embedding,
                payload=payload,
            ))
        
        if points:
            self._client.upsert(
                collection_name=collection,
                points=points,
            )
            logger.debug(f"Inseridos {len(points)} chunks em {collection}")
    
    def search(
        self,
        collection: str,
        query_vector: list[float],
        top_k: int = 10,
        namespace: str | None = None,
        filter_metadata: dict[str, Any] | None = None
    ) -> list[SearchResult]:
        """
        Busca vetores similares.

        Args:
            collection: Nome da coleção.
            query_vector: Vetor de query.
            top_k: Número de resultados.
            namespace: Namespace para filtrar.
            filter_metadata: Filtros adicionais.

        Returns:
            Lista de resultados ordenados por similaridade.
        """
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        print(f"[ODECI QdrantStore.search] collection={collection}, top_k={top_k}")
        print(f"[ODECI QdrantStore.search] query_vector dim={len(query_vector)}, first 5 values={query_vector[:5]}")
        print(f"[ODECI QdrantStore.search] namespace={namespace}, filter_metadata={filter_metadata}")

        # Construir filtro
        conditions = []

        if namespace:
            conditions.append(FieldCondition(
                key="namespace",
                match=MatchValue(value=namespace),
            ))

        if filter_metadata:
            for key, value in filter_metadata.items():
                conditions.append(FieldCondition(
                    key=key,
                    match=MatchValue(value=value),
                ))

        query_filter = Filter(must=conditions) if conditions else None
        print(f"[ODECI QdrantStore.search] query_filter={query_filter}")

        # Executar busca (qdrant-client v1.7+ usa query_points)
        try:
            response = self._client.query_points(
                collection_name=collection,
                query=query_vector,
                limit=top_k,
                query_filter=query_filter,
                with_payload=True,
            )
            results = response.points
            print(f"[ODECI QdrantStore.search] Qdrant retornou {len(results)} pontos")

            # Mostrar scores dos resultados
            if results:
                scores = [r.score for r in results[:5]]
                print(f"[ODECI QdrantStore.search] Top 5 scores: {scores}")
        except Exception as e:
            print(f"[ODECI QdrantStore.search] ERRO na busca: {e}")
            logger.error(f"Erro na busca Qdrant: {e}")
            return []

        # Converter resultados
        search_results = []
        for result in results:
            payload = result.payload or {}
            search_results.append(SearchResult(
                chunk_id=str(result.id),
                text=payload.get("text", ""),
                score=result.score,
                metadata=payload,
            ))

        print(f"[ODECI QdrantStore.search] Retornando {len(search_results)} resultados")
        return search_results
    
    def get_chunk_by_id(
        self,
        collection: str,
        chunk_id: str
    ) -> Chunk | None:
        """Recupera chunk por ID."""
        try:
            results = self._client.retrieve(
                collection_name=collection,
                ids=[chunk_id],
                with_payload=True,
                with_vectors=True,
            )
            
            if not results:
                return None
            
            point = results[0]
            payload = point.payload or {}
            
            # Reconstruir chunk
            metadata = ChunkMetadata(
                document_id=UUID(payload.get("document_id", "")),
                document_name=payload.get("document_name", ""),
                section=payload.get("section"),
                domain=Domain(payload.get("domain", "general")),
                has_code=payload.get("has_code", False),
                token_count=payload.get("token_count", 0),
                language=payload.get("language", "pt-br"),
            )
            
            return Chunk(
                id=UUID(chunk_id),
                text=payload.get("text", ""),
                level=ChunkLevel(payload.get("level", "child")),
                parent_id=UUID(payload["parent_id"]) if payload.get("parent_id") else None,
                metadata=metadata,
                embedding=point.vector,
            )
        except Exception as e:
            logger.error(f"Erro ao recuperar chunk {chunk_id}: {e}")
            return None
    
    def count(
        self,
        collection: str,
        namespace: str | None = None
    ) -> int:
        """Conta vetores na coleção."""
        try:
            info = self._client.get_collection(collection_name=collection)
            return info.points_count
        except Exception:
            return 0


class ChromaVectorStore(BaseVectorStore):
    """
    Vector store usando ChromaDB.
    
    Alternativa leve para desenvolvimento local.
    """
    
    def __init__(
        self,
        path: str | None = None,
        config: VectorStoreConfig | None = None,
    ) -> None:
        """
        Inicializa o ChromaDB vector store.
        
        Args:
            path: Caminho para armazenamento.
            config: Configuração completa.
        """
        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError:
            raise ImportError(
                "Pacote 'chromadb' não instalado. "
                "Execute: pip install chromadb"
            )
        
        # Configuração
        if config:
            path = config.chroma.path if hasattr(config, 'chroma') else "./data/chroma"
        
        storage_path = path or "./data/chroma"
        Path(storage_path).mkdir(parents=True, exist_ok=True)
        
        self._client = chromadb.PersistentClient(
            path=storage_path,
            settings=Settings(anonymized_telemetry=False),
        )
        
        logger.info(f"ChromaVectorStore inicializado em: {storage_path}")
    
    def create_collection(
        self,
        name: str,
        dimensions: int,
        metric: str = "cosine",
        **kwargs: Any
    ) -> None:
        """Cria uma nova coleção."""
        # ChromaDB cria automaticamente na primeira inserção
        self._client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": metric},
        )
        logger.info(f"Coleção criada/obtida: {name}")
    
    def delete_collection(self, name: str) -> None:
        """Remove uma coleção."""
        self._client.delete_collection(name=name)
        logger.info(f"Coleção removida: {name}")
    
    def collection_exists(self, name: str) -> bool:
        """Verifica se coleção existe."""
        try:
            collections = self._client.list_collections()
            return any(c.name == name for c in collections)
        except Exception:
            return False
    
    def insert_chunk(
        self,
        collection: str,
        chunk: Chunk,
        namespace: str | None = None
    ) -> None:
        """Insere um chunk na coleção."""
        self.insert_chunks(collection, [chunk], namespace)
    
    def insert_chunks(
        self,
        collection: str,
        chunks: list[Chunk],
        namespace: str | None = None
    ) -> None:
        """Insere múltiplos chunks."""
        col = self._client.get_or_create_collection(name=collection)
        
        ids = []
        embeddings = []
        documents = []
        metadatas = []
        
        for chunk in chunks:
            if not chunk.has_embedding:
                continue
            
            payload = chunk.to_vector_payload()
            if namespace:
                payload["namespace"] = namespace
            
            ids.append(str(chunk.id))
            embeddings.append(chunk.embedding)
            documents.append(chunk.text)
            metadatas.append(payload)
        
        if ids:
            col.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )
            logger.debug(f"Inseridos {len(ids)} chunks em {collection}")
    
    def search(
        self,
        collection: str,
        query_vector: list[float],
        top_k: int = 10,
        namespace: str | None = None,
        filter_metadata: dict[str, Any] | None = None
    ) -> list[SearchResult]:
        """Busca vetores similares."""
        col = self._client.get_collection(name=collection)
        
        # Construir filtro
        where_filter = {}
        if namespace:
            where_filter["namespace"] = namespace
        if filter_metadata:
            where_filter.update(filter_metadata)
        
        results = col.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            where=where_filter if where_filter else None,
            include=["embeddings", "documents", "metadatas", "distances"],
        )
        
        # Converter resultados
        search_results = []
        
        if results["ids"] and results["ids"][0]:
            for i, chunk_id in enumerate(results["ids"][0]):
                search_results.append(SearchResult(
                    chunk_id=chunk_id,
                    text=results["documents"][0][i] if results["documents"] else "",
                    score=1 - results["distances"][0][i],  # Converter distância em score
                    metadata=results["metadatas"][0][i] if results["metadatas"] else {},
                ))
        
        return search_results
    
    def get_chunk_by_id(
        self,
        collection: str,
        chunk_id: str
    ) -> Chunk | None:
        """Recupera chunk por ID."""
        try:
            col = self._client.get_collection(name=collection)
            result = col.get(
                ids=[chunk_id],
                include=["embeddings", "documents", "metadatas"],
            )
            
            if not result["ids"]:
                return None
            
            metadata_dict = result["metadatas"][0] if result["metadatas"] else {}
            
            metadata = ChunkMetadata(
                document_id=UUID(metadata_dict.get("document_id", "")),
                document_name=metadata_dict.get("document_name", ""),
                section=metadata_dict.get("section"),
                domain=Domain(metadata_dict.get("domain", "general")),
                has_code=metadata_dict.get("has_code", False),
                token_count=metadata_dict.get("token_count", 0),
            )
            
            return Chunk(
                id=UUID(chunk_id),
                text=result["documents"][0] if result["documents"] else "",
                level=ChunkLevel(metadata_dict.get("level", "child")),
                metadata=metadata,
                embedding=result["embeddings"][0] if result["embeddings"] else None,
            )
        except Exception as e:
            logger.error(f"Erro ao recuperar chunk {chunk_id}: {e}")
            return None
    
    def count(
        self,
        collection: str,
        namespace: str | None = None
    ) -> int:
        """Conta vetores na coleção."""
        try:
            col = self._client.get_collection(name=collection)
            return col.count()
        except Exception:
            return 0


def create_vector_store(
    backend: str = "qdrant",
    **kwargs: Any
) -> BaseVectorStore:
    """
    Factory para criar vector store.
    
    Args:
        backend: Nome do backend (qdrant, chroma).
        **kwargs: Configurações do backend.
        
    Returns:
        Instância do vector store.
    """
    backends = {
        "qdrant": QdrantVectorStore,
        "chroma": ChromaVectorStore,
    }
    
    if backend not in backends:
        raise ValueError(
            f"Backend '{backend}' não suportado. "
            f"Disponíveis: {list(backends.keys())}"
        )
    
    return backends[backend](**kwargs)
