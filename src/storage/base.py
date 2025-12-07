"""
Interface base para vector stores.

Define o contrato que todas as implementações de armazenamento devem seguir.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from src.models.chunk import Chunk


@dataclass
class SearchResult:
    """Resultado de uma busca vetorial."""
    
    chunk_id: str
    text: str
    score: float
    metadata: dict[str, Any]
    
    # Contexto adicional
    parent_text: str | None = None
    
    def __repr__(self) -> str:
        preview = self.text[:50] + "..." if len(self.text) > 50 else self.text
        return f"SearchResult(score={self.score:.4f}, text='{preview}')"


class BaseVectorStore(ABC):
    """
    Interface abstrata para vector stores.
    
    Define métodos que todas as implementações devem fornecer.
    """
    
    @abstractmethod
    def create_collection(
        self,
        name: str,
        dimensions: int,
        **kwargs: Any
    ) -> None:
        """
        Cria uma nova coleção.
        
        Args:
            name: Nome da coleção.
            dimensions: Dimensões dos vetores.
            **kwargs: Configurações adicionais.
        """
        pass
    
    @abstractmethod
    def delete_collection(self, name: str) -> None:
        """
        Remove uma coleção.
        
        Args:
            name: Nome da coleção.
        """
        pass
    
    @abstractmethod
    def collection_exists(self, name: str) -> bool:
        """
        Verifica se coleção existe.
        
        Args:
            name: Nome da coleção.
            
        Returns:
            True se existe.
        """
        pass
    
    @abstractmethod
    def insert_chunk(
        self,
        collection: str,
        chunk: Chunk,
        namespace: str | None = None
    ) -> None:
        """
        Insere um chunk na coleção.
        
        Args:
            collection: Nome da coleção.
            chunk: Chunk a inserir.
            namespace: Namespace opcional.
        """
        pass
    
    @abstractmethod
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
            chunks: Lista de chunks.
            namespace: Namespace opcional.
        """
        pass
    
    @abstractmethod
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
            namespace: Namespace opcional.
            filter_metadata: Filtros de metadados.
            
        Returns:
            Lista de resultados ordenados por similaridade.
        """
        pass
    
    @abstractmethod
    def get_chunk_by_id(
        self,
        collection: str,
        chunk_id: str
    ) -> Chunk | None:
        """
        Recupera chunk por ID.
        
        Args:
            collection: Nome da coleção.
            chunk_id: ID do chunk.
            
        Returns:
            Chunk ou None se não encontrado.
        """
        pass
    
    @abstractmethod
    def count(
        self,
        collection: str,
        namespace: str | None = None
    ) -> int:
        """
        Conta vetores na coleção.
        
        Args:
            collection: Nome da coleção.
            namespace: Namespace opcional.
            
        Returns:
            Número de vetores.
        """
        pass
