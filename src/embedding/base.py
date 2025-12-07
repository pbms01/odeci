"""
Interface base para embedders.

Define o contrato que todas as implementações de embedding devem seguir.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.models.chunk import Chunk


@dataclass
class EmbeddingResult:
    """Resultado de uma operação de embedding."""
    
    text: str
    embedding: list[float]
    model: str
    dimensions: int
    tokens_used: int
    
    def __repr__(self) -> str:
        return (
            f"EmbeddingResult(model={self.model}, "
            f"dims={self.dimensions}, tokens={self.tokens_used})"
        )


class BaseEmbedder(ABC):
    """
    Interface abstrata para embedders.
    
    Define métodos que todas as implementações devem fornecer.
    """
    
    @abstractmethod
    def embed_text(self, text: str, input_type: str = "document") -> EmbeddingResult:
        """
        Gera embedding para um texto.
        
        Args:
            text: Texto para embedding.
            input_type: Tipo de input (document, query).
            
        Returns:
            Resultado do embedding.
        """
        pass
    
    @abstractmethod
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
        pass
    
    @abstractmethod
    def embed_chunk(self, chunk: Chunk) -> Chunk:
        """
        Gera embedding para um chunk.
        
        Args:
            chunk: Chunk para embedding.
            
        Returns:
            Chunk com embedding preenchido.
        """
        pass
    
    @abstractmethod
    def embed_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        """
        Gera embeddings para múltiplos chunks.
        
        Args:
            chunks: Lista de chunks.
            
        Returns:
            Lista de chunks com embeddings.
        """
        pass
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Nome do modelo de embedding."""
        pass
    
    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Dimensões do embedding."""
        pass
