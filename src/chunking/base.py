"""
Interface base para chunkers.

Define o contrato que todas as implementações de chunking devem seguir.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.models.chunk import ChunkCollection
    from src.models.document import Document


class BaseChunker(ABC):
    """
    Interface abstrata para chunkers.
    
    Define métodos que todas as implementações devem fornecer.
    """
    
    @abstractmethod
    def chunk(self, document: Document) -> ChunkCollection:
        """
        Processa documento e gera coleção de chunks.
        
        Args:
            document: Documento a ser processado.
            
        Returns:
            Coleção de chunks organizados hierarquicamente.
        """
        pass
    
    @abstractmethod
    def estimate_tokens(self, text: str) -> int:
        """
        Estima número de tokens em um texto.
        
        Args:
            text: Texto para estimativa.
            
        Returns:
            Número estimado de tokens.
        """
        pass
    
    @abstractmethod
    def split_by_tokens(
        self,
        text: str,
        max_tokens: int,
        overlap_tokens: int = 0
    ) -> list[str]:
        """
        Divide texto em partes com número máximo de tokens.
        
        Args:
            text: Texto a ser dividido.
            max_tokens: Número máximo de tokens por parte.
            overlap_tokens: Tokens de sobreposição entre partes.
            
        Returns:
            Lista de partes do texto.
        """
        pass
