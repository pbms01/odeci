"""
Modelo de Chunk para representação de fragmentos de documento.

Define a estrutura hierárquica de chunks (parent, child, atomic)
com metadados para classificação de domínio e retrieval.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, computed_field


class ChunkLevel(str, Enum):
    """Níveis hierárquicos de chunks."""
    
    PARENT = "parent"    # Contexto completo (2000+ tokens)
    CHILD = "child"      # Unidade de retrieval (300-500 tokens)
    ATOMIC = "atomic"    # Definições/dados pontuais (50-150 tokens)


class Domain(str, Enum):
    """Domínios de classificação de conteúdo."""
    
    LEGAL = "legal"      # Conteúdo jurídico
    CODE = "code"        # Código Solidity/smart contracts
    TECH = "tech"        # Conteúdo técnico geral
    GENERAL = "general"  # Conteúdo não classificado


class ChunkMetadata(BaseModel):
    """
    Metadados associados a um chunk.
    
    Usado para filtragem e contextualização no retrieval.
    """
    
    # Identificação
    document_id: UUID
    document_name: str
    
    # Posição no documento
    section: str | None = None
    chapter: str | None = None
    page_number: int | None = None
    start_char: int | None = None
    end_char: int | None = None
    
    # Classificação
    domain: Domain = Domain.GENERAL
    content_type: str | None = None  # concept, case, code, argument, data
    
    # Entidades extraídas
    entities: list[str] = Field(default_factory=list)
    
    # Para casos jurídicos
    case_name: str | None = None
    case_year: int | None = None
    jurisdiction: str | None = None
    case_outcome: str | None = None
    
    # Para código
    has_code: bool = False
    programming_language: str | None = None
    
    # Temporal
    temporal_ref: str | None = None
    
    # Idioma
    language: str = "pt-br"
    
    # Estatísticas
    token_count: int = 0
    word_count: int = 0
    char_count: int = 0
    
    class Config:
        use_enum_values = True


class Chunk(BaseModel):
    """
    Representa um fragmento de documento para embedding.
    
    Estrutura hierárquica:
    - PARENT: Contexto amplo para reranking
    - CHILD: Unidade principal de retrieval
    - ATOMIC: Definições e dados pontuais
    
    Attributes:
        id: Identificador único do chunk
        text: Conteúdo textual do chunk
        level: Nível hierárquico (parent/child/atomic)
        parent_id: ID do chunk pai (para child/atomic)
        children_ids: IDs dos chunks filhos (para parent)
        metadata: Metadados para filtragem e contexto
        embedding: Vetor de embedding (preenchido após processamento)
        created_at: Timestamp de criação
    """
    
    # Identificação
    id: UUID = Field(default_factory=uuid4)
    
    # Conteúdo
    text: str
    
    # Hierarquia
    level: ChunkLevel
    parent_id: UUID | None = None
    children_ids: list[UUID] = Field(default_factory=list)
    
    # Metadados
    metadata: ChunkMetadata
    
    # Embedding (preenchido após processamento)
    embedding: list[float] | None = None
    embedding_model: str | None = None
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    @computed_field
    @property
    def domain(self) -> str:
        """Retorna o domínio do chunk."""
        return self.metadata.domain
    
    @computed_field
    @property
    def has_embedding(self) -> bool:
        """Verifica se o chunk possui embedding."""
        return self.embedding is not None and len(self.embedding) > 0
    
    def to_vector_payload(self) -> dict[str, Any]:
        """
        Converte para payload de armazenamento vetorial.
        
        Returns:
            Dicionário com dados para inserção no vector store.
        """
        return {
            "id": str(self.id),
            "text": self.text,
            "level": self.level.value,
            "parent_id": str(self.parent_id) if self.parent_id else None,
            "domain": self.metadata.domain,
            "section": self.metadata.section,
            "chapter": self.metadata.chapter,
            "document_id": str(self.metadata.document_id),
            "document_name": self.metadata.document_name,
            "content_type": self.metadata.content_type,
            "entities": self.metadata.entities,
            "case_name": self.metadata.case_name,
            "has_code": self.metadata.has_code,
            "token_count": self.metadata.token_count,
            "language": self.metadata.language,
        }
    
    def get_context_window(self, max_tokens: int = 2000) -> str:
        """
        Retorna texto com contexto para exibição.
        
        Args:
            max_tokens: Número máximo de tokens aproximado.
            
        Returns:
            Texto truncado se necessário.
        """
        # Aproximação: 1 token ≈ 4 caracteres
        max_chars = max_tokens * 4
        
        if len(self.text) <= max_chars:
            return self.text
        
        return self.text[:max_chars] + "..."
    
    def __repr__(self) -> str:
        """Representação legível do chunk."""
        text_preview = self.text[:50] + "..." if len(self.text) > 50 else self.text
        return (
            f"Chunk(id={self.id}, level={self.level.value}, "
            f"domain={self.domain}, text='{text_preview}')"
        )
    
    def __hash__(self) -> int:
        """Hash baseado no ID."""
        return hash(self.id)
    
    def __eq__(self, other: object) -> bool:
        """Igualdade baseada no ID."""
        if not isinstance(other, Chunk):
            return False
        return self.id == other.id


class ChunkCollection(BaseModel):
    """
    Coleção de chunks de um documento.
    
    Organiza chunks por nível hierárquico e domínio.
    """
    
    document_id: UUID
    document_name: str
    
    # Chunks organizados por nível
    parent_chunks: list[Chunk] = Field(default_factory=list)
    child_chunks: list[Chunk] = Field(default_factory=list)
    atomic_chunks: list[Chunk] = Field(default_factory=list)
    
    # Estatísticas
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    @computed_field
    @property
    def total_chunks(self) -> int:
        """Total de chunks na coleção."""
        return (
            len(self.parent_chunks) +
            len(self.child_chunks) +
            len(self.atomic_chunks)
        )
    
    @computed_field
    @property
    def chunks_by_domain(self) -> dict[str, int]:
        """Contagem de chunks por domínio."""
        all_chunks = (
            self.parent_chunks +
            self.child_chunks +
            self.atomic_chunks
        )
        
        counts: dict[str, int] = {}
        for chunk in all_chunks:
            domain = chunk.domain
            counts[domain] = counts.get(domain, 0) + 1
        
        return counts
    
    def get_all_chunks(self) -> list[Chunk]:
        """Retorna todos os chunks da coleção."""
        return (
            self.parent_chunks +
            self.child_chunks +
            self.atomic_chunks
        )
    
    def get_chunks_by_level(self, level: ChunkLevel) -> list[Chunk]:
        """Retorna chunks de um nível específico."""
        if level == ChunkLevel.PARENT:
            return self.parent_chunks
        elif level == ChunkLevel.CHILD:
            return self.child_chunks
        else:
            return self.atomic_chunks
    
    def get_chunks_by_domain(self, domain: Domain) -> list[Chunk]:
        """Retorna chunks de um domínio específico."""
        return [
            chunk for chunk in self.get_all_chunks()
            if chunk.metadata.domain == domain
        ]
