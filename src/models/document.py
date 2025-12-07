"""
Modelo de Documento para representação de arquivos processados.

Define a estrutura de documentos com metadados e seções extraídas.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, computed_field


class DocumentSection(BaseModel):
    """
    Representa uma seção do documento.
    
    Preserva a estrutura hierárquica de headings.
    """
    
    # Identificação
    id: UUID = Field(default_factory=uuid4)
    
    # Hierarquia
    level: int  # 1 = H1, 2 = H2, etc.
    title: str
    
    # Conteúdo
    content: str
    
    # Posição
    start_page: int | None = None
    end_page: int | None = None
    start_char: int
    end_char: int
    
    # Subseções
    subsections: list[DocumentSection] = Field(default_factory=list)
    
    @computed_field
    @property
    def char_count(self) -> int:
        """Contagem de caracteres."""
        return len(self.content)
    
    @computed_field  
    @property
    def word_count(self) -> int:
        """Contagem aproximada de palavras."""
        return len(self.content.split())


class DocumentMetadata(BaseModel):
    """
    Metadados do documento.
    
    Informações extraídas e inferidas do arquivo.
    """
    
    # Arquivo
    file_name: str
    file_path: str | None = None
    file_size_bytes: int | None = None
    file_type: str  # docx, pdf, txt, md
    
    # Documento
    title: str | None = None
    author: str | None = None
    subject: str | None = None
    keywords: list[str] = Field(default_factory=list)
    
    # Datas
    created_date: datetime | None = None
    modified_date: datetime | None = None
    
    # Estatísticas
    page_count: int | None = None
    word_count: int | None = None
    char_count: int | None = None
    
    # Idioma
    language: str = "pt-br"
    
    # Processamento
    processed_at: datetime = Field(default_factory=datetime.utcnow)
    processing_version: str = "1.0.0"
    
    # Custom
    custom_metadata: dict[str, Any] = Field(default_factory=dict)


class Document(BaseModel):
    """
    Representa um documento processado.
    
    Contém o texto completo, seções estruturadas e metadados.
    
    Attributes:
        id: Identificador único do documento
        text: Texto completo do documento
        sections: Lista de seções estruturadas
        metadata: Metadados do documento
        footnotes: Notas de rodapé extraídas
        tables: Tabelas extraídas
    """
    
    # Identificação
    id: UUID = Field(default_factory=uuid4)
    
    # Conteúdo
    text: str
    sections: list[DocumentSection] = Field(default_factory=list)
    
    # Metadados
    metadata: DocumentMetadata
    
    # Elementos especiais
    footnotes: dict[str, str] = Field(default_factory=dict)  # {ref: texto}
    tables: list[dict[str, Any]] = Field(default_factory=list)
    code_blocks: list[str] = Field(default_factory=list)
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    @computed_field
    @property
    def name(self) -> str:
        """Nome do documento (título ou nome do arquivo)."""
        return self.metadata.title or self.metadata.file_name
    
    @computed_field
    @property
    def has_structure(self) -> bool:
        """Verifica se o documento tem estrutura de seções."""
        return len(self.sections) > 0
    
    @computed_field
    @property
    def has_footnotes(self) -> bool:
        """Verifica se o documento tem notas de rodapé."""
        return len(self.footnotes) > 0
    
    @computed_field
    @property
    def has_code(self) -> bool:
        """Verifica se o documento contém blocos de código."""
        return len(self.code_blocks) > 0
    
    def get_section_by_title(self, title: str) -> DocumentSection | None:
        """
        Busca seção por título.
        
        Args:
            title: Título da seção (case-insensitive, parcial).
            
        Returns:
            Seção encontrada ou None.
        """
        title_lower = title.lower()
        
        def search_recursive(sections: list[DocumentSection]) -> DocumentSection | None:
            for section in sections:
                if title_lower in section.title.lower():
                    return section
                if section.subsections:
                    found = search_recursive(section.subsections)
                    if found:
                        return found
            return None
        
        return search_recursive(self.sections)
    
    def get_text_with_expanded_footnotes(self) -> str:
        """
        Retorna texto com notas de rodapé expandidas inline.
        
        Útil para chunking que preserve contexto das referências.
        """
        if not self.footnotes:
            return self.text
        
        result = self.text
        
        for ref, footnote_text in self.footnotes.items():
            # Padrões comuns de referência: [1], ¹, (1)
            patterns = [
                f"[{ref}]",
                f"¹" if ref == "1" else f"^{ref}",
                f"({ref})",
            ]
            
            expansion = f" [{footnote_text}]"
            
            for pattern in patterns:
                if pattern in result:
                    result = result.replace(pattern, pattern + expansion, 1)
                    break
        
        return result
    
    def to_dict(self) -> dict[str, Any]:
        """Converte para dicionário serializável."""
        return {
            "id": str(self.id),
            "name": self.name,
            "text": self.text,
            "metadata": self.metadata.model_dump(),
            "section_count": len(self.sections),
            "footnote_count": len(self.footnotes),
            "has_code": self.has_code,
            "created_at": self.created_at.isoformat(),
        }
    
    @classmethod
    def from_file(
        cls,
        file_path: str | Path,
        text: str,
        **kwargs: Any
    ) -> Document:
        """
        Cria documento a partir de um arquivo.
        
        Args:
            file_path: Caminho do arquivo
            text: Texto extraído
            **kwargs: Metadados adicionais
            
        Returns:
            Instância de Document
        """
        path = Path(file_path)
        
        metadata = DocumentMetadata(
            file_name=path.name,
            file_path=str(path.absolute()),
            file_size_bytes=path.stat().st_size if path.exists() else None,
            file_type=path.suffix.lstrip(".").lower(),
            char_count=len(text),
            word_count=len(text.split()),
            **kwargs
        )
        
        return cls(text=text, metadata=metadata)
    
    def __repr__(self) -> str:
        """Representação legível do documento."""
        return (
            f"Document(id={self.id}, name='{self.name}', "
            f"sections={len(self.sections)}, chars={len(self.text)})"
        )
