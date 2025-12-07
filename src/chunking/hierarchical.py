"""
Chunker hierárquico para documentos jurídico-tecnológicos.

Implementa estratégia de chunking em três níveis:
- Parent: Contexto completo para reranking
- Child: Unidade principal de retrieval
- Atomic: Definições e dados pontuais
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING
from uuid import uuid4

import tiktoken

if TYPE_CHECKING:
    from src.config import ChunkingConfig

from src.chunking.base import BaseChunker
from src.chunking.domain_classifier import DomainClassifier
from src.models.chunk import (
    Chunk,
    ChunkCollection,
    ChunkLevel,
    ChunkMetadata,
    Domain,
)
from src.models.document import Document, DocumentSection

logger = logging.getLogger(__name__)


class HierarchicalChunker(BaseChunker):
    """
    Chunker hierárquico com três níveis de granularidade.
    
    Estratégia:
    1. Divide documento em PARENT chunks (seções/capítulos)
    2. Subdivide em CHILD chunks (unidades semânticas)
    3. Extrai ATOMIC chunks (definições, dados)
    
    Preserva:
    - Estrutura de headings
    - Blocos de código
    - Referências anafóricas (via overlap)
    
    Attributes:
        config: Configuração de tamanhos e overlap.
        classifier: Classificador de domínio.
        tokenizer: Tokenizador para contagem de tokens.
    """
    
    # Padrões para detecção de estrutura
    HEADING_PATTERN = re.compile(
        r"^(#{1,6})\s+(.+)$|"                    # Markdown
        r"^(\d+\.)+\s+(.+)$|"                    # Numeração
        r"^(CAPÍTULO|SEÇÃO|ARTIGO|ART\.)\s+",   # Jurídico
        re.MULTILINE | re.IGNORECASE
    )
    
    CODE_BLOCK_PATTERN = re.compile(
        r"```[\w]*\n[\s\S]*?```",
        re.MULTILINE
    )
    
    DEFINITION_PATTERN = re.compile(
        r"^[A-Z][^.!?]*:\s*[^.!?]+[.!?]$|"      # "Termo: definição."
        r"^•\s*[A-Z][^.!?]*:\s*|"                # "• Termo:"
        r"^\([a-z]\)\s*[A-Z]",                   # "(a) Definição"
        re.MULTILINE
    )
    
    def __init__(
        self,
        config: ChunkingConfig | None = None,
        classifier: DomainClassifier | None = None
    ) -> None:
        """
        Inicializa o chunker.
        
        Args:
            config: Configuração de chunking.
            classifier: Classificador de domínio.
        """
        self.config = config
        self.classifier = classifier or DomainClassifier()
        
        # Tokenizador (cl100k_base é usado pelo GPT-4 e modelos similares)
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception:
            logger.warning("Tiktoken não disponível, usando estimativa simples")
            self.tokenizer = None
        
        # Tamanhos default
        self.parent_size = 2000 if config is None else config.sizes.parent
        self.child_size = 400 if config is None else config.sizes.child
        self.atomic_size = 100 if config is None else config.sizes.atomic
        
        # Overlap
        self.parent_overlap = 200 if config is None else config.overlap.parent
        self.child_overlap = 80 if config is None else config.overlap.child
        self.atomic_overlap = 20 if config is None else config.overlap.atomic
        
        # Separadores
        self.separators = (
            ["\n\n\n", "\n\n", "\n", ". ", "; ", ", "]
            if config is None else config.separators
        )
    
    def estimate_tokens(self, text: str) -> int:
        """
        Estima número de tokens em um texto.
        
        Args:
            text: Texto para estimativa.
            
        Returns:
            Número estimado de tokens.
        """
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        
        # Estimativa simples: ~4 caracteres por token
        return len(text) // 4
    
    def split_by_tokens(
        self,
        text: str,
        max_tokens: int,
        overlap_tokens: int = 0
    ) -> list[str]:
        """
        Divide texto respeitando limite de tokens.
        
        Tenta cortar em separadores naturais.
        
        Args:
            text: Texto a dividir.
            max_tokens: Máximo de tokens por chunk.
            overlap_tokens: Tokens de sobreposição.
            
        Returns:
            Lista de chunks de texto.
        """
        if self.estimate_tokens(text) <= max_tokens:
            return [text]
        
        chunks = []
        current_text = text
        
        while current_text:
            # Estimar onde cortar
            approx_chars = max_tokens * 4
            
            if len(current_text) <= approx_chars:
                chunks.append(current_text.strip())
                break
            
            # Tentar cortar em separador natural
            cut_point = self._find_best_cut_point(
                current_text,
                approx_chars
            )
            
            chunk_text = current_text[:cut_point].strip()
            if chunk_text:
                chunks.append(chunk_text)
            
            # Aplicar overlap
            overlap_chars = overlap_tokens * 4
            next_start = max(0, cut_point - overlap_chars)
            current_text = current_text[next_start:].strip()
        
        return chunks
    
    def _find_best_cut_point(
        self,
        text: str,
        target_pos: int
    ) -> int:
        """
        Encontra melhor ponto de corte próximo à posição alvo.
        
        Args:
            text: Texto completo.
            target_pos: Posição alvo aproximada.
            
        Returns:
            Posição de corte.
        """
        # Janela de busca
        search_start = max(0, target_pos - 200)
        search_end = min(len(text), target_pos + 200)
        search_window = text[search_start:search_end]
        
        best_pos = target_pos
        
        # Tentar cada separador em ordem de prioridade
        for separator in self.separators:
            # Buscar última ocorrência antes do alvo
            pos = search_window.rfind(separator)
            if pos != -1:
                best_pos = search_start + pos + len(separator)
                break
        
        return best_pos
    
    def _preserve_code_blocks(self, text: str) -> tuple[str, dict[str, str]]:
        """
        Remove blocos de código e substitui por placeholders.
        
        Args:
            text: Texto original.
            
        Returns:
            Tuple (texto com placeholders, mapeamento placeholder->código)
        """
        placeholders = {}
        
        def replace_block(match: re.Match) -> str:
            placeholder = f"__CODE_BLOCK_{len(placeholders)}__"
            placeholders[placeholder] = match.group(0)
            return placeholder
        
        processed_text = self.CODE_BLOCK_PATTERN.sub(replace_block, text)
        
        return processed_text, placeholders
    
    def _restore_code_blocks(
        self,
        text: str,
        placeholders: dict[str, str]
    ) -> str:
        """
        Restaura blocos de código dos placeholders.
        
        Args:
            text: Texto com placeholders.
            placeholders: Mapeamento placeholder->código.
            
        Returns:
            Texto com código restaurado.
        """
        result = text
        for placeholder, code in placeholders.items():
            result = result.replace(placeholder, code)
        return result
    
    def _create_chunk(
        self,
        text: str,
        level: ChunkLevel,
        document: Document,
        parent_id: str | None = None,
        section: str | None = None,
        chapter: str | None = None
    ) -> Chunk:
        """
        Cria um chunk com metadados apropriados.
        
        Args:
            text: Conteúdo do chunk.
            level: Nível hierárquico.
            document: Documento de origem.
            parent_id: ID do chunk pai.
            section: Nome da seção.
            chapter: Nome do capítulo.
            
        Returns:
            Instância de Chunk.
        """
        # Classificar domínio
        classification = self.classifier.classify(text)
        
        # Detectar se tem código
        has_code = bool(self.CODE_BLOCK_PATTERN.search(text))
        
        # Contagens
        token_count = self.estimate_tokens(text)
        word_count = len(text.split())
        char_count = len(text)
        
        metadata = ChunkMetadata(
            document_id=document.id,
            document_name=document.name,
            section=section,
            chapter=chapter,
            domain=classification.domain,
            has_code=has_code,
            programming_language="solidity" if has_code else None,
            token_count=token_count,
            word_count=word_count,
            char_count=char_count,
            language=document.metadata.language,
        )
        
        return Chunk(
            text=text,
            level=level,
            parent_id=parent_id,
            metadata=metadata,
        )
    
    def _extract_atomic_chunks(
        self,
        text: str,
        document: Document,
        parent_id: str,
        section: str | None = None
    ) -> list[Chunk]:
        """
        Extrai chunks atômicos (definições, dados pontuais).
        
        Args:
            text: Texto para extração.
            document: Documento de origem.
            parent_id: ID do chunk pai.
            section: Nome da seção.
            
        Returns:
            Lista de chunks atômicos.
        """
        atomic_chunks = []
        
        # Buscar definições
        for match in self.DEFINITION_PATTERN.finditer(text):
            definition_text = match.group(0).strip()
            
            if self.estimate_tokens(definition_text) <= self.atomic_size:
                chunk = self._create_chunk(
                    text=definition_text,
                    level=ChunkLevel.ATOMIC,
                    document=document,
                    parent_id=parent_id,
                    section=section,
                )
                chunk.metadata.content_type = "definition"
                atomic_chunks.append(chunk)
        
        return atomic_chunks
    
    def _chunk_section(
        self,
        section: DocumentSection,
        document: Document,
        parent_chunk: Chunk | None = None
    ) -> tuple[list[Chunk], list[Chunk], list[Chunk]]:
        """
        Processa uma seção do documento.
        
        Args:
            section: Seção a processar.
            document: Documento de origem.
            parent_chunk: Chunk pai se existir.
            
        Returns:
            Tuple (parent_chunks, child_chunks, atomic_chunks).
        """
        parent_chunks = []
        child_chunks = []
        atomic_chunks = []
        
        # Texto da seção
        full_text = f"{section.title}\n\n{section.content}"
        
        # Preservar blocos de código
        text_without_code, code_placeholders = self._preserve_code_blocks(
            full_text
        )
        
        # Criar parent chunk
        parent = self._create_chunk(
            text=full_text,
            level=ChunkLevel.PARENT,
            document=document,
            section=section.title,
        )
        parent_chunks.append(parent)
        
        # Dividir em child chunks
        child_texts = self.split_by_tokens(
            text_without_code,
            self.child_size,
            self.child_overlap
        )
        
        for child_text in child_texts:
            # Restaurar código
            restored_text = self._restore_code_blocks(
                child_text,
                code_placeholders
            )
            
            child = self._create_chunk(
                text=restored_text,
                level=ChunkLevel.CHILD,
                document=document,
                parent_id=str(parent.id),
                section=section.title,
            )
            child_chunks.append(child)
            parent.children_ids.append(child.id)
            
            # Extrair atomic chunks
            atomics = self._extract_atomic_chunks(
                restored_text,
                document,
                str(child.id),
                section.title
            )
            atomic_chunks.extend(atomics)
        
        # Processar subseções recursivamente
        for subsection in section.subsections:
            sub_parents, sub_children, sub_atomics = self._chunk_section(
                subsection,
                document,
                parent
            )
            parent_chunks.extend(sub_parents)
            child_chunks.extend(sub_children)
            atomic_chunks.extend(sub_atomics)
        
        return parent_chunks, child_chunks, atomic_chunks
    
    def chunk(self, document: Document) -> ChunkCollection:
        """
        Processa documento e gera coleção de chunks.
        
        Args:
            document: Documento a ser processado.
            
        Returns:
            Coleção de chunks organizados hierarquicamente.
        """
        logger.info(f"Iniciando chunking de: {document.name}")
        
        parent_chunks: list[Chunk] = []
        child_chunks: list[Chunk] = []
        atomic_chunks: list[Chunk] = []
        
        # Se documento tem estrutura de seções
        if document.has_structure:
            logger.debug(f"Processando {len(document.sections)} seções")
            
            for section in document.sections:
                parents, children, atomics = self._chunk_section(
                    section,
                    document
                )
                parent_chunks.extend(parents)
                child_chunks.extend(children)
                atomic_chunks.extend(atomics)
        
        else:
            # Documento sem estrutura: dividir texto diretamente
            logger.debug("Documento sem estrutura, dividindo texto")
            
            text_without_code, code_placeholders = self._preserve_code_blocks(
                document.text
            )
            
            # Criar parents
            parent_texts = self.split_by_tokens(
                text_without_code,
                self.parent_size,
                self.parent_overlap
            )
            
            for i, parent_text in enumerate(parent_texts):
                restored_parent = self._restore_code_blocks(
                    parent_text,
                    code_placeholders
                )
                
                parent = self._create_chunk(
                    text=restored_parent,
                    level=ChunkLevel.PARENT,
                    document=document,
                    section=f"Parte {i + 1}",
                )
                parent_chunks.append(parent)
                
                # Criar children
                child_texts = self.split_by_tokens(
                    parent_text,
                    self.child_size,
                    self.child_overlap
                )
                
                for child_text in child_texts:
                    restored_child = self._restore_code_blocks(
                        child_text,
                        code_placeholders
                    )
                    
                    child = self._create_chunk(
                        text=restored_child,
                        level=ChunkLevel.CHILD,
                        document=document,
                        parent_id=str(parent.id),
                        section=f"Parte {i + 1}",
                    )
                    child_chunks.append(child)
                    parent.children_ids.append(child.id)
                    
                    # Extrair atomics
                    atomics = self._extract_atomic_chunks(
                        restored_child,
                        document,
                        str(child.id)
                    )
                    atomic_chunks.extend(atomics)
        
        # Criar coleção
        collection = ChunkCollection(
            document_id=document.id,
            document_name=document.name,
            parent_chunks=parent_chunks,
            child_chunks=child_chunks,
            atomic_chunks=atomic_chunks,
        )
        
        logger.info(
            f"Chunking concluído: {collection.total_chunks} chunks "
            f"(parent: {len(parent_chunks)}, child: {len(child_chunks)}, "
            f"atomic: {len(atomic_chunks)})"
        )
        logger.info(f"Distribuição por domínio: {collection.chunks_by_domain}")
        
        return collection
