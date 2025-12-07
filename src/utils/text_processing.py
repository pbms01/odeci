"""
Utilitários de processamento de texto.

Funções para limpeza, extração e manipulação de texto.
"""

from __future__ import annotations

import re
import unicodedata
from typing import NamedTuple


class CodeBlock(NamedTuple):
    """Representa um bloco de código extraído."""
    
    language: str
    code: str
    start: int
    end: int


# Padrões regex compilados
CODE_BLOCK_PATTERN = re.compile(
    r"```(\w*)\n([\s\S]*?)```",
    re.MULTILINE
)

INLINE_CODE_PATTERN = re.compile(r"`([^`]+)`")

WHITESPACE_PATTERN = re.compile(r"\s+")

URL_PATTERN = re.compile(
    r"https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+"
)

EMAIL_PATTERN = re.compile(
    r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
)

# Padrões para limpeza
CONTROL_CHARS_PATTERN = re.compile(
    r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]"
)


def clean_text(
    text: str,
    remove_urls: bool = False,
    remove_emails: bool = False,
    normalize_whitespace: bool = True,
    normalize_unicode: bool = True,
    lowercase: bool = False,
) -> str:
    """
    Limpa e normaliza texto.
    
    Args:
        text: Texto original.
        remove_urls: Remover URLs.
        remove_emails: Remover emails.
        normalize_whitespace: Normalizar espaços em branco.
        normalize_unicode: Normalizar caracteres Unicode.
        lowercase: Converter para minúsculas.
        
    Returns:
        Texto limpo.
    """
    if not text:
        return ""
    
    result = text
    
    # Remover caracteres de controle
    result = CONTROL_CHARS_PATTERN.sub("", result)
    
    # Normalizar Unicode (NFC = Canonical Decomposition, then Canonical Composition)
    if normalize_unicode:
        result = unicodedata.normalize("NFC", result)
    
    # Remover URLs
    if remove_urls:
        result = URL_PATTERN.sub("", result)
    
    # Remover emails
    if remove_emails:
        result = EMAIL_PATTERN.sub("", result)
    
    # Normalizar whitespace
    if normalize_whitespace:
        # Preservar quebras de parágrafo (2+ newlines)
        result = re.sub(r"\n{3,}", "\n\n", result)
        # Normalizar espaços horizontais
        result = re.sub(r"[ \t]+", " ", result)
        # Remover espaços no início/fim de linhas
        result = re.sub(r"^ +| +$", "", result, flags=re.MULTILINE)
    
    # Lowercase
    if lowercase:
        result = result.lower()
    
    return result.strip()


def extract_code_blocks(text: str) -> tuple[str, list[CodeBlock]]:
    """
    Extrai blocos de código do texto.
    
    Args:
        text: Texto com blocos de código markdown.
        
    Returns:
        Tuple (texto sem código, lista de CodeBlock).
    """
    blocks = []
    
    for match in CODE_BLOCK_PATTERN.finditer(text):
        language = match.group(1) or "unknown"
        code = match.group(2).strip()
        blocks.append(CodeBlock(
            language=language,
            code=code,
            start=match.start(),
            end=match.end(),
        ))
    
    # Remover blocos do texto
    text_without_code = CODE_BLOCK_PATTERN.sub("", text)
    
    return text_without_code.strip(), blocks


def count_tokens(
    text: str,
    encoding: str = "cl100k_base"
) -> int:
    """
    Conta tokens usando tiktoken.
    
    Args:
        text: Texto para contar.
        encoding: Nome do encoding tiktoken.
        
    Returns:
        Número de tokens.
    """
    try:
        import tiktoken
        enc = tiktoken.get_encoding(encoding)
        return len(enc.encode(text))
    except ImportError:
        # Fallback: estimativa de 4 caracteres por token
        return len(text) // 4


def truncate_text(
    text: str,
    max_tokens: int,
    encoding: str = "cl100k_base",
    suffix: str = "..."
) -> str:
    """
    Trunca texto para máximo de tokens.
    
    Args:
        text: Texto original.
        max_tokens: Máximo de tokens.
        encoding: Nome do encoding tiktoken.
        suffix: Sufixo a adicionar se truncado.
        
    Returns:
        Texto truncado.
    """
    try:
        import tiktoken
        enc = tiktoken.get_encoding(encoding)
        tokens = enc.encode(text)
        
        if len(tokens) <= max_tokens:
            return text
        
        # Reservar espaço para sufixo
        suffix_tokens = enc.encode(suffix)
        truncate_at = max_tokens - len(suffix_tokens)
        
        truncated_tokens = tokens[:truncate_at]
        return enc.decode(truncated_tokens) + suffix
        
    except ImportError:
        # Fallback: estimativa de 4 caracteres por token
        max_chars = max_tokens * 4
        if len(text) <= max_chars:
            return text
        return text[:max_chars - len(suffix)] + suffix


def split_sentences(text: str) -> list[str]:
    """
    Divide texto em sentenças.
    
    Suporta português e inglês.
    
    Args:
        text: Texto para dividir.
        
    Returns:
        Lista de sentenças.
    """
    # Padrão simples para divisão de sentenças
    # Preserva abreviações comuns
    abbreviations = r"(?<!Dr)(?<!Sr)(?<!Sra)(?<!Art)(?<!Inc)(?<!Ltd)(?<!etc)"
    pattern = rf"{abbreviations}[.!?]+\s+"
    
    sentences = re.split(pattern, text)
    
    # Limpar e filtrar vazios
    return [s.strip() for s in sentences if s.strip()]


def normalize_legal_citations(text: str) -> str:
    """
    Normaliza citações legais no texto.
    
    Padroniza formatos de artigos, leis, etc.
    
    Args:
        text: Texto com citações.
        
    Returns:
        Texto com citações normalizadas.
    """
    # Art. / art. → Artigo
    text = re.sub(r"\bArt\.?\s*", "Artigo ", text, flags=re.IGNORECASE)
    
    # § → Parágrafo
    text = re.sub(r"§\s*", "Parágrafo ", text)
    
    # Inc. → Inciso
    text = re.sub(r"\bInc\.?\s*", "Inciso ", text, flags=re.IGNORECASE)
    
    return text


def extract_entities_simple(text: str) -> dict[str, list[str]]:
    """
    Extração simples de entidades nomeadas.
    
    Detecta padrões comuns sem NER completo.
    
    Args:
        text: Texto para análise.
        
    Returns:
        Dicionário com entidades por tipo.
    """
    entities: dict[str, list[str]] = {
        "cases": [],
        "laws": [],
        "organizations": [],
        "technologies": [],
    }
    
    # Casos jurídicos (The DAO, bZx, etc.)
    case_pattern = r"\b(The DAO|DAO|bZx|Multichain|Uniswap|Tornado Cash)\b"
    entities["cases"] = list(set(re.findall(case_pattern, text, re.IGNORECASE)))
    
    # Leis e artigos
    law_pattern = r"(Lei\s+(?:n[°º]?\s*)?\d+[\d./]*|Artigo\s+\d+)"
    entities["laws"] = list(set(re.findall(law_pattern, text, re.IGNORECASE)))
    
    # Organizações
    org_pattern = r"\b(SEC|CFTC|BCE|Ethereum Foundation|Bitcoin Foundation)\b"
    entities["organizations"] = list(set(re.findall(org_pattern, text)))
    
    # Tecnologias
    tech_pattern = r"\b(Ethereum|Bitcoin|Solidity|ERC-20|ERC-721|DeFi|NFT|DAO|Smart Contract)\b"
    entities["technologies"] = list(set(re.findall(tech_pattern, text, re.IGNORECASE)))
    
    return entities


def calculate_text_statistics(text: str) -> dict[str, int | float]:
    """
    Calcula estatísticas do texto.
    
    Args:
        text: Texto para análise.
        
    Returns:
        Dicionário com estatísticas.
    """
    words = text.split()
    sentences = split_sentences(text)
    
    return {
        "char_count": len(text),
        "word_count": len(words),
        "sentence_count": len(sentences),
        "avg_word_length": sum(len(w) for w in words) / max(len(words), 1),
        "avg_sentence_length": len(words) / max(len(sentences), 1),
        "token_count": count_tokens(text),
    }
