"""
Módulo de configuração do ODECI.

Carrega configurações de variáveis de ambiente e arquivo YAML,
com validação via Pydantic.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# ==============================================================================
# Caminhos Base
# ==============================================================================

ROOT_DIR = Path(__file__).parent.parent
CONFIG_DIR = ROOT_DIR / "config"
DATA_DIR = ROOT_DIR / "data"
LOGS_DIR = ROOT_DIR / "logs"


# ==============================================================================
# Modelos de Configuração
# ==============================================================================


class EmbeddingModelsConfig(BaseModel):
    """Configuração de modelos de embedding por domínio."""
    
    general: str = "voyage-3-large"
    legal: str = "voyage-law-2"
    code: str = "voyage-code-3"
    multilingual: str = "voyage-multilingual-2"


class EmbeddingConfig(BaseModel):
    """Configuração do módulo de embedding."""
    
    provider: Literal["voyage", "openai", "local"] = "voyage"
    models: EmbeddingModelsConfig = Field(default_factory=EmbeddingModelsConfig)
    dimensions: int = Field(default=1024, ge=256, le=4096)
    output_dtype: Literal["float", "int8", "binary"] = "float"
    batch_size: int = Field(default=32, ge=1, le=128)
    timeout: int = Field(default=60, ge=10, le=300)
    max_retries: int = Field(default=3, ge=1, le=10)
    retry_delay: float = Field(default=1.0, ge=0.1, le=10.0)


class ChunkSizesConfig(BaseModel):
    """Configuração de tamanhos de chunk."""
    
    parent: int = Field(default=2000, ge=500, le=8000)
    child: int = Field(default=400, ge=100, le=2000)
    atomic: int = Field(default=100, ge=50, le=500)


class ChunkOverlapConfig(BaseModel):
    """Configuração de sobreposição de chunks."""
    
    parent: int = Field(default=200, ge=0)
    child: int = Field(default=80, ge=0)
    atomic: int = Field(default=20, ge=0)


class ChunkingConfig(BaseModel):
    """Configuração do módulo de chunking."""
    
    strategy: Literal["hierarchical", "semantic", "fixed"] = "hierarchical"
    sizes: ChunkSizesConfig = Field(default_factory=ChunkSizesConfig)
    overlap: ChunkOverlapConfig = Field(default_factory=ChunkOverlapConfig)
    separators: list[str] = Field(
        default=["\n\n\n", "\n\n", "\n", ". ", "; ", ", "]
    )
    preserve_code_blocks: bool = True
    expand_footnotes: bool = True


class DomainPatternsConfig(BaseModel):
    """Configuração de padrões para classificação de domínio."""
    
    keywords: list[str] = Field(default_factory=list)
    min_score: int = Field(default=3, ge=1)


class DomainClassifierConfig(BaseModel):
    """Configuração do classificador de domínio."""
    
    legal_patterns: DomainPatternsConfig = Field(
        default_factory=lambda: DomainPatternsConfig(
            keywords=[
                "jurisdição", "invalidade", "responsabilidade",
                "jurisprudência", "doutrina", "tribunal", "sentença",
                "decisão", "contrato", "cláusula", "artigo", "lei"
            ],
            min_score=3
        )
    )
    code_patterns: DomainPatternsConfig = Field(
        default_factory=lambda: DomainPatternsConfig(
            keywords=[
                "function", "contract", "pragma", "solidity",
                "require(", "emit", "mapping", "uint256", "address"
            ],
            min_score=2
        )
    )
    tech_patterns: DomainPatternsConfig = Field(
        default_factory=lambda: DomainPatternsConfig(
            keywords=[
                "blockchain", "criptografia", "hash", "consenso",
                "descentralizado", "token", "protocolo"
            ],
            min_score=3
        )
    )


class QdrantConfig(BaseModel):
    """Configuração do Qdrant."""
    
    mode: Literal["local", "cloud"] = "local"
    path: str = "./data/qdrant"
    url: str | None = None
    api_key: str | None = None


class CollectionsConfig(BaseModel):
    """Configuração de coleções."""
    
    prefix: str = "odeci_"
    separate_by_domain: bool = True
    namespaces: list[str] = Field(
        default=["legal", "tech", "code", "general"]
    )


class VectorStoreConfig(BaseModel):
    """Configuração do armazenamento vetorial."""
    
    backend: Literal["qdrant", "chroma", "pinecone", "memory"] = "qdrant"
    qdrant: QdrantConfig = Field(default_factory=QdrantConfig)
    collections: CollectionsConfig = Field(default_factory=CollectionsConfig)


class RerankingConfig(BaseModel):
    """Configuração de reranking."""
    
    enabled: bool = True
    provider: Literal["cohere", "cross-encoder"] = "cohere"
    model: str = "rerank-multilingual-v3.0"
    top_n: int = Field(default=10, ge=1, le=100)


class RetrievalConfig(BaseModel):
    """Configuração do módulo de retrieval."""
    
    strategy: Literal["dense", "sparse", "hybrid"] = "hybrid"
    top_k_per_namespace: int = Field(default=20, ge=1, le=100)
    final_top_k: int = Field(default=10, ge=1, le=100)
    hybrid_alpha: float = Field(default=0.7, ge=0.0, le=1.0)
    include_parent_context: bool = True
    reranking: RerankingConfig = Field(default_factory=RerankingConfig)


class LoggingConfig(BaseModel):
    """Configuração de logging."""
    
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    format: str = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    file: str = "./logs/odeci.log"


class CacheConfig(BaseModel):
    """Configuração de cache."""
    
    enabled: bool = True
    backend: Literal["memory", "redis", "disk"] = "disk"
    ttl: int = Field(default=86400, ge=60)
    path: str = "./data/cache"
    max_size_mb: int = Field(default=1000, ge=100)


# ==============================================================================
# Configuração Principal
# ==============================================================================


class Settings(BaseSettings):
    """
    Configurações principais do ODECI.
    
    Carrega de variáveis de ambiente e arquivo YAML.
    """
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    
    # API Keys (de variáveis de ambiente)
    voyage_api_key: str | None = Field(default=None, alias="VOYAGE_API_KEY")
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    cohere_api_key: str | None = Field(default=None, alias="COHERE_API_KEY")
    qdrant_api_key: str | None = Field(default=None, alias="QDRANT_API_KEY")
    qdrant_url: str | None = Field(default=None, alias="QDRANT_URL")
    
    # Configurações de ambiente
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    data_dir: str = Field(default="./data", alias="DATA_DIR")
    environment: Literal["development", "production"] = Field(
        default="development", alias="ENVIRONMENT"
    )
    
    # Configurações carregadas do YAML
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    domain_classifier: DomainClassifierConfig = Field(
        default_factory=DomainClassifierConfig
    )
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    
    @field_validator("voyage_api_key", "openai_api_key", "cohere_api_key")
    @classmethod
    def validate_api_key(cls, v: str | None) -> str | None:
        """Valida que API keys não são placeholders."""
        if v and "your_" in v.lower():
            return None
        return v
    
    def is_voyage_configured(self) -> bool:
        """Verifica se Voyage AI está configurado."""
        return self.voyage_api_key is not None
    
    def is_openai_configured(self) -> bool:
        """Verifica se OpenAI está configurado."""
        return self.openai_api_key is not None
    
    def is_cohere_configured(self) -> bool:
        """Verifica se Cohere está configurado."""
        return self.cohere_api_key is not None


def load_yaml_config(config_path: Path | None = None) -> dict[str, Any]:
    """
    Carrega configurações do arquivo YAML.
    
    Args:
        config_path: Caminho para o arquivo YAML. Se None, usa o padrão.
        
    Returns:
        Dicionário com configurações.
    """
    if config_path is None:
        config_path = CONFIG_DIR / "settings.yaml"
    
    if not config_path.exists():
        return {}
    
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@lru_cache
def get_settings(config_path: str | None = None) -> Settings:
    """
    Obtém instância singleton das configurações.
    
    Carrega configurações de:
    1. Variáveis de ambiente
    2. Arquivo .env
    3. Arquivo settings.yaml
    
    Args:
        config_path: Caminho opcional para arquivo YAML.
        
    Returns:
        Instância de Settings configurada.
    """
    # Carregar YAML
    yaml_path = Path(config_path) if config_path else None
    yaml_config = load_yaml_config(yaml_path)
    
    # Criar settings mesclando YAML com env vars
    return Settings(**yaml_config)


def ensure_directories() -> None:
    """Garante que diretórios necessários existem."""
    directories = [
        DATA_DIR,
        DATA_DIR / "qdrant",
        DATA_DIR / "chroma",
        DATA_DIR / "cache",
        LOGS_DIR,
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


# Inicializar diretórios ao importar módulo
ensure_directories()
