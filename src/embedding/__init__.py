"""Módulo de embedding do ODECI."""

from src.embedding.base import BaseEmbedder
from src.embedding.voyage_embedder import VoyageEmbedder
from src.embedding.hybrid_embedder import HybridEmbedder
from src.embedding.unified_embedder import UnifiedEmbedder, create_embedder

__all__ = [
    "BaseEmbedder",
    "VoyageEmbedder",
    "HybridEmbedder",
    "UnifiedEmbedder",
    "create_embedder",
]
