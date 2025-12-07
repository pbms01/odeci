"""
ODECI - Otimização de Documentos para Embedding e Consulta Inteligente

Pipeline profissional para processamento, chunking hierárquico e embedding
de documentos jurídico-tecnológicos.
"""

__version__ = "1.0.0"
__author__ = "ODECI Team"

from src.main import ODECIPipeline
from src.config import Settings, get_settings

__all__ = [
    "ODECIPipeline",
    "Settings",
    "get_settings",
    "__version__",
]
