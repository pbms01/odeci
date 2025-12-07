"""
Configuração de logging para o ODECI.

Fornece logging estruturado com suporte a arquivo e console.
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.config import LoggingConfig

# Formatadores
CONSOLE_FORMAT = "%(asctime)s | %(levelname)-8s | %(message)s"
FILE_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Cores ANSI para console
COLORS = {
    "DEBUG": "\033[36m",     # Cyan
    "INFO": "\033[32m",      # Green
    "WARNING": "\033[33m",   # Yellow
    "ERROR": "\033[31m",     # Red
    "CRITICAL": "\033[35m",  # Magenta
    "RESET": "\033[0m",      # Reset
}


class ColoredFormatter(logging.Formatter):
    """Formatter com cores para console."""
    
    def format(self, record: logging.LogRecord) -> str:
        # Adicionar cor ao nível
        levelname = record.levelname
        if levelname in COLORS:
            record.levelname = (
                f"{COLORS[levelname]}{levelname}{COLORS['RESET']}"
            )
        
        return super().format(record)


def setup_logging(
    level: str = "INFO",
    log_file: str | Path | None = None,
    config: LoggingConfig | None = None,
) -> None:
    """
    Configura logging global do ODECI.
    
    Args:
        level: Nível de log (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file: Caminho para arquivo de log.
        config: Configuração completa de logging.
    """
    # Usar config se fornecida
    if config:
        level = config.level
        log_file = config.file
    
    # Converter nível
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    # Logger raiz
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Limpar handlers existentes
    root_logger.handlers.clear()
    
    # Handler de console com cores
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(ColoredFormatter(
        fmt=CONSOLE_FORMAT,
        datefmt=DATE_FORMAT,
    ))
    root_logger.addHandler(console_handler)
    
    # Handler de arquivo (se especificado)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = RotatingFileHandler(
            filename=log_path,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(logging.Formatter(
            fmt=FILE_FORMAT,
            datefmt=DATE_FORMAT,
        ))
        root_logger.addHandler(file_handler)
    
    # Configurar loggers de bibliotecas para WARNING
    for lib_logger in ["httpx", "httpcore", "urllib3", "asyncio"]:
        logging.getLogger(lib_logger).setLevel(logging.WARNING)
    
    logging.info(f"Logging configurado: level={level}")


def get_logger(name: str) -> logging.Logger:
    """
    Obtém logger para um módulo.
    
    Args:
        name: Nome do módulo (use __name__).
        
    Returns:
        Logger configurado.
    """
    return logging.getLogger(name)


class LogContext:
    """Context manager para logging contextual."""
    
    def __init__(
        self,
        logger: logging.Logger,
        operation: str,
        level: int = logging.INFO
    ) -> None:
        """
        Inicializa contexto de log.
        
        Args:
            logger: Logger a usar.
            operation: Nome da operação.
            level: Nível de log.
        """
        self._logger = logger
        self._operation = operation
        self._level = level
    
    def __enter__(self) -> "LogContext":
        self._logger.log(self._level, f"Iniciando: {self._operation}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if exc_type:
            self._logger.error(
                f"Erro em {self._operation}: {exc_val}",
                exc_info=True
            )
        else:
            self._logger.log(self._level, f"Concluído: {self._operation}")
