"""
Classificador de domínio para chunks.

Identifica se o conteúdo é jurídico, código ou técnico geral
para roteamento ao modelo de embedding apropriado.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.config import DomainClassifierConfig

from src.models.chunk import Domain


@dataclass
class ClassificationResult:
    """Resultado da classificação de domínio."""
    
    domain: Domain
    confidence: float
    scores: dict[str, int]
    
    def __repr__(self) -> str:
        return (
            f"ClassificationResult(domain={self.domain.value}, "
            f"confidence={self.confidence:.2f})"
        )


class DomainClassifier:
    """
    Classifica texto em domínios para roteamento de embedding.
    
    Domínios:
    - LEGAL: Conteúdo jurídico (jurisdição, contratos, jurisprudência)
    - CODE: Código Solidity e smart contracts
    - TECH: Conteúdo técnico geral (blockchain, criptografia)
    - GENERAL: Conteúdo não classificado
    
    A classificação é baseada em padrões de palavras-chave e
    detecção de blocos de código.
    
    Attributes:
        config: Configuração de padrões de classificação.
    """
    
    # Padrões de código (regex)
    CODE_BLOCK_PATTERN = re.compile(
        r"```[\s\S]*?```|"           # Blocos markdown
        r"`[^`]+`|"                   # Inline code
        r"^\s*(function|contract|pragma|import)\s",  # Keywords Solidity
        re.MULTILINE
    )
    
    SOLIDITY_PATTERN = re.compile(
        r"\b(pragma\s+solidity|contract\s+\w+|function\s+\w+\s*\(|"
        r"mapping\s*\(|uint\d*|address\s+\w+|require\s*\(|emit\s+\w+)\b",
        re.IGNORECASE
    )
    
    def __init__(self, config: DomainClassifierConfig | None = None) -> None:
        """
        Inicializa o classificador.
        
        Args:
            config: Configuração de padrões. Se None, usa padrões default.
        """
        self.config = config
        
        # Padrões default se config não fornecida
        self.legal_keywords = self._get_legal_keywords()
        self.code_keywords = self._get_code_keywords()
        self.tech_keywords = self._get_tech_keywords()
        
        # Thresholds
        self.legal_min_score = 3 if config is None else config.legal_patterns.min_score
        self.code_min_score = 2 if config is None else config.code_patterns.min_score
        self.tech_min_score = 3 if config is None else config.tech_patterns.min_score
    
    def _get_legal_keywords(self) -> set[str]:
        """Retorna keywords jurídicas."""
        default = {
            # Português
            "jurisdição", "invalidade", "responsabilidade", "jurisprudência",
            "doutrina", "tribunal", "sentença", "decisão", "contrato",
            "cláusula", "artigo", "lei", "direito", "obrigação", "litígio",
            "réu", "autor", "demandante", "demandado", "competência",
            "prescrição", "decadência", "nulidade", "anulabilidade",
            "tutela", "medida cautelar", "agravo", "recurso", "apelação",
            "embargo", "mandado", "habeas", "ação", "processo", "autos",
            "petição", "contestação", "réplica", "tréplica", "parecer",
            "acórdão", "súmula", "precedente", "jurídico", "legal",
            "ilícito", "lícito", "culpa", "dolo", "negligência",
            "imprudência", "imperícia", "dano", "indenização",
            # Inglês (para textos de casos internacionais)
            "jurisdiction", "liability", "court", "statute", "plaintiff",
            "defendant", "ruling", "verdict", "appeal", "judgment",
            "partnership", "tort", "breach", "fiduciary", "damages",
        }
        
        if self.config and self.config.legal_patterns.keywords:
            return set(self.config.legal_patterns.keywords)
        
        return default
    
    def _get_code_keywords(self) -> set[str]:
        """Retorna keywords de código."""
        default = {
            "function", "contract", "pragma", "solidity", "require(",
            "emit", "mapping", "uint256", "uint", "address", "modifier",
            "event", "struct", "import", "interface", "library",
            "constructor", "public", "private", "internal", "external",
            "view", "pure", "payable", "returns", "memory", "storage",
            "calldata", "msg.sender", "msg.value", "block.timestamp",
            "keccak256", "abi.encode", "transfer(", "call(", "delegatecall",
            "selfdestruct", "revert", "assert", "erc20", "erc721",
            "balanceof", "totalsupply", "approve", "allowance",
            "transferfrom", "mint", "burn", "ownable", "safemath",
        }
        
        if self.config and self.config.code_patterns.keywords:
            return set(self.config.code_patterns.keywords)
        
        return default
    
    def _get_tech_keywords(self) -> set[str]:
        """Retorna keywords técnicas."""
        default = {
            "blockchain", "criptografia", "hash", "consenso",
            "descentralizado", "descentralização", "token", "protocolo",
            "rede", "transação", "mineração", "nó", "node", "peer",
            "p2p", "distributed", "ledger", "merkle", "árvore",
            "prova de trabalho", "proof of work", "pow",
            "prova de participação", "proof of stake", "pos",
            "validador", "validator", "staking", "gas", "gwei", "wei",
            "ethereum", "bitcoin", "btc", "eth", "ether", "satoshi",
            "wallet", "carteira", "chave privada", "chave pública",
            "assinatura digital", "sha256", "sha-256", "rsa", "ecc",
            "curva elíptica", "diffie-hellman", "criptográfico",
            "defi", "dao", "nft", "dapp", "web3", "oracle", "oráculo",
            "liquidez", "amm", "swap", "yield", "stablecoin",
            "smart contract", "contrato inteligente", "evm",
            "solidity", "bytecode", "abi", "json-rpc", "infura",
            "metamask", "hardhat", "truffle", "openzeppelin",
        }
        
        if self.config and self.config.tech_patterns.keywords:
            return set(self.config.tech_patterns.keywords)
        
        return default
    
    def _count_keyword_matches(
        self,
        text: str,
        keywords: set[str]
    ) -> int:
        """
        Conta matches de keywords no texto.
        
        Args:
            text: Texto para análise.
            keywords: Conjunto de keywords a buscar.
            
        Returns:
            Número de keywords encontradas.
        """
        text_lower = text.lower()
        count = 0
        
        for keyword in keywords:
            # Busca keyword como palavra completa ou parte
            if keyword.lower() in text_lower:
                count += 1
        
        return count
    
    def _detect_code_blocks(self, text: str) -> bool:
        """
        Detecta presença de blocos de código.
        
        Args:
            text: Texto para análise.
            
        Returns:
            True se contém código significativo.
        """
        # Verifica blocos de código markdown
        if self.CODE_BLOCK_PATTERN.search(text):
            return True
        
        # Verifica padrões Solidity específicos
        solidity_matches = self.SOLIDITY_PATTERN.findall(text)
        if len(solidity_matches) >= 2:
            return True
        
        return False
    
    def classify(self, text: str) -> ClassificationResult:
        """
        Classifica texto em domínio.
        
        Args:
            text: Texto a ser classificado.
            
        Returns:
            Resultado da classificação com domínio e confiança.
        """
        # Calcular scores
        legal_score = self._count_keyword_matches(text, self.legal_keywords)
        code_score = self._count_keyword_matches(text, self.code_keywords)
        tech_score = self._count_keyword_matches(text, self.tech_keywords)
        
        # Bonus para código se detectar blocos
        if self._detect_code_blocks(text):
            code_score += 5
        
        scores = {
            "legal": legal_score,
            "code": code_score,
            "tech": tech_score,
        }
        
        # Determinar domínio
        # Prioridade: CODE > LEGAL > TECH > GENERAL
        # (código tem prioridade pois é mais específico)
        
        if code_score >= self.code_min_score and code_score >= legal_score:
            domain = Domain.CODE
            max_score = code_score
        elif legal_score >= self.legal_min_score:
            domain = Domain.LEGAL
            max_score = legal_score
        elif tech_score >= self.tech_min_score:
            domain = Domain.TECH
            max_score = tech_score
        else:
            domain = Domain.GENERAL
            max_score = max(legal_score, code_score, tech_score, 1)
        
        # Calcular confiança (normalizada)
        total_score = legal_score + code_score + tech_score + 1
        confidence = max_score / total_score
        
        return ClassificationResult(
            domain=domain,
            confidence=min(confidence, 1.0),
            scores=scores
        )
    
    def classify_batch(
        self,
        texts: list[str]
    ) -> list[ClassificationResult]:
        """
        Classifica múltiplos textos.
        
        Args:
            texts: Lista de textos a classificar.
            
        Returns:
            Lista de resultados de classificação.
        """
        return [self.classify(text) for text in texts]
    
    def get_embedding_model(self, domain: Domain) -> str:
        """
        Retorna modelo de embedding recomendado para o domínio.
        
        Args:
            domain: Domínio classificado.
            
        Returns:
            Nome do modelo de embedding.
        """
        model_map = {
            Domain.LEGAL: "voyage-law-2",
            Domain.CODE: "voyage-code-3",
            Domain.TECH: "voyage-3-large",
            Domain.GENERAL: "voyage-3-large",
        }
        
        return model_map.get(domain, "voyage-3-large")
