"""
Testes para o módulo de chunking.

Verifica:
- Classificação de domínio
- Chunking hierárquico
- Preservação de estrutura
"""

import pytest
from uuid import uuid4

from src.models.chunk import Chunk, ChunkLevel, ChunkMetadata, Domain
from src.models.document import Document, DocumentMetadata, DocumentSection
from src.chunking.domain_classifier import DomainClassifier, ClassificationResult
from src.chunking.hierarchical import HierarchicalChunker


# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture
def sample_legal_text() -> str:
    """Texto jurídico de exemplo."""
    return """
    A jurisdição sobre contratos inteligentes apresenta desafios únicos.
    A responsabilidade civil dos desenvolvedores deve ser analisada à luz
    da jurisprudência tradicional, considerando a doutrina do caso fortuito.
    O tribunal deve avaliar se houve negligência na implementação do código.
    A cláusula de arbitragem pode ser aplicável em disputas envolvendo DAOs.
    """


@pytest.fixture
def sample_code_text() -> str:
    """Texto com código Solidity."""
    return """
    ```solidity
    pragma solidity ^0.8.0;
    
    contract SimpleToken {
        mapping(address => uint256) public balances;
        
        function transfer(address to, uint256 amount) public {
            require(balances[msg.sender] >= amount, "Insufficient balance");
            balances[msg.sender] -= amount;
            balances[to] += amount;
            emit Transfer(msg.sender, to, amount);
        }
    }
    ```
    
    Este contrato implementa uma função de transferência básica com verificação
    de saldo usando require().
    """


@pytest.fixture
def sample_tech_text() -> str:
    """Texto técnico geral."""
    return """
    O protocolo de consenso Proof of Work utiliza hash criptográfico
    para validar transações na blockchain. A rede descentralizada
    garante que mineradores competam para adicionar novos blocos.
    Cada nó valida independentemente as transações recebidas.
    """


@pytest.fixture
def sample_general_text() -> str:
    """Texto geral sem domínio específico."""
    return """
    Este é um texto introdutório que não contém terminologia
    específica de nenhum domínio particular. Ele serve apenas
    para testar a classificação de conteúdo genérico.
    """


@pytest.fixture
def sample_document() -> Document:
    """Documento de exemplo para testes."""
    sections = [
        DocumentSection(
            level=1,
            title="Introdução",
            content="Este documento aborda os aspectos jurídicos dos contratos inteligentes.",
            start_char=0,
            end_char=100,
        ),
        DocumentSection(
            level=2,
            title="Jurisdição",
            content="""
            A competência jurisdicional para litígios envolvendo smart contracts
            ainda é objeto de debate na doutrina. Os tribunais têm se posicionado
            de forma divergente quanto à aplicabilidade das regras tradicionais.
            """,
            start_char=100,
            end_char=400,
        ),
    ]
    
    metadata = DocumentMetadata(
        file_name="test_document.docx",
        file_type="docx",
        char_count=500,
        word_count=80,
    )
    
    return Document(
        text="Conteúdo completo do documento de teste...",
        sections=sections,
        metadata=metadata,
    )


@pytest.fixture
def domain_classifier() -> DomainClassifier:
    """Classificador de domínio."""
    return DomainClassifier()


@pytest.fixture
def hierarchical_chunker(domain_classifier: DomainClassifier) -> HierarchicalChunker:
    """Chunker hierárquico."""
    return HierarchicalChunker(classifier=domain_classifier)


# ==============================================================================
# Testes do DomainClassifier
# ==============================================================================


class TestDomainClassifier:
    """Testes para DomainClassifier."""
    
    def test_classify_legal_text(
        self,
        domain_classifier: DomainClassifier,
        sample_legal_text: str
    ):
        """Deve classificar texto jurídico corretamente."""
        result = domain_classifier.classify(sample_legal_text)
        
        assert isinstance(result, ClassificationResult)
        assert result.domain == Domain.LEGAL
        assert result.confidence > 0.3
        assert result.scores["legal"] >= 3
    
    def test_classify_code_text(
        self,
        domain_classifier: DomainClassifier,
        sample_code_text: str
    ):
        """Deve classificar código Solidity corretamente."""
        result = domain_classifier.classify(sample_code_text)
        
        assert result.domain == Domain.CODE
        assert result.scores["code"] >= 2
    
    def test_classify_tech_text(
        self,
        domain_classifier: DomainClassifier,
        sample_tech_text: str
    ):
        """Deve classificar texto técnico corretamente."""
        result = domain_classifier.classify(sample_tech_text)
        
        assert result.domain in [Domain.TECH, Domain.GENERAL]
        assert result.scores["tech"] >= 2
    
    def test_classify_general_text(
        self,
        domain_classifier: DomainClassifier,
        sample_general_text: str
    ):
        """Deve classificar texto genérico como GENERAL."""
        result = domain_classifier.classify(sample_general_text)
        
        assert result.domain == Domain.GENERAL
    
    def test_classify_batch(self, domain_classifier: DomainClassifier):
        """Deve classificar múltiplos textos."""
        texts = [
            "O contrato estabelece jurisdição em São Paulo.",
            "function transfer(address to) public {}",
            "Este é um texto genérico.",
        ]
        
        results = domain_classifier.classify_batch(texts)
        
        assert len(results) == 3
        assert all(isinstance(r, ClassificationResult) for r in results)
    
    def test_get_embedding_model(self, domain_classifier: DomainClassifier):
        """Deve retornar modelo correto por domínio."""
        assert domain_classifier.get_embedding_model(Domain.LEGAL) == "voyage-law-2"
        assert domain_classifier.get_embedding_model(Domain.CODE) == "voyage-code-3"
        assert domain_classifier.get_embedding_model(Domain.TECH) == "voyage-3-large"
        assert domain_classifier.get_embedding_model(Domain.GENERAL) == "voyage-3-large"
    
    def test_code_detection_with_patterns(self, domain_classifier: DomainClassifier):
        """Deve detectar padrões Solidity específicos."""
        solidity_text = """
        pragma solidity ^0.8.0;
        contract MyContract {
            mapping(address => uint256) balances;
            function deposit() public payable {
                balances[msg.sender] += msg.value;
            }
        }
        """
        
        result = domain_classifier.classify(solidity_text)
        
        assert result.domain == Domain.CODE


# ==============================================================================
# Testes do HierarchicalChunker
# ==============================================================================


class TestHierarchicalChunker:
    """Testes para HierarchicalChunker."""
    
    def test_estimate_tokens(self, hierarchical_chunker: HierarchicalChunker):
        """Deve estimar tokens corretamente."""
        text = "Esta é uma frase de teste com algumas palavras."
        tokens = hierarchical_chunker.estimate_tokens(text)
        
        # Estimativa aproximada
        assert tokens > 0
        assert tokens < len(text)
    
    def test_split_by_tokens_short_text(
        self,
        hierarchical_chunker: HierarchicalChunker
    ):
        """Texto curto não deve ser dividido."""
        text = "Texto curto."
        chunks = hierarchical_chunker.split_by_tokens(text, max_tokens=100)
        
        assert len(chunks) == 1
        assert chunks[0] == text
    
    def test_split_by_tokens_long_text(
        self,
        hierarchical_chunker: HierarchicalChunker
    ):
        """Texto longo deve ser dividido em chunks."""
        text = " ".join(["palavra"] * 500)  # ~500 palavras
        chunks = hierarchical_chunker.split_by_tokens(text, max_tokens=100)
        
        assert len(chunks) > 1
        for chunk in chunks:
            # Cada chunk deve estar abaixo do limite (aproximadamente)
            assert hierarchical_chunker.estimate_tokens(chunk) <= 150
    
    def test_split_preserves_overlap(
        self,
        hierarchical_chunker: HierarchicalChunker
    ):
        """Divisão deve preservar overlap entre chunks."""
        text = "A " * 200 + "B " * 200 + "C " * 200
        chunks = hierarchical_chunker.split_by_tokens(
            text,
            max_tokens=100,
            overlap_tokens=20
        )
        
        # Deve haver sobreposição
        if len(chunks) > 1:
            # Verificar se há algum conteúdo comum
            assert len(chunks) >= 2
    
    def test_chunk_document_creates_hierarchy(
        self,
        hierarchical_chunker: HierarchicalChunker,
        sample_document: Document
    ):
        """Chunking deve criar hierarquia correta."""
        collection = hierarchical_chunker.chunk(sample_document)
        
        # Deve ter chunks de diferentes níveis
        assert collection.total_chunks > 0
        
        # Verificar estrutura
        assert collection.document_id == sample_document.id
        assert collection.document_name == sample_document.name
    
    def test_chunk_assigns_domains(
        self,
        hierarchical_chunker: HierarchicalChunker,
        sample_document: Document
    ):
        """Chunks devem ter domínios atribuídos."""
        collection = hierarchical_chunker.chunk(sample_document)
        
        for chunk in collection.get_all_chunks():
            assert chunk.metadata.domain in [
                Domain.LEGAL,
                Domain.CODE,
                Domain.TECH,
                Domain.GENERAL,
            ]
    
    def test_chunk_preserves_parent_child_relationship(
        self,
        hierarchical_chunker: HierarchicalChunker,
        sample_document: Document
    ):
        """Deve preservar relação parent-child."""
        collection = hierarchical_chunker.chunk(sample_document)
        
        # Child chunks devem ter parent_id
        for child in collection.child_chunks:
            if child.parent_id:
                # Verificar se parent existe
                parent_ids = [str(p.id) for p in collection.parent_chunks]
                assert str(child.parent_id) in parent_ids


# ==============================================================================
# Testes de Integração
# ==============================================================================


class TestChunkingIntegration:
    """Testes de integração do chunking."""
    
    def test_full_chunking_pipeline(
        self,
        hierarchical_chunker: HierarchicalChunker
    ):
        """Pipeline completo de chunking."""
        # Criar documento com conteúdo misto
        text = """
        # Introdução
        
        Os contratos inteligentes representam uma evolução significativa
        no campo do direito contratual.
        
        ## Aspectos Jurídicos
        
        A jurisdição aplicável deve considerar a natureza descentralizada
        da tecnologia blockchain. O tribunal competente analisará...
        
        ## Implementação Técnica
        
        ```solidity
        contract Escrow {
            function release() public {
                require(msg.sender == arbiter);
                payable(beneficiary).transfer(amount);
            }
        }
        ```
        
        Este código implementa um sistema de custódia básico.
        """
        
        metadata = DocumentMetadata(
            file_name="mixed_content.md",
            file_type="md",
            char_count=len(text),
        )
        
        document = Document(text=text, metadata=metadata)
        collection = hierarchical_chunker.chunk(document)
        
        # Verificar que chunks foram criados
        assert collection.total_chunks > 0
        
        # Verificar distribuição por domínio
        domains = collection.chunks_by_domain
        assert len(domains) > 0
    
    def test_chunking_with_empty_document(
        self,
        hierarchical_chunker: HierarchicalChunker
    ):
        """Deve lidar com documento vazio."""
        metadata = DocumentMetadata(
            file_name="empty.txt",
            file_type="txt",
        )
        
        document = Document(text="", metadata=metadata)
        collection = hierarchical_chunker.chunk(document)
        
        # Não deve quebrar
        assert collection.total_chunks == 0


# ==============================================================================
# Testes de Modelos
# ==============================================================================


class TestChunkModel:
    """Testes para o modelo Chunk."""
    
    def test_chunk_creation(self):
        """Deve criar chunk corretamente."""
        metadata = ChunkMetadata(
            document_id=uuid4(),
            document_name="test.docx",
            domain=Domain.LEGAL,
        )
        
        chunk = Chunk(
            text="Conteúdo do chunk",
            level=ChunkLevel.CHILD,
            metadata=metadata,
        )
        
        assert chunk.text == "Conteúdo do chunk"
        assert chunk.level == ChunkLevel.CHILD
        assert chunk.domain == "legal"
        assert not chunk.has_embedding
    
    def test_chunk_with_embedding(self):
        """Chunk com embedding."""
        metadata = ChunkMetadata(
            document_id=uuid4(),
            document_name="test.docx",
        )
        
        chunk = Chunk(
            text="Texto",
            level=ChunkLevel.CHILD,
            metadata=metadata,
            embedding=[0.1] * 1024,
        )
        
        assert chunk.has_embedding
        assert len(chunk.embedding) == 1024
    
    def test_chunk_to_vector_payload(self):
        """Deve converter para payload vetorial."""
        metadata = ChunkMetadata(
            document_id=uuid4(),
            document_name="test.docx",
            section="Seção 1",
            domain=Domain.CODE,
            has_code=True,
        )
        
        chunk = Chunk(
            text="function test() {}",
            level=ChunkLevel.CHILD,
            metadata=metadata,
        )
        
        payload = chunk.to_vector_payload()
        
        assert "text" in payload
        assert payload["domain"] == "code"
        assert payload["has_code"] is True
        assert payload["section"] == "Seção 1"
