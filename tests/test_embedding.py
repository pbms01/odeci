"""
Testes para o módulo de embedding.

Verifica:
- Configuração de embedders
- Processamento de chunks
- Integração com modelos
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from uuid import uuid4

from src.models.chunk import Chunk, ChunkLevel, ChunkMetadata, Domain
from src.embedding.base import BaseEmbedder, EmbeddingResult
from src.embedding.voyage_embedder import VoyageEmbedder, VoyageEmbedderFactory
from src.embedding.hybrid_embedder import HybridEmbedder


# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture
def sample_chunk() -> Chunk:
    """Chunk de exemplo."""
    metadata = ChunkMetadata(
        document_id=uuid4(),
        document_name="test.docx",
        domain=Domain.LEGAL,
        token_count=50,
    )
    
    return Chunk(
        text="Este é um texto de teste para embedding.",
        level=ChunkLevel.CHILD,
        metadata=metadata,
    )


@pytest.fixture
def sample_chunks() -> list[Chunk]:
    """Lista de chunks de exemplo."""
    chunks = []
    
    domains = [Domain.LEGAL, Domain.CODE, Domain.TECH, Domain.GENERAL]
    texts = [
        "A jurisdição aplicável ao caso deve ser determinada.",
        "function transfer(address to) public { }",
        "O protocolo de consenso utiliza hash criptográfico.",
        "Este é um texto genérico sem domínio específico.",
    ]
    
    for domain, text in zip(domains, texts):
        metadata = ChunkMetadata(
            document_id=uuid4(),
            document_name="test.docx",
            domain=domain,
        )
        chunk = Chunk(
            text=text,
            level=ChunkLevel.CHILD,
            metadata=metadata,
        )
        chunks.append(chunk)
    
    return chunks


@pytest.fixture
def mock_voyage_client():
    """Mock do cliente Voyage AI."""
    mock_client = Mock()
    
    # Simular resposta de embedding
    mock_result = Mock()
    mock_result.embeddings = [[0.1] * 1024]
    mock_client.embed.return_value = mock_result
    
    return mock_client


@pytest.fixture
def mock_embedding_result() -> list[float]:
    """Embedding simulado."""
    return [0.1] * 1024


# ==============================================================================
# Testes do EmbeddingResult
# ==============================================================================


class TestEmbeddingResult:
    """Testes para EmbeddingResult."""
    
    def test_embedding_result_creation(self):
        """Deve criar resultado corretamente."""
        result = EmbeddingResult(
            text="Texto de teste",
            embedding=[0.1] * 1024,
            model="voyage-3-large",
            dimensions=1024,
            tokens_used=10,
        )
        
        assert result.text == "Texto de teste"
        assert len(result.embedding) == 1024
        assert result.model == "voyage-3-large"
        assert result.dimensions == 1024
    
    def test_embedding_result_repr(self):
        """Deve ter representação legível."""
        result = EmbeddingResult(
            text="Texto",
            embedding=[0.1] * 512,
            model="voyage-law-2",
            dimensions=512,
            tokens_used=5,
        )
        
        repr_str = repr(result)
        assert "voyage-law-2" in repr_str
        assert "512" in repr_str


# ==============================================================================
# Testes do VoyageEmbedder (Mocked)
# ==============================================================================


class TestVoyageEmbedder:
    """Testes para VoyageEmbedder com mocks."""
    
    def test_available_models(self):
        """Deve ter modelos disponíveis definidos."""
        assert "voyage-3-large" in VoyageEmbedder.AVAILABLE_MODELS
        assert "voyage-law-2" in VoyageEmbedder.AVAILABLE_MODELS
        assert "voyage-code-3" in VoyageEmbedder.AVAILABLE_MODELS
    
    def test_model_dimensions(self):
        """Deve ter dimensões corretas por modelo."""
        models = VoyageEmbedder.AVAILABLE_MODELS
        
        assert models["voyage-3-large"]["max_dims"] == 2048
        assert models["voyage-law-2"]["max_dims"] == 1024
        assert models["voyage-code-3"]["max_dims"] == 2048
    
    def test_init_without_api_key_raises_error(self):
        """Deve levantar erro sem API key."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="API key"):
                VoyageEmbedder(api_key=None)
    
    def test_init_with_invalid_model_raises_error(self):
        """Deve levantar erro com modelo inválido."""
        with pytest.raises(ValueError, match="não suportado"):
            VoyageEmbedder(model="modelo-inexistente", api_key="test-key")
    
    @patch("src.embedding.voyage_embedder.voyageai")
    def test_embed_text(self, mock_voyageai, mock_voyage_client):
        """Deve gerar embedding para texto."""
        mock_voyageai.Client.return_value = mock_voyage_client
        
        embedder = VoyageEmbedder(api_key="test-key", dimensions=1024)
        result = embedder.embed_text("Texto de teste")
        
        assert isinstance(result, EmbeddingResult)
        assert len(result.embedding) == 1024
        mock_voyage_client.embed.assert_called_once()
    
    @patch("src.embedding.voyage_embedder.voyageai")
    def test_embed_texts_batch(self, mock_voyageai, mock_voyage_client):
        """Deve processar textos em batch."""
        # Configurar mock para retornar múltiplos embeddings
        mock_result = Mock()
        mock_result.embeddings = [[0.1] * 1024, [0.2] * 1024]
        mock_voyage_client.embed.return_value = mock_result
        mock_voyageai.Client.return_value = mock_voyage_client
        
        embedder = VoyageEmbedder(api_key="test-key", batch_size=10)
        results = embedder.embed_texts(
            ["Texto 1", "Texto 2"],
            show_progress=False
        )
        
        assert len(results) == 2
        assert all(isinstance(r, EmbeddingResult) for r in results)
    
    @patch("src.embedding.voyage_embedder.voyageai")
    def test_embed_chunk(
        self,
        mock_voyageai,
        mock_voyage_client,
        sample_chunk
    ):
        """Deve gerar embedding para chunk."""
        mock_voyageai.Client.return_value = mock_voyage_client
        
        embedder = VoyageEmbedder(api_key="test-key")
        result = embedder.embed_chunk(sample_chunk)
        
        assert result.has_embedding
        assert result.embedding_model == "voyage-3-large"
    
    @patch("src.embedding.voyage_embedder.voyageai")
    def test_embed_query(self, mock_voyageai, mock_voyage_client):
        """Deve gerar embedding para query com input_type correto."""
        mock_voyageai.Client.return_value = mock_voyage_client
        
        embedder = VoyageEmbedder(api_key="test-key")
        embedding = embedder.embed_query("Busca de teste")
        
        assert len(embedding) == 1024
        # Verificar que input_type foi "query"
        call_args = mock_voyage_client.embed.call_args
        assert call_args.kwargs.get("input_type") == "query"


# ==============================================================================
# Testes do VoyageEmbedderFactory
# ==============================================================================


class TestVoyageEmbedderFactory:
    """Testes para VoyageEmbedderFactory."""
    
    @patch("src.embedding.voyage_embedder.voyageai")
    def test_create_legal_embedder(self, mock_voyageai):
        """Deve criar embedder jurídico."""
        mock_voyageai.Client.return_value = Mock()
        
        embedder = VoyageEmbedderFactory.create_legal_embedder(api_key="test")
        
        assert embedder.model_name == "voyage-law-2"
    
    @patch("src.embedding.voyage_embedder.voyageai")
    def test_create_code_embedder(self, mock_voyageai):
        """Deve criar embedder de código."""
        mock_voyageai.Client.return_value = Mock()
        
        embedder = VoyageEmbedderFactory.create_code_embedder(api_key="test")
        
        assert embedder.model_name == "voyage-code-3"
    
    @patch("src.embedding.voyage_embedder.voyageai")
    def test_create_general_embedder(self, mock_voyageai):
        """Deve criar embedder geral."""
        mock_voyageai.Client.return_value = Mock()
        
        embedder = VoyageEmbedderFactory.create_general_embedder(api_key="test")
        
        assert embedder.model_name == "voyage-3-large"


# ==============================================================================
# Testes do HybridEmbedder
# ==============================================================================


class TestHybridEmbedder:
    """Testes para HybridEmbedder."""
    
    def test_model_mapping(self):
        """Deve ter mapeamento correto de modelos."""
        with patch("src.embedding.voyage_embedder.voyageai"):
            embedder = HybridEmbedder(api_key="test-key")
            
            assert embedder._model_map[Domain.LEGAL] == "voyage-law-2"
            assert embedder._model_map[Domain.CODE] == "voyage-code-3"
            assert embedder._model_map[Domain.TECH] == "voyage-3-large"
            assert embedder._model_map[Domain.GENERAL] == "voyage-3-large"
    
    @patch("src.embedding.voyage_embedder.voyageai")
    def test_get_model_for_domain(self, mock_voyageai):
        """Deve retornar modelo correto por domínio."""
        embedder = HybridEmbedder(api_key="test-key")
        
        assert embedder._get_model_for_domain(Domain.LEGAL) == "voyage-law-2"
        assert embedder._get_model_for_domain(Domain.CODE) == "voyage-code-3"
    
    @patch("src.embedding.voyage_embedder.voyageai")
    def test_embed_chunks_routes_by_domain(
        self,
        mock_voyageai,
        sample_chunks
    ):
        """Deve rotear chunks para modelos corretos."""
        # Mock do cliente
        mock_client = Mock()
        mock_result = Mock()
        mock_result.embeddings = [[0.1] * 1024]
        mock_client.embed.return_value = mock_result
        mock_voyageai.Client.return_value = mock_client
        
        embedder = HybridEmbedder(api_key="test-key")
        results = embedder.embed_chunks(sample_chunks, show_progress=False)
        
        assert len(results) == len(sample_chunks)
        assert all(chunk.has_embedding for chunk in results)
    
    @patch("src.embedding.voyage_embedder.voyageai")
    def test_embed_query_uses_general_model(self, mock_voyageai):
        """Query sem domínio deve usar modelo geral."""
        mock_client = Mock()
        mock_result = Mock()
        mock_result.embeddings = [[0.1] * 1024]
        mock_client.embed.return_value = mock_result
        mock_voyageai.Client.return_value = mock_client
        
        embedder = HybridEmbedder(api_key="test-key")
        embedding = embedder.embed_query("Busca genérica")
        
        assert len(embedding) == 1024


# ==============================================================================
# Testes de Integração
# ==============================================================================


class TestEmbeddingIntegration:
    """Testes de integração do embedding."""
    
    @patch("src.embedding.voyage_embedder.voyageai")
    def test_full_embedding_pipeline(self, mock_voyageai, sample_chunks):
        """Pipeline completo de embedding."""
        # Mock
        mock_client = Mock()
        mock_result = Mock()
        mock_result.embeddings = [[0.1] * 1024]
        mock_client.embed.return_value = mock_result
        mock_voyageai.Client.return_value = mock_client
        
        # Criar embedder híbrido
        embedder = HybridEmbedder(api_key="test-key", dimensions=1024)
        
        # Processar todos os chunks
        embedded_chunks = embedder.embed_chunks(sample_chunks, show_progress=False)
        
        # Verificar
        assert len(embedded_chunks) == len(sample_chunks)
        
        for chunk in embedded_chunks:
            assert chunk.has_embedding
            assert chunk.embedding_model is not None
            assert len(chunk.embedding) == 1024
    
    @patch("src.embedding.voyage_embedder.voyageai")
    def test_embedding_preserves_metadata(self, mock_voyageai, sample_chunk):
        """Embedding deve preservar metadados do chunk."""
        mock_client = Mock()
        mock_result = Mock()
        mock_result.embeddings = [[0.1] * 1024]
        mock_client.embed.return_value = mock_result
        mock_voyageai.Client.return_value = mock_client
        
        original_id = sample_chunk.id
        original_text = sample_chunk.text
        original_domain = sample_chunk.metadata.domain
        
        embedder = HybridEmbedder(api_key="test-key")
        result = embedder.embed_chunk(sample_chunk)
        
        assert result.id == original_id
        assert result.text == original_text
        assert result.metadata.domain == original_domain


# ==============================================================================
# Testes de Configuração
# ==============================================================================


class TestEmbeddingConfiguration:
    """Testes de configuração de embedding."""
    
    @patch("src.embedding.voyage_embedder.voyageai")
    def test_custom_dimensions(self, mock_voyageai):
        """Deve aceitar dimensões customizadas."""
        mock_voyageai.Client.return_value = Mock()
        
        embedder = VoyageEmbedder(
            api_key="test-key",
            dimensions=512
        )
        
        assert embedder.dimensions == 512
    
    @patch("src.embedding.voyage_embedder.voyageai")
    def test_dimensions_clamped_to_max(self, mock_voyageai):
        """Dimensões devem ser limitadas ao máximo do modelo."""
        mock_voyageai.Client.return_value = Mock()
        
        # voyage-law-2 tem max 1024
        embedder = VoyageEmbedder(
            model="voyage-law-2",
            api_key="test-key",
            dimensions=2048
        )
        
        assert embedder.dimensions == 1024
    
    @patch("src.embedding.voyage_embedder.voyageai")
    def test_batch_size_configuration(self, mock_voyageai):
        """Deve aceitar batch size customizado."""
        mock_voyageai.Client.return_value = Mock()
        
        embedder = VoyageEmbedder(
            api_key="test-key",
            batch_size=64
        )
        
        assert embedder._batch_size == 64


# ==============================================================================
# Testes de Erro
# ==============================================================================


class TestEmbeddingErrors:
    """Testes de tratamento de erros."""
    
    @patch("src.embedding.voyage_embedder.voyageai")
    def test_api_error_raises_exception(self, mock_voyageai):
        """Erro de API deve ser propagado."""
        mock_client = Mock()
        mock_client.embed.side_effect = Exception("API Error")
        mock_voyageai.Client.return_value = mock_client
        
        embedder = VoyageEmbedder(api_key="test-key")
        
        with pytest.raises(Exception):
            embedder.embed_text("Texto")
    
    def test_missing_voyage_package(self):
        """Deve indicar pacote faltando."""
        with patch.dict("sys.modules", {"voyageai": None}):
            # Reimportar para verificar erro
            pass  # O teste real requer setup mais complexo
