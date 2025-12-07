# ODECI - Otimização de Documentos para Embedding e Consulta Inteligente

## 📋 Visão Geral

Pipeline profissional para processamento, chunking hierárquico e embedding de documentos jurídico-tecnológicos, com suporte a múltiplos modelos especializados (Voyage AI) e retrieval híbrido.

## 🏗️ Arquitetura

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         PIPELINE ODECI                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  DOCUMENTO    →   CHUNKING      →   CLASSIFICAÇÃO   →   EMBEDDING       │
│  (.docx)          HIERÁRQUICO       DE DOMÍNIO          MULTI-MODELO    │
│                                                                         │
│  ┌─────────┐     ┌───────────┐     ┌─────────────┐     ┌─────────────┐  │
│  │ python- │     │ Parent    │     │ Legal       │────▶│voyage-law-2 │  │
│  │ docx    │────▶│ Child     │────▶│ Tech        │────▶│voyage-3-lg  │  │
│  │         │     │ Atomic    │     │ Code        │────▶│voyage-code-3│  │
│  └─────────┘     └───────────┘     └─────────────┘     └─────────────┘  │
│                                                                         │
│                              ▼                                          │
│                    ┌─────────────────┐                                  │
│                    │  VECTOR STORE   │                                  │
│                    │  (Qdrant/Chroma)│                                  │
│                    └─────────────────┘                                  │
│                              ▼                                          │
│                    ┌─────────────────┐                                  │
│                    │  HYBRID SEARCH  │                                  │
│                    │  + RERANKING    │                                  │
│                    └─────────────────┘                                  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## 🚀 Instalação

### 1. Clone ou extraia o projeto

```bash
# Extraia para o diretório desejado
cd c:\Users\pbm_s\ondrive\odeci
```

### 2. Crie um ambiente virtual

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate
```

### 3. Instale as dependências

```bash
pip install -r requirements.txt
```

### 4. Configure as variáveis de ambiente

```bash
# Copie o arquivo de exemplo
cp .env.example .env

# Edite o .env com suas chaves de API
notepad .env  # Windows
```

## ⚙️ Configuração

Edite `config/settings.yaml` para personalizar:

```yaml
embedding:
  primary_model: "voyage-3-large"
  legal_model: "voyage-law-2"
  code_model: "voyage-code-3"
  dimensions: 1024
  
chunking:
  parent_size: 2000
  child_size: 400
  overlap: 100
```

## 📖 Uso

### Ingestão de Documento

```bash
python scripts/ingest_document.py --input "documento.docx" --collection "minha_colecao"
```

### Busca Semântica

```bash
python scripts/query_search.py --query "O que é o caso The DAO?" --top_k 5
```

### Uso Programático

```python
from src.main import ODECIPipeline

# Inicializar pipeline
pipeline = ODECIPipeline()

# Processar documento
pipeline.ingest_document("documento.docx", collection_name="juridico")

# Realizar busca
results = pipeline.search(
    query="Quais são os desafios jurisdicionais das DAOs?",
    top_k=5,
    rerank=True
)

for result in results:
    print(f"Score: {result.score:.4f}")
    print(f"Texto: {result.text[:200]}...")
    print("---")
```

## 📁 Estrutura do Projeto

```
odeci/
├── README.md                    # Este arquivo
├── requirements.txt             # Dependências Python
├── pyproject.toml              # Configuração do projeto
├── .env.example                # Template de variáveis de ambiente
│
├── config/
│   └── settings.yaml           # Configurações do pipeline
│
├── src/
│   ├── __init__.py
│   ├── main.py                 # Pipeline principal
│   ├── config.py               # Carregamento de configurações
│   │
│   ├── models/                 # Modelos de dados
│   │   ├── __init__.py
│   │   ├── chunk.py           # Modelo de Chunk
│   │   └── document.py        # Modelo de Documento
│   │
│   ├── chunking/               # Módulo de chunking
│   │   ├── __init__.py
│   │   ├── base.py            # Interface base
│   │   ├── hierarchical.py    # Chunking hierárquico
│   │   └── domain_classifier.py # Classificador de domínio
│   │
│   ├── embedding/              # Módulo de embedding
│   │   ├── __init__.py
│   │   ├── base.py            # Interface base
│   │   ├── voyage_embedder.py # Implementação Voyage AI
│   │   └── hybrid_embedder.py # Embedding multi-modelo
│   │
│   ├── storage/                # Armazenamento vetorial
│   │   ├── __init__.py
│   │   ├── base.py            # Interface base
│   │   └── vector_store.py    # Implementações (Qdrant, Chroma)
│   │
│   ├── retrieval/              # Busca e reranking
│   │   ├── __init__.py
│   │   ├── retriever.py       # Retrieval híbrido
│   │   └── reranker.py        # Reranking com Cohere
│   │
│   └── utils/                  # Utilitários
│       ├── __init__.py
│       ├── logging_config.py  # Configuração de logs
│       └── text_processing.py # Processamento de texto
│
├── scripts/                    # Scripts de linha de comando
│   ├── ingest_document.py     # Ingestão de documentos
│   └── query_search.py        # Busca interativa
│
└── tests/                      # Testes automatizados
    ├── __init__.py
    ├── test_chunking.py
    └── test_embedding.py
```

## 🔑 Modelos de Embedding Utilizados

| Modelo | Domínio | Dimensões | Contexto |
|--------|---------|-----------|----------|
| `voyage-3-large` | Geral/Multilíngue | 2048 (1024 recomendado) | 32K tokens |
| `voyage-law-2` | Jurídico | 1024 | 32K tokens |
| `voyage-code-3` | Código/Solidity | 2048 | 32K tokens |

## 📊 Métricas e Logging

O pipeline gera logs detalhados em `logs/odeci.log`:

```
2024-12-07 14:30:15 | INFO | Processando documento: O_Direito_na_Era_dos_Contratos.docx
2024-12-07 14:30:16 | INFO | Chunks gerados: 142 (parent: 15, child: 89, atomic: 38)
2024-12-07 14:30:45 | INFO | Embeddings gerados: 142 (legal: 67, tech: 52, code: 23)
2024-12-07 14:30:46 | INFO | Armazenados em collection: juridico_tech
```

## 🧪 Testes

```bash
# Executar todos os testes
pytest tests/ -v

# Executar com cobertura
pytest tests/ --cov=src --cov-report=html
```

## 📝 Licença

MIT License - Veja LICENSE para detalhes.

## 🤝 Contribuições

Contribuições são bem-vindas! Por favor, abra uma issue ou pull request.

---

**Desenvolvido para processamento de documentos jurídico-tecnológicos com foco em qualidade máxima de retrieval.**
