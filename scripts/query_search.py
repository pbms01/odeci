#!/usr/bin/env python3
"""
Script de busca semântica.

Realiza buscas em coleções do vector store.

Uso:
    python scripts/query_search.py -c minha_colecao -q "O que é o caso The DAO?"
    python scripts/query_search.py -c minha_colecao --interactive
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import typer
from rich import print as rprint
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text

# Adicionar src ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.main import ODECIPipeline
from src.retrieval.retriever import RetrievalResult
from src.utils.logging_config import setup_logging

app = typer.Typer(
    name="odeci-search",
    help="Busca semântica no ODECI",
    add_completion=False,
)

console = Console()


def display_results(
    results: list[RetrievalResult],
    show_parent: bool = False,
    max_length: int = 500
) -> None:
    """Exibe resultados formatados."""
    if not results:
        console.print("[yellow]Nenhum resultado encontrado[/yellow]")
        return
    
    for i, result in enumerate(results, 1):
        # Score
        score_text = f"Score: {result.final_score:.4f}"
        if result.rerank_score is not None:
            score_text += f" (rerank: {result.rerank_score:.4f})"
        
        # Metadados
        domain = result.metadata.get("domain", "general")
        section = result.metadata.get("section", "")
        level = result.metadata.get("level", "child")
        
        # Truncar texto se necessário
        text = result.text
        if len(text) > max_length:
            text = text[:max_length] + "..."
        
        # Header do resultado
        header = f"[bold cyan]#{i}[/bold cyan] [{domain}] {section}"
        if level != "child":
            header += f" ({level})"
        
        console.print(header)
        console.print(f"[dim]{score_text}[/dim]")
        console.print()
        
        # Texto do resultado
        console.print(Panel(
            text,
            border_style="dim",
            padding=(0, 1),
        ))
        
        # Contexto do parent (se solicitado)
        if show_parent and result.parent_text:
            console.print("[dim]Contexto (parent):[/dim]")
            parent_preview = result.parent_text[:300] + "..." if len(result.parent_text) > 300 else result.parent_text
            console.print(f"[dim italic]{parent_preview}[/dim italic]")
        
        console.print()


@app.command()
def search(
    collection: str = typer.Option(
        ...,
        "--collection", "-c",
        help="Nome da coleção para buscar",
    ),
    query: Optional[str] = typer.Option(
        None,
        "--query", "-q",
        help="Texto da busca",
    ),
    top_k: int = typer.Option(
        5,
        "--top-k", "-k",
        help="Número de resultados",
        min=1,
        max=50,
    ),
    rerank: bool = typer.Option(
        True,
        "--rerank/--no-rerank",
        help="Aplicar reranking (requer Cohere)",
    ),
    show_parent: bool = typer.Option(
        False,
        "--parent", "-p",
        help="Mostrar contexto do parent chunk",
    ),
    domain: Optional[str] = typer.Option(
        None,
        "--domain", "-d",
        help="Filtrar por domínio (legal, code, tech, general)",
    ),
    interactive: bool = typer.Option(
        False,
        "--interactive", "-i",
        help="Modo interativo",
    ),
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        help="Caminho para arquivo de configuração",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
        help="Modo verbose",
    ),
) -> None:
    """
    Realiza busca semântica em uma coleção.
    """
    # Configurar logging
    log_level = "DEBUG" if verbose else "WARNING"
    setup_logging(level=log_level)
    
    # Validar argumentos
    if not interactive and not query:
        console.print("[red]❌ Forneça uma query com --query ou use --interactive[/red]")
        raise typer.Exit(code=1)
    
    # Header
    console.print()
    console.print(Panel.fit(
        "[bold blue]ODECI[/bold blue] - Busca Semântica",
        border_style="blue",
    ))
    console.print()
    
    try:
        # Inicializar pipeline
        console.print("[dim]Inicializando...[/dim]")
        config_path = str(config) if config else None
        pipeline = ODECIPipeline(config_path=config_path)
        
        # Verificar coleção
        stats = pipeline.get_collection_stats(collection)
        if not stats["exists"]:
            console.print(f"[red]❌ Coleção '{collection}' não existe[/red]")
            raise typer.Exit(code=1)
        
        console.print(f"[dim]Coleção: {collection} ({stats['vector_count']} vetores)[/dim]")
        console.print()
        
        # Filtros
        filter_metadata = {}
        if domain:
            filter_metadata["domain"] = domain
        
        if interactive:
            # Modo interativo
            console.print("[bold]Modo interativo[/bold] (digite 'sair' para encerrar)")
            console.print()
            
            while True:
                try:
                    user_query = Prompt.ask("[bold cyan]Busca[/bold cyan]")
                    
                    if user_query.lower() in ["sair", "exit", "quit", "q"]:
                        console.print("[dim]Encerrando...[/dim]")
                        break
                    
                    if not user_query.strip():
                        continue
                    
                    # Executar busca
                    results = pipeline.search(
                        query=user_query,
                        collection_name=collection,
                        top_k=top_k,
                        rerank=rerank,
                        include_parent=show_parent,
                        filter_metadata=filter_metadata if filter_metadata else None,
                    )
                    
                    console.print()
                    display_results(results, show_parent=show_parent)
                    
                except KeyboardInterrupt:
                    console.print("\n[dim]Encerrando...[/dim]")
                    break
        else:
            # Busca única
            console.print(f"[bold]Query:[/bold] {query}")
            console.print()
            
            results = pipeline.search(
                query=query,
                collection_name=collection,
                top_k=top_k,
                rerank=rerank,
                include_parent=show_parent,
                filter_metadata=filter_metadata if filter_metadata else None,
            )
            
            display_results(results, show_parent=show_parent)
        
    except Exception as e:
        console.print(f"[red]❌ Erro: {e}[/red]")
        if verbose:
            console.print_exception()
        raise typer.Exit(code=1)


@app.command()
def similar(
    collection: str = typer.Option(
        ...,
        "--collection", "-c",
        help="Nome da coleção",
    ),
    chunk_id: str = typer.Option(
        ...,
        "--id",
        help="ID do chunk de referência",
    ),
    top_k: int = typer.Option(
        5,
        "--top-k", "-k",
        help="Número de resultados",
    ),
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        help="Caminho para arquivo de configuração",
    ),
) -> None:
    """
    Encontra chunks similares a um chunk específico.
    """
    setup_logging(level="WARNING")
    
    try:
        config_path = str(config) if config else None
        pipeline = ODECIPipeline(config_path=config_path)
        
        results = pipeline._retriever.search_similar(
            chunk_id=chunk_id,
            collection=collection,
            top_k=top_k,
        )
        
        console.print()
        console.print(f"[bold]Chunks similares a:[/bold] {chunk_id}")
        console.print()
        
        display_results(results)
        
    except Exception as e:
        console.print(f"[red]❌ Erro: {e}[/red]")
        raise typer.Exit(code=1)


@app.command()
def list_collections(
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        help="Caminho para arquivo de configuração",
    ),
) -> None:
    """
    Lista coleções disponíveis.
    """
    setup_logging(level="WARNING")
    
    try:
        config_path = str(config) if config else None
        pipeline = ODECIPipeline(config_path=config_path)
        
        # Qdrant
        if hasattr(pipeline._vector_store, '_client'):
            collections = pipeline._vector_store._client.get_collections().collections
            
            if not collections:
                console.print("[yellow]Nenhuma coleção encontrada[/yellow]")
                return
            
            table = Table(title="Coleções Disponíveis", show_header=True)
            table.add_column("Nome", style="cyan")
            table.add_column("Vetores", style="green")
            
            for col in collections:
                stats = pipeline.get_collection_stats(col.name)
                table.add_row(col.name, str(stats.get("vector_count", "?")))
            
            console.print()
            console.print(table)
            console.print()
        else:
            console.print("[yellow]Listagem não suportada para este backend[/yellow]")
        
    except Exception as e:
        console.print(f"[red]❌ Erro: {e}[/red]")
        raise typer.Exit(code=1)


def main() -> None:
    """Entry point."""
    app()


if __name__ == "__main__":
    main()
