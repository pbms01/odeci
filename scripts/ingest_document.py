#!/usr/bin/env python3
"""
Script de ingestão de documentos.

Processa documentos e armazena no vector store.

Uso:
    python scripts/ingest_document.py --input documento.docx --collection minha_colecao
    python scripts/ingest_document.py -i documento.docx -c minha_colecao --levels child
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import typer
from rich import print as rprint
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

# Adicionar src ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.main import ODECIPipeline
from src.models.chunk import ChunkLevel
from src.utils.logging_config import setup_logging

app = typer.Typer(
    name="odeci-ingest",
    help="Ingestão de documentos para o ODECI",
    add_completion=False,
)

console = Console()


def parse_chunk_levels(levels_str: str | None) -> list[ChunkLevel] | None:
    """Converte string de níveis para lista de ChunkLevel."""
    if not levels_str:
        return None
    
    level_map = {
        "parent": ChunkLevel.PARENT,
        "child": ChunkLevel.CHILD,
        "atomic": ChunkLevel.ATOMIC,
        "all": None,
    }
    
    levels = []
    for level in levels_str.split(","):
        level = level.strip().lower()
        if level == "all":
            return None
        if level in level_map:
            levels.append(level_map[level])
    
    return levels if levels else None


@app.command()
def ingest(
    input_file: Path = typer.Option(
        ...,
        "--input", "-i",
        help="Caminho do documento a processar",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    collection: Optional[str] = typer.Option(
        None,
        "--collection", "-c",
        help="Nome da coleção (default: odeci_<nome_arquivo>)",
    ),
    levels: Optional[str] = typer.Option(
        None,
        "--levels", "-l",
        help="Níveis de chunk: parent,child,atomic ou all (default: all)",
    ),
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        help="Caminho para arquivo de configuração YAML",
        exists=True,
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
        help="Modo verbose (DEBUG)",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Simular sem armazenar",
    ),
) -> None:
    """
    Processa documento e armazena no vector store.
    
    Suporta formatos: .docx, .pdf, .txt, .md
    """
    # Configurar logging
    log_level = "DEBUG" if verbose else "INFO"
    setup_logging(level=log_level)
    
    # Header
    console.print()
    console.print(Panel.fit(
        "[bold blue]ODECI[/bold blue] - Ingestão de Documento",
        border_style="blue",
    ))
    console.print()
    
    # Informações do documento
    console.print(f"📄 Documento: [cyan]{input_file.name}[/cyan]")
    console.print(f"📁 Tamanho: [cyan]{input_file.stat().st_size / 1024:.1f} KB[/cyan]")
    
    if collection:
        console.print(f"📦 Coleção: [cyan]{collection}[/cyan]")
    
    if levels:
        console.print(f"📊 Níveis: [cyan]{levels}[/cyan]")
    
    if dry_run:
        console.print("[yellow]⚠️  Modo dry-run: não será armazenado[/yellow]")
    
    console.print()
    
    try:
        # Inicializar pipeline
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Inicializando pipeline...", total=None)
            
            config_path = str(config) if config else None
            pipeline = ODECIPipeline(config_path=config_path)
            
            progress.update(task, description="Pipeline inicializado ✓")
        
        # Parse níveis
        chunk_levels = parse_chunk_levels(levels)
        
        # Processar documento
        console.print("[bold]Processando documento...[/bold]")
        console.print()
        
        if dry_run:
            # Apenas carregar e chunkar
            document = pipeline.load_document(input_file)
            chunks = pipeline.chunk_document(document)
            
            stats = {
                "document_name": document.name,
                "collection_name": collection or f"odeci_{input_file.stem}",
                "total_chunks": chunks.total_chunks,
                "chunks_by_level": {
                    "parent": len(chunks.parent_chunks),
                    "child": len(chunks.child_chunks),
                    "atomic": len(chunks.atomic_chunks),
                },
                "chunks_by_domain": chunks.chunks_by_domain,
            }
        else:
            # Pipeline completo
            stats = pipeline.ingest_document(
                file_path=input_file,
                collection_name=collection,
                chunk_levels=chunk_levels,
            )
        
        # Exibir resultados
        console.print()
        console.print("[bold green]✓ Processamento concluído![/bold green]")
        console.print()
        
        # Tabela de estatísticas
        table = Table(title="Estatísticas", show_header=True)
        table.add_column("Métrica", style="cyan")
        table.add_column("Valor", style="green")
        
        table.add_row("Documento", stats["document_name"])
        table.add_row("Coleção", stats["collection_name"])
        table.add_row("Total de Chunks", str(stats["total_chunks"]))
        table.add_row("Parent Chunks", str(stats["chunks_by_level"]["parent"]))
        table.add_row("Child Chunks", str(stats["chunks_by_level"]["child"]))
        table.add_row("Atomic Chunks", str(stats["chunks_by_level"]["atomic"]))
        
        console.print(table)
        console.print()
        
        # Distribuição por domínio
        if stats.get("chunks_by_domain"):
            domain_table = Table(title="Distribuição por Domínio", show_header=True)
            domain_table.add_column("Domínio", style="cyan")
            domain_table.add_column("Quantidade", style="green")
            
            for domain, count in stats["chunks_by_domain"].items():
                domain_table.add_row(domain, str(count))
            
            console.print(domain_table)
            console.print()
        
        if not dry_run:
            console.print(
                f"[dim]Use 'python scripts/query_search.py -c {stats['collection_name']}' "
                f"para buscar[/dim]"
            )
        
    except FileNotFoundError as e:
        console.print(f"[red]❌ Erro: {e}[/red]")
        raise typer.Exit(code=1)
    except ValueError as e:
        console.print(f"[red]❌ Erro de configuração: {e}[/red]")
        raise typer.Exit(code=1)
    except Exception as e:
        console.print(f"[red]❌ Erro inesperado: {e}[/red]")
        if verbose:
            console.print_exception()
        raise typer.Exit(code=1)


@app.command()
def info(
    collection: str = typer.Option(
        ...,
        "--collection", "-c",
        help="Nome da coleção",
    ),
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        help="Caminho para arquivo de configuração",
    ),
) -> None:
    """
    Exibe informações sobre uma coleção.
    """
    setup_logging(level="WARNING")
    
    try:
        config_path = str(config) if config else None
        pipeline = ODECIPipeline(config_path=config_path)
        
        stats = pipeline.get_collection_stats(collection)
        
        console.print()
        
        if not stats["exists"]:
            console.print(f"[yellow]⚠️  Coleção '{collection}' não existe[/yellow]")
            return
        
        table = Table(title=f"Coleção: {collection}", show_header=True)
        table.add_column("Propriedade", style="cyan")
        table.add_column("Valor", style="green")
        
        table.add_row("Vetores", str(stats["vector_count"]))
        table.add_row("Dimensões", str(stats["dimensions"]))
        table.add_row("Backend", stats["backend"])
        
        console.print(table)
        console.print()
        
    except Exception as e:
        console.print(f"[red]❌ Erro: {e}[/red]")
        raise typer.Exit(code=1)


@app.command()
def delete(
    collection: str = typer.Option(
        ...,
        "--collection", "-c",
        help="Nome da coleção a remover",
    ),
    force: bool = typer.Option(
        False,
        "--force", "-f",
        help="Não pedir confirmação",
    ),
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        help="Caminho para arquivo de configuração",
    ),
) -> None:
    """
    Remove uma coleção do vector store.
    """
    setup_logging(level="WARNING")
    
    if not force:
        confirm = typer.confirm(
            f"Tem certeza que deseja remover a coleção '{collection}'?"
        )
        if not confirm:
            console.print("[yellow]Operação cancelada[/yellow]")
            raise typer.Exit(code=0)
    
    try:
        config_path = str(config) if config else None
        pipeline = ODECIPipeline(config_path=config_path)
        
        pipeline.delete_collection(collection)
        
        console.print(f"[green]✓ Coleção '{collection}' removida[/green]")
        
    except Exception as e:
        console.print(f"[red]❌ Erro: {e}[/red]")
        raise typer.Exit(code=1)


def main() -> None:
    """Entry point."""
    app()


if __name__ == "__main__":
    main()
