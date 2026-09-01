"""Operational CLI: `python -m pathwise.cli <command>`.

Used for seeding the knowledge graph and resource catalog, validating resource URLs,
and backfilling embeddings — all of which are operator tasks rather than API surface.
"""

from __future__ import annotations

import typer

from pathwise.config import get_settings
from pathwise.logging_config import configure_logging

app = typer.Typer(help="Pathwise operational commands.", no_args_is_help=True)


@app.callback()
def _init() -> None:
    settings = get_settings()
    configure_logging(level=settings.log_level, json_output=False)


@app.command()
def info() -> None:
    """Print the resolved (non-secret) configuration."""
    settings = get_settings()
    typer.echo(f"env             : {settings.env}")
    typer.echo(f"llm provider    : {settings.llm_provider} ({settings.llm_model})")
    typer.echo(f"embeddings      : {settings.embedding_provider} ({settings.embedding_model})")
    typer.echo(f"embedding dim   : {settings.embedding_dim}")
    typer.echo(f"database host   : {settings.database_url.hosts()[0].get('host')}")


if __name__ == "__main__":
    app()
