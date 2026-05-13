"""MCP server: RAG over airline delay CSV + lightweight schema tool."""

from __future__ import annotations

import csv
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from airline_reliability.csv_path import resolved_csv_path
from airline_reliability.rag_index import search_chunks

mcp = FastMCP("airline_reliability")


def _schema_text(path: Path) -> str:
    if not path.is_file():
        return (
            f"ERROR: CSV not found at {path}. "
            "Set AIRLINE_DELAY_CSV_PATH or place Airline_Delay_Cause.csv next to the "
            "installed package modules (same directory as app.py)."
        )
    with path.open(newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return "ERROR: empty CSV."
    cols = ", ".join(header)
    return (
        "Dataset: U.S. airline reporting — delay causes by carrier and airport.\n"
        f"Columns ({len(header)}): {cols}\n"
        "Numeric fields include arrival counts, delay counts by cause, and delay minutes. "
        "Use search_airline_delay_causes to pull relevant rows for a question."
    )


@mcp.tool()
def get_airline_delay_dataset_schema() -> str:
    """Return column names and a short description of the Airline_Delay_Cause CSV (no row data).

    Call this first if you need to know which fields exist before searching.
    """
    return _schema_text(resolved_csv_path())


@mcp.tool()
def search_airline_delay_causes(query: str, n_results: int = 10) -> str:
    """Semantic search over the airline delay CSV (RAG). Returns only the top matching row groups.

    Use natural language (e.g. carrier name, airport code, delay causes, weather delays).
    Prefer several focused searches over one vague query. n_results is capped (default 10).

    Args:
        query: What to look for in the delay data.
        n_results: How many text chunks to return (1–20).
    """
    path = resolved_csv_path()
    n = max(1, min(int(n_results), 20))
    try:
        return search_chunks(path, query.strip(), n_results=n)
    except Exception as e:
        return f"ERROR: RAG search failed: {e}"


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
