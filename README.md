# Airline reliability

A **UV**-managed Python app (**airline-reliability**) that combines **Gradio**, **LangGraph**, **OpenAI (gpt-4o-mini)**, and an **MCP** server backed by **RAG** (Chroma + OpenAI embeddings) over the bundled **Airline_Delay_Cause.csv** (shipped next to `app.py` under `src/airline_reliability/`).

## What it does

- **Gradio** exposes a browser chat UI.
- **LangGraph** runs a tool-calling loop: the model decides when to call MCP tools, and a `ToolNode` executes them.
- **MCP (stdio)** runs as a child process. It exposes tools that describe the dataset schema and **semantically search** chunked CSV text so the chat model is not fed the entire file at once.
- **Chroma** persists a local vector index under **`.chroma_airline_reliability/`** (rebuilt when the CSV changes or the index version bumps). Embeddings use OpenAI’s API.

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/)
- `OPENAI_API_KEY` (chat + embeddings)

## Setup

```bash
cd airline-reliability
uv sync
cp .env.example .env
# Edit .env and set OPENAI_API_KEY (and optional variables below).
```

By default the CSV is **`src/airline_reliability/Airline_Delay_Cause.csv`**. Override with **`AIRLINE_DELAY_CSV_PATH`** if you want another file.

## Run

```bash
uv run airline-reliability-chat
```

Then open the URL Gradio prints (default `http://127.0.0.1:7860`). Override the port with **`GRADIO_SERVER_PORT`**.

### MCP server only (stdio)

Useful for wiring the same tools into Cursor or another MCP client:

```bash
uv run airline-reliability-mcp
```

## Environment variables

| Variable | Purpose |
|----------|---------|
| `OPENAI_API_KEY` | Required for gpt-4o-mini and embedding calls (parent process forwards into the MCP subprocess for RAG). |
| `AIRLINE_DELAY_CSV_PATH` | Optional absolute path to the CSV. |
| `AIRLINE_DELAY_EMBED_MODEL` | Optional embedding model (default `text-embedding-3-small`). |
| `AIRLINE_DELAY_RAG_ROWS_PER_CHUNK` | Optional rows per text chunk when indexing (default `15`). |
| `GRADIO_SERVER_PORT` | Optional Gradio port (default `7860`). |

## Architecture diagram

**C4 (PlantUML)** diagrams:

- [`docs/c4-containers.puml`](docs/c4-containers.puml) — system boundary: Gradio, LangGraph, MCP, Chroma, CSV, OpenAI.
- [`docs/c4-components-mcp.puml`](docs/c4-components-mcp.puml) — components inside the MCP / RAG process.

Render with any PlantUML-compatible viewer, for example:

```bash
# If you have the plantuml.jar CLI:
java -jar plantuml.jar docs/c4-containers.puml docs/c4-components-mcp.puml
```

Or use the PlantUML extension in VS Code / Cursor.

## Project layout

```text
airline-reliability/
  pyproject.toml
  README.md
  src/airline_reliability/
    Airline_Delay_Cause.csv   # bundled dataset
    app.py                    # Gradio entrypoint
    graph.py                  # LangGraph + MCP client + ChatOpenAI
    mcp_server.py             # FastMCP tools (schema + RAG search)
    rag_index.py              # Chroma index build/query, token-safe batches
    csv_path.py               # Resolves CSV path from env / defaults
  docs/
    c4-containers.puml
    c4-components-mcp.puml
```

## Notes

- The first RAG query after a CSV or index-version change may take a while while chunks are embedded and written to Chroma.
- **`.chroma_airline_reliability/`** is a local cache; safe to delete to force a full re-index.
