# GroundMemory - Documentation

This document covers installation, integration guides, configuration reference, and environment variables.
For a project overview and quick start, see [README.md](README.md).

---

## Table of Contents

- [GroundMemory - Documentation](#groundmemory---documentation)
  - [Table of Contents](#table-of-contents)
  - [Installation \& Configuration](#installation--configuration)
    - [Option 1 - Docker](#option-1---docker)
    - [Option 2 - pip](#option-2---pip)
    - [Network Access](#network-access)
    - [Embedding Providers](#embedding-providers)
  - [MCP Server](#mcp-server)
    - [Running the Server](#running-the-server)
    - [Client Configuration](#client-configuration)
    - [Available MCP Tools](#available-mcp-tools)
    - [Bootstrap - Loading Memory at Session Start](#bootstrap---loading-memory-at-session-start)
  - [Connecting to Your AI Agent Using The Python API](#connecting-to-your-ai-agent-using-the-python-api)
    - [OpenAI](#openai)
    - [Anthropic](#anthropic)
  - [Python API Example](#python-api-example)
  - [Tools Reference](#tools-reference)
  - [Architecture](#architecture)
    - [Architectural Layers](#architectural-layers)
      - [1. Workspace (`GroundMemory/core/workspace.py`)](#1-workspace-groundmemorycoreworkspacepy)
      - [2. Memory Storage (`GroundMemory/core/storage.py`)](#2-memory-storage-groundmemorycorestoragepy)
      - [3. Text Chunker (`GroundMemory/core/chunker.py`)](#3-text-chunker-groundmemorycorechunkerpy)
      - [4. Embedding Providers (`GroundMemory/core/embeddings.py`)](#4-embedding-providers-groundmemorycoreembeddingspy)
      - [5. Memory Index (`GroundMemory/core/index.py`)](#5-memory-index-groundmemorycoreindexpy)
      - [6. Hybrid Search (`GroundMemory/core/search.py`)](#6-hybrid-search-groundmemorycoresearchpy)
      - [7. Relation Graph (`GroundMemory/core/relations.py`)](#7-relation-graph-groundmemorycorerelationspy)
      - [8. Sync (`GroundMemory/core/sync.py`)](#8-sync-groundmemorycoresyncpy)
      - [9. Bootstrap Injector (`GroundMemory/bootstrap/injector.py`)](#9-bootstrap-injector-groundmemorybootstrapinjectorpy)
      - [9a. Token Counter (`GroundMemory/bootstrap/token_counter.py`)](#9a-token-counter-groundmemorybootstraptoken_counterpy)
      - [9b. Workspace Backup (`GroundMemory/core/backup.py`)](#9b-workspace-backup-groundmemorycorebackuppy)
      - [10. Tools (`GroundMemory/tools/`)](#10-tools-groundmemorytools)
      - [11. LLM Adapters (`GroundMemory/adapters/`)](#11-llm-adapters-groundmemoryadapters)
      - [12. Session (`GroundMemory/session.py`)](#12-session-groundmemorysessionpy)
      - [12 (note). Session vs Workspace - not two different things](#12-note-session-vs-workspace---not-two-different-things)
  - [Data Flow](#data-flow)
  - [Tech Stack](#tech-stack)
  - [Configuration](#configuration)
    - [Minimum Config](#minimum-config)
    - [Environment Variables](#environment-variables)

---

## Installation & Configuration

### Option 1 - Docker

Docker is the recommended way to run GroundMemory. It requires no Python environment setup and keeps your workspace data in a local `./data` directory.

```bash
git clone https://github.com/huss-mo/GroundMemory && cd GroundMemory
cp groundmemory/config/.env.example .env
docker compose up -d
# → listening on http://127.0.0.1:4242/mcp
```

The default compose file starts a single `GroundMemory` service using BM25-only search (no embedding API required). Edit `.env` to switch providers - see [Embedding Providers](#embedding-providers) below.

To run a **second workspace** on a different port (e.g. for a separate project or user), uncomment the `GroundMemory-personal` service in `docker-compose.yml`:

```yaml
# GroundMemory-personal:
#   build:
#     context: .
#   image: GroundMemory:latest
#   restart: unless-stopped
#   ports:
#     - "4243:4242"
#   volumes:
#     - ./data:/data
#   env_file:
#     - .env
#   environment:
#     GROUNDMEMORY_WORKSPACE: personal
```

Workspace data is stored in `./data/<workspace-name>/` on the host and persists across container restarts.

**Building with sentence-transformers (local embeddings)**

The default Docker image does not include `sentence-transformers`. To build an image that supports the `local` embedding provider, pass the `EXTRAS=local` build argument:

```bash
docker compose build --build-arg EXTRAS=local
docker compose up -d
```

Then set `GROUNDMEMORY_EMBEDDING__PROVIDER=local` in your `.env`.

### Option 2 - pip

For development or direct integration without Docker:

```bash
# BM25-only - no extra dependencies
pip install groundmemory

# With local sentence-transformers embeddings
pip install "groundmemory[local]"
```

Then start the MCP server:

```bash
GROUNDMEMORY_WORKSPACE=my-project groundmemory-mcp
# → listening on http://127.0.0.1:4242/mcp
```

**Configuration for pip installs**

GroundMemory reads config from `~/.groundmemory/` - the same directory where workspace data lives. Place your config files there once and they'll be found regardless of which directory you run `groundmemory-mcp` from:

```
~/.groundmemory/
├── .env                  ← config file (copy from .env.example in the repo root)
└── default/              ← workspace data (auto-created on first run)
    ├── MEMORY.md
    └── ...
```

`.env` is optional - you can also set any setting directly as an environment variable. A `./.env` in the current working directory is also checked as a fallback, which is useful for per-project overrides in dev mode.

Copy the example file to get started:

```bash
cp .env.example ~/.groundmemory/.env
```

### Network Access

By default, GroundMemory binds to `127.0.0.1` (localhost only). This means only processes on the same machine can reach the server - which is the right default for a single-user setup.

**LAN access (same local network)**

To accept connections from other devices on your network, set the host to `0.0.0.0` and add your server's LAN address to the allowed-hosts list:

```bash
# pip / direct install
GROUNDMEMORY_MCP__HOST=0.0.0.0 \
GROUNDMEMORY_MCP__ALLOWED_HOSTS="192.168.1.50:4242" \
groundmemory-mcp
```

For Docker, uncomment the two required network-access lines in `docker-compose.yml` (see the comments in that file).

`allowed_hosts` is the DNS-rebinding protection allowlist - it controls which `Host:` header values the server accepts. List every address clients will use to reach the server (e.g. `192.168.1.50:4242`). `localhost` and `127.0.0.1` are always allowed implicitly. Only exact strings are supported - wildcards and CIDR ranges are not.

**Authentication**

When exposing the server beyond localhost, set `GROUNDMEMORY_MCP__API_KEY` to a secret token. Every request must then include:

```
Authorization: Bearer <your-token>
```

MCP clients that support custom headers (Cline, Cursor, Claude Desktop) can be configured like this:

```json
{
  "mcpServers": {
    "GroundMemory": {
      "url": "http://192.168.1.50:4242/mcp",
      "headers": {
        "Authorization": "Bearer <your-secret-token>"
      }
    }
  }
}
```

When `api_key` is not set (the default), no authentication is enforced and the server behaves exactly as before - no breaking change.

**Public internet access**

Do not expose GroundMemory directly on the public internet. Use a reverse proxy (nginx, Caddy, Traefik) with TLS in front of it, and set `api_key` for authentication.

The `GROUNDMEMORY_MCP__FORWARDED_ALLOW_IPS` setting controls which upstream IPs uvicorn trusts to pass `X-Forwarded-For` / `X-Real-IP` headers. Set it to your proxy's internal IP when running behind a reverse proxy (default: `127.0.0.1`).

### Embedding Providers

GroundMemory supports three embedding providers. You can switch between them at any time by changing `GROUNDMEMORY_EMBEDDING__PROVIDER`. No data migration is required.

| Provider | Value | Extra install required? | When to use |
|---|---|---|---|
| BM25-only | `none` | No | Default. Pure keyword search via SQLite FTS5. Works offline, no API key needed. |
| OpenAI-compatible API | `openai` | No | Any HTTP embedding API: OpenAI, Ollama, LM Studio, LiteLLM, vLLM, Mistral, etc. Requires `GROUNDMEMORY_EMBEDDING__BASE_URL` and `GROUNDMEMORY_EMBEDDING__API_KEY`. |
| Local sentence-transformers | `local` | Yes - `pip install "groundmemory[local]"` or `--build-arg EXTRAS=local` for Docker | Fully offline vector embeddings. Downloads model on first run. |

**`none` - BM25-only (default)**

No configuration needed. GroundMemory uses SQLite FTS5 for all search. Ideal for getting started quickly or for air-gapped environments.

```bash
GROUNDMEMORY_EMBEDDING__PROVIDER=none groundmemory-mcp
```

**`openai` - OpenAI-compatible HTTP API**

Works with any endpoint that follows the OpenAI embeddings API format. No extra Python packages are required - only `httpx`, which is a core dependency already installed with GroundMemory.

```bash
# Real OpenAI
GROUNDMEMORY_EMBEDDING__PROVIDER=openai \
GROUNDMEMORY_EMBEDDING__API_KEY=sk-... \
GROUNDMEMORY_EMBEDDING__MODEL=text-embedding-3-small \
groundmemory-mcp

# Ollama (local server, no API key needed)
GROUNDMEMORY_EMBEDDING__PROVIDER=openai \
GROUNDMEMORY_EMBEDDING__BASE_URL=http://localhost:11434/v1 \
GROUNDMEMORY_EMBEDDING__API_KEY=ollama \
GROUNDMEMORY_EMBEDDING__MODEL=nomic-embed-text \
groundmemory-mcp

# LM Studio
GROUNDMEMORY_EMBEDDING__PROVIDER=openai \
GROUNDMEMORY_EMBEDDING__BASE_URL=http://localhost:1234/v1 \
GROUNDMEMORY_EMBEDDING__API_KEY=lm-studio \
GROUNDMEMORY_EMBEDDING__MODEL=nomic-ai/nomic-embed-text-v1.5-GGUF \
groundmemory-mcp
```

**`local` - sentence-transformers (offline)**

Runs a local embedding model entirely on your machine - no network call, no API key. Requires installing the optional `local` extra, which pulls in `sentence-transformers` and its dependencies (PyTorch, Transformers, etc.). The model is downloaded from HuggingFace on first use.

```bash
# Install the extra first
pip install "groundmemory[local]"

GROUNDMEMORY_EMBEDDING__PROVIDER=local \
GROUNDMEMORY_EMBEDDING__LOCAL_MODEL=all-MiniLM-L6-v2 \
groundmemory-mcp
```

---

## MCP Server

GroundMemory can run as a standalone MCP (Model Context Protocol) server over HTTP, exposing all 8 memory tools to any MCP-compatible client - including Claude Desktop, Cursor, Cline, and custom agents.

Each server instance owns a single workspace. Multiple workspaces require multiple server processes running on different ports.

### Running the Server

The `groundmemory-mcp` command is available after installing GroundMemory (MCP support is included by default).

```bash
# Default: workspace "default", host 127.0.0.1, port 4242
groundmemory-mcp

# Custom workspace
GROUNDMEMORY_WORKSPACE=my-project groundmemory-mcp

# Custom host and port
GROUNDMEMORY_MCP__HOST=0.0.0.0 GROUNDMEMORY_MCP__PORT=9000 groundmemory-mcp
```

The server starts at `http://<host>:<port>/mcp` using the `streamable-http` MCP transport.

### Client Configuration

GroundMemory speaks standard MCP over HTTP, so any MCP-compatible client works. The table below shows the most common categories and examples - the list is not exhaustive.

| Category | Examples | How to connect |
|---|---|---|
| AI coding assistants | Cursor, Cline, Windsurf, Claude Code, Codex CLI | Add the JSON snippet below to your client's MCP server config |
| AI desktop clients | Claude Desktop, Open WebUI | Add the JSON snippet below via the client's settings |
| Agent frameworks & platforms | LangChain, LangGraph, CrewAI, AutoGen, Google ADK, LiteLLM, n8n | Python API (`MemorySession`) or point an HTTP tool node at the MCP endpoint |

```json
{
  "mcpServers": {
    "GroundMemory": {
      "url": "http://<server-ip>:4242/mcp"
    }
  }
}
```

For clients that use the `stdio` transport, add the following block instead:
```json
{
  "mcpServers": {
    "GroundMemory": {
      "command": "npx",
      "args": [
        "mcp-remote@latest", 
        "http://<server-ip>:4242/mcp", 
        "--allow-http"
      ]
    }
  }
}
```

If an api key is set on the server, add `--header` and the token value to the `args` list (both lines are required):
```json
{
  "mcpServers": {
    "GroundMemory": {
      "command": "npx",
      "args": [
        "mcp-remote@latest",
        "http://<server-ip>:4242/mcp",
        "--allow-http",
        "--header",
        "Authorization: Bearer your-secret-token"
      ]
    }
  }
}
```

Clients that support the MCP Prompts primitive (such as Cline and Claude Desktop) will also show a `memory_bootstrap_prompt` entry in their Prompts panel - click it at session start to inject memory context without waiting for the agent to call the tool. For agent frameworks and platforms that use the Python API, see [Connecting to Your AI Agent Using The Python API](#connecting-to-your-ai-agent-using-the-python-api).

### Available MCP Tools

Once connected, the client has access to **4 core tools** and 1 prompt. Two additional tools are available behind config flags.

**Core tools (always registered)**

| Tool | Description |
|---|---|
| `memory_bootstrap` | **Call once at session start.** Returns the full memory context (MEMORY.md, USER.md, AGENTS.md, RELATIONS.md, daily logs) as a formatted string. |
| `memory_read` | Unified read tool. Supply `query` for hybrid search (SEARCH mode) or `file` for direct file/line-range access (GET mode). |
| `memory_write` | Unified write tool. APPEND, REPLACE_TEXT, REPLACE_LINES, or DELETE - mode is selected by the combination of parameters supplied. |
| `memory_relate` | Record a typed entity relationship (`subject → predicate → object`) with semantic deduplication. |

**Optional tools (config-gated)**

| Tool | Config flag | Description |
|---|---|---|
| `memory_list` | `mcp.expose_memory_list: true` | List workspace files with sizes and line counts. |
| `memory_tool` | `mcp.dispatcher_mode: true` | Single dispatcher tool - replaces all four core tools with one `action` + `args` call. |

| Prompt | Description |
|---|---|
| `memory_bootstrap_prompt` | Same content as `memory_bootstrap`, exposed as an MCP Prompt for clients that support the Prompts primitive (Cline, Claude Desktop). Click it in your client's Prompts panel at session start instead of waiting for the agent to call the tool. |

### Bootstrap - Loading Memory at Session Start

GroundMemory's memory context (long-term facts, user profile, agent instructions, entity graph, daily logs) needs to be loaded at the start of each session. Two mechanisms are provided:

**Tool-based bootstrap (all clients)**

The `memory_bootstrap` tool description is written so that most agents call it automatically at the start of a session without any explicit instruction - the tool's description alone signals that it should be the first action taken. **No system-prompt changes are necessary in most cases.**

If you find that your agent does not call `memory_bootstrap` on its own, you can add an explicit fallback instruction to the system prompt:

```
At the start of every session, call memory_bootstrap before doing anything else.
Use the returned context as your background knowledge for the rest of the session.
```

**Prompt-based bootstrap (Cline, Claude Desktop)**

Clients that support the MCP Prompts primitive (Cline, Claude Desktop) will show a `memory_bootstrap_prompt` entry in their Prompts panel. Click it at the start of a session to inject the memory context directly into the conversation - no agent tool call required. This is an alternative to the tool-based path, useful when you want to load memory context manually rather than waiting for the agent to call the tool.

The content returned by both mechanisms is identical.

---

## Connecting to Your AI Agent Using The Python API

GroundMemory exposes standard JSON schemas for function calling, compatible with OpenAI and Anthropic out of the box. The primary export is `ALL_TOOLS` - a list of `(schema, run)` pairs. Pass the schemas to the model so it knows what tools are available; when the model calls a tool, dispatch it back through the paired `run` function (or use `session.execute_tool` directly). Both paths are shown below.

### OpenAI

```python
from openai import OpenAI
from GroundMemory.session import MemorySession
from GroundMemory.tools import ALL_TOOLS
import json

session = MemorySession.create("my-project")
client = OpenAI()

# Build the tools list for the API call
tools = [{"type": "function", "function": schema} for schema, _ in ALL_TOOLS]

response = client.chat.completions.create(
    model="gpt-4.1",
    messages=[
        {"role": "system", "content": session.bootstrap()},
        {"role": "user", "content": "What do you remember about my preferences?"},
    ],
    tools=tools,
)

# Dispatch tool calls the model makes back to GroundMemory
for call in response.choices[0].message.tool_calls or []:
    result = session.execute_tool(call.function.name, **json.loads(call.function.arguments))
    print(result)
```

### Anthropic

```python
from anthropic import Anthropic
from GroundMemory.session import MemorySession
from GroundMemory.tools import ALL_TOOLS

session = MemorySession.create("my-project")
client = Anthropic()

# Anthropic tool schema uses input_schema instead of parameters
tools = [
    {
        "name": schema["name"],
        "description": schema["description"],
        "input_schema": schema["parameters"],
    }
    for schema, _ in ALL_TOOLS
]

response = client.messages.create(
    model="claude-opus-4-6",
    max_tokens=1024,
    system=session.bootstrap(),
    messages=[{"role": "user", "content": "Summarize what you know about me."}],
    tools=tools,
)

# Dispatch tool use blocks back to GroundMemory
for block in response.content:
    if block.type == "tool_use":
        result = session.execute_tool(block.name, **block.input)
        print(result)
```

> **Complete runnable agents** - `examples/openai_agent.py` and `examples/anthropic_agent.py`
> show a full interactive loop with workspace sync on startup and
> graceful shutdown. Run them with:
> ```bash
> uv run python examples/openai_agent.py
> uv run python examples/anthropic_agent.py
> ```

---

## Python API Example

```python
from groundmemory.session import MemorySession

# Create (or reopen) a named workspace
session = MemorySession.create("my-project")

# Append a long-term memory
session.execute_tool("memory_write", file="MEMORY.md", content="User prefers concise answers.")

# Append a daily log entry
session.execute_tool("memory_write", file="USER.md", content="Working on the auth service refactor.")

# Record a relationship between entities
session.execute_tool("memory_relate", subject="Alice", predicate="works_at", object="Acme Corp")

# Search across all memory tiers (SEARCH mode)
result = session.execute_tool("memory_read", query="communication preferences")
for item in result["results"]:
    print(item["content"])

# Read a specific file (GET mode)
result = session.execute_tool("memory_read", file="USER.md")
print(result["content"])

# Read a line range (1-indexed, GET mode)
result = session.execute_tool("memory_read", file="USER.md", start_line=5, end_line=10)
print(result["content"])

# Replace the first occurrence of a string (REPLACE_TEXT mode)
session.execute_tool("memory_write", file="USER.md", search="old text", content="new text")

# Replace a line range (REPLACE_LINES mode)
session.execute_tool("memory_write", file="USER.md", start_line=5, end_line=7, content="new content")

# Hard-delete a line range - physically removes lines, no tombstone (DELETE mode)
session.execute_tool("memory_write", file="USER.md", start_line=5, end_line=7, content="")

# Build the system prompt context block for your agent
system_prompt = session.bootstrap()
```

`MemorySession.create("my-project")` creates a workspace directory at `~/.groundmemory/my-project/` on first run, seeding all required files. Subsequent calls reopen the same workspace.

---

## Tools Reference

Use `session.execute_tool(name, **kwargs)` to call tools programmatically, or pass the schemas from `build_tool_registry(config)` to your model framework.

> **Immutability rule:** `MEMORY.md` and all `daily/*.md` files are append-only. The DELETE, REPLACE_TEXT, and REPLACE_LINES modes of `memory_write` will reject any attempt to modify them. Use `memory_write` in APPEND mode to add new information to these files.
>
> **Hard-delete:** DELETE mode physically removes lines from the file - no tombstone comment is written.

**`memory_read`** - Unified read tool

| Parameter | Type | Description |
|---|---|---|
| `query` | string (optional) | Natural-language search query → **SEARCH mode** |
| `file` | string (optional) | File to read → **GET mode** (or SEARCH filter when combined with `query`) |
| `top_k` | int (optional) | Max results to return in SEARCH mode |
| `start_line` | int (optional) | 1-based first line to return in GET mode |
| `end_line` | int (optional) | 1-based last line (inclusive) in GET mode |

Mode dispatch: `query` alone → SEARCH; `file` alone → GET; both → GET (file wins for slicing); neither → error.

**`memory_write`** - Unified write tool

| Mode | Trigger | Parameters |
|---|---|---|
| **APPEND** | No `search`, no `start_line`/`end_line` | `file`, `content` (+ optional `tags`) |
| **REPLACE_TEXT** | `search` provided | `file`, `search`, `content` |
| **REPLACE_LINES** | `start_line` + `end_line` + non-empty `content` | `file`, `start_line`, `end_line`, `content` |
| **DELETE** | `start_line` + `end_line` + `content=""` | `file`, `start_line`, `end_line`, `content=""` |

APPEND targets and their destination files:

| `file` value | Written to | Behaviour |
|---|---|---|
| `MEMORY.md` | `MEMORY.md` | Appended permanently. **Immutable** - append only. |
| `daily` | `daily/YYYY-MM-DD.md` | Appended to today's log. **Immutable** - append only. |
| `USER.md` | `USER.md` | Updates the stable user profile. Mutable. |
| `AGENTS.md` | `AGENTS.md` | Updates agent operating instructions. Mutable. |

**`memory_read` source filters** (SEARCH mode, pass `file=` to restrict):

| `file` value | Searches |
|---|---|
| *(omitted)* | All files |
| `MEMORY.md` | Long-term memory only |
| `USER.md` | User profile only |
| `AGENTS.md` | Agent instructions only |
| `daily` | All daily logs |
| `RELATIONS.md` | Relations file + SQLite graph |

**`memory_relate`** - Record a typed entity relationship

| Parameter | Type | Default | Description |
|---|---|---|---|
| `subject` | string | required | Source entity (e.g. "Alice") |
| `predicate` | string | required | Relationship type (e.g. "works_at") |
| `object` | string | required | Target entity (e.g. "Acme Corp") |
| `note` | string | `""` | Optional free-text annotation |
| `source_file` | string | `"RELATIONS.md"` | Workspace-relative file |
| `confidence` | float | `1.0` | Confidence score 0.0–1.0 |
| `supersedes` | bool | `false` | Delete all prior `(subject, predicate)` triples before writing |

**`memory_list`** *(optional - requires `mcp.expose_memory_list: true`)*

Lists all workspace files with sizes and line counts. No required parameters.

**`memory_tool`** *(optional - requires `mcp.dispatcher_mode: true`)*

Single dispatcher that replaces all four core tools. Pass `action` (one of `read`, `write`, `bootstrap`, `relate`, `list`, `describe`) and `args` (the same parameters the underlying tool accepts).

---

## Architecture

### Architectural Layers

#### 1. Workspace (`GroundMemory/core/workspace.py`)
`Workspace` is a pure filesystem abstraction for a single memory workspace. On first use it creates the directory tree and seeds the default Markdown files (`MEMORY.md`, `USER.md`, `AGENTS.md`, `RELATIONS.md`). All other layers receive a `Workspace` object to resolve file paths - it never holds runtime state such as a database connection or embedding provider.

The on-disk layout is a **single directory level** under `~/.groundmemory`:

```
~/.groundmemory/
└── <workspace_name>/          ← one directory per named workspace
    ├── MEMORY.md              long-term curated memory
    ├── USER.md                stable user profile
    ├── AGENTS.md              agent operating instructions
    ├── RELATIONS.md           human-readable entity relation graph
    ├── daily/
    │   └── YYYY-MM-DD.md     append-only daily logs
    └── .index/
        └── memory.db          SQLite index (chunks + FTS5 + relations + embeddings)
```

#### 2. Memory Storage (`GroundMemory/core/storage.py`)
Low-level atomic Markdown file I/O. All writes go through a temp-file + rename cycle to prevent partial writes. Provides `write_long_term`, `write_daily`, `read_file`, `delete_lines`, and `list_daily_files`.

#### 3. Text Chunker (`GroundMemory/core/chunker.py`)
Splits Markdown files into overlapping chunks that respect heading boundaries. Each `Chunk` carries a deterministic `chunk_id` (SHA-256 of path + start line + text) and 0-indexed line ranges for precise `memory_get` retrieval.

#### 4. Embedding Providers (`GroundMemory/core/embeddings.py`)
Abstract `EmbeddingProvider` with three concrete implementations:

| Provider | Class | Notes |
|---|---|---|
| `none` | `NullEmbeddingProvider` | Returns empty vectors; BM25-only path, no network or GPU |
| `openai` | `OpenAICompatibleProvider` | HTTP calls via `httpx`; works with any OpenAI-compatible endpoint |
| `local` | `SentenceTransformerProvider` | Runs a local model in-process via `sentence-transformers` |

For provider configuration, install commands, and usage examples, see [Embedding Providers](#embedding-providers).

#### 5. Memory Index (`GroundMemory/core/index.py`)
SQLite database (`memory.db`) with five tables:

| Table | Purpose |
|---|---|
| `files` | Tracks indexed files with SHA-256 hash + mtime for change detection |
| `chunks` | Text chunks with JSON-serialised embedding vectors |
| `chunks_fts` | FTS5 virtual table - BM25 keyword search via SQLite triggers |
| `relations` | Named entity relationships (subject → predicate → object) |
| `embedding_cache` | Reuses embeddings when chunk content hasn't changed |

Vector search is implemented in pure Python (NumPy cosine similarity) so it works everywhere without native extensions. The database runs in WAL (Write-Ahead Logging) mode (`PRAGMA journal_mode=WAL`) for better concurrent read performance.

#### 6. Hybrid Search (`GroundMemory/core/search.py`)
Nine-step pipeline:
1. **Embed** the query via the configured provider.
2. **Vector search** - cosine similarity over all chunk embeddings → top `k × candidate_multiplier` candidates.
3. **Keyword search** - FTS5 BM25 → top `k × candidate_multiplier` candidates.
4. **Merge & re-score** - `score = vector_weight × vec_score + (1 − vector_weight) × bm25_score`.
5. **Cross-encoder reranking** (optional) - if `search.rerank_model` is set, a cross-encoder rescores the merged candidates for higher precision.
6. **Temporal decay** - `score × exp(−decay_rate × days_old)` (disabled by default; applied post-rerank so recency nudges the final order).
7. **MMR diversification** (optional) - if `search.mmr_lambda > 0`, Maximal Marginal Relevance greedily selects `top_k` results that balance relevance against similarity to already-selected results: `mmr_score = mmr_lambda × relevance − (1 − mmr_lambda) × max_cosine_sim_to_selected`. Set via config only - not exposed as a tool parameter.
8. **Graph expansion** - extract entity mentions from top results, attach related relation triples as `relation_context`.
9. Return top `k` as `SearchResult` objects.

#### 7. Relation Graph (`GroundMemory/core/relations.py`)
Single consolidated module for all relation logic. Stores typed entity triples (`subject → predicate → object`) in two places simultaneously:
- **SQLite** `relations` table - fast structured lookup used by graph expansion during search.
- **`RELATIONS.md`** - human-readable, editable mirror, injected at bootstrap.

**Source-of-truth model:** `RELATIONS.md` is the authoritative record. Any change to the file is automatically reconciled back into SQLite.

**Public API:**

| Symbol | Description |
|---|---|
| `add_relation(...)` | Write a relation to both SQLite and RELATIONS.md with semantic dedup |
| `get_relations(...)` | Read relations from SQLite |
| `parse_relations_from_text(text)` | Parse valid relation lines from a raw string; used by `memory_delete` to identify rows to remove without a temp-file round-trip |
| `parse_relations_from_file(path)` | Parse valid lines from RELATIONS.md into `{subject, predicate, object, note}` dicts (delegates to `parse_relations_from_text`) |
| `sync_relations_from_file(path, index)` | Upsert/delete SQLite rows to match RELATIONS.md exactly; called by `sync_file` / `sync_workspace` and via `sync_after_edit` after every in-place edit |
| `validate_relations_replacement(text)` | Validate that every non-blank, non-comment line in a replacement string matches the RELATIONS.md format; returns `(all_valid, valid_lines, invalid_lines)` |
| `format_relations_for_context` | Format the relation graph as a Markdown block for bootstrap injection |
| `RELATION_LINE_RE` | Compiled regex for a valid RELATIONS.md line |
| `RELATIONS_FORMAT_REMINDER` | Human-readable format reminder string included in validation error responses |

Format for each line in RELATIONS.md:
```
- [Subject] --predicate--> [Object] (YYYY-MM-DD) - "optional note"
```

Semantic deduplication: before inserting, the new triple is embedded and compared (cosine similarity) against all existing triples. If similarity ≥ `dedup_threshold` (default 0.92) the write is skipped and the existing triple is returned.

#### 8. Sync (`GroundMemory/core/sync.py`)
Keeps the SQLite index consistent with the Markdown files using SHA-256 content hashing (not timestamps). `sync_workspace` walks all files and re-indexes changed ones. `sync_file` force-syncs a single file - called immediately after every `memory_write` so new content is searchable within the same session.

#### 9. Bootstrap Injector (`GroundMemory/bootstrap/injector.py`)
Assembles a system-prompt block from workspace files, respecting per-file and total character budgets (`max_chars_per_file`, `max_total_chars`). Truncated files get a visible `[TRUNCATED - use memory_get to read the rest]` marker. Injects (in order): long-term memory, user profile, agent instructions, relation graph, and daily logs. The number of daily log files injected is controlled by `daily_log_days` (default: 1 = today only; set to 2 for today + yesterday).

When `compaction_token_threshold > 0`, the injector counts the tokens in the assembled prompt (using either `len // 4` or `tiktoken`) and, if the count is above the threshold, appends a compaction notice instructing the agent to compact the configured tiers one by one using `memory_compact`.

#### 9a. Token Counter (`GroundMemory/bootstrap/token_counter.py`)
Counts tokens in the bootstrap string. Two methods: `"approx"` (`len(text) // 4`, no extra deps, default) and `"tiktoken"` (accurate BPE count via the tiktoken library, which is included as a core dependency). Falls back to `"approx"` if tiktoken is not importable.

#### 9b. Workspace Backup (`GroundMemory/core/backup.py`)
Creates timestamped zip archives of the workspace before compaction. Archives are stored in `<workspace>/backups/YYYY-MM-DD_HHmmss.zip` and contain all Markdown files, the `daily/` directory, and `.index/memory.db`. Provides `create_backup`, `list_backups`, `parse_spec` (resolves user-supplied restore specs), and `restore_backup`.

#### 10. Tools (`GroundMemory/tools/`)
Four core tools + two optional (config-gated) tools exposed to the LLM via function calling:

| Tool | File | Notes |
|---|---|---|
| `memory_bootstrap` | - | Assembles and returns the full workspace context as a Markdown string. |
| `memory_read` | - | Unified read: SEARCH mode (hybrid vector+BM25) or GET mode (file/line-range). |
| `memory_write` | various | Unified write: APPEND, REPLACE_TEXT, REPLACE_LINES, DELETE - dispatched by parameter combination. Hard-delete (physical line removal); rejected on `MEMORY.md`/`daily/*.md` for edit modes. |
| `memory_relate` | `RELATIONS.md` + SQLite | Semantic dedup before insert. `supersedes=True` deletes prior `(subject, predicate)` triples. |
| `memory_list` *(optional)* | - | Lists all workspace files with sizes and line counts. Gated by `mcp.expose_memory_list`. |
| `memory_tool` *(optional)* | - | Single dispatcher - routes `action` + `args` to the appropriate underlying tool. Gated by `mcp.dispatcher_mode`. |

**Tool registry (`GroundMemory/tools/__init__.py`):**

`build_tool_registry(config)` returns `(all_tools, tool_runners, tool_schemas)` based on config flags. `dispatcher_mode=True` replaces all four core tools with the single `memory_tool` dispatcher. `expose_memory_list=True` adds `memory_list` to the core set.

**Shared utilities (`GroundMemory/tools/base.py`):**

| Symbol | Description |
|---|---|
| `ok(data)` / `err(msg)` | Wrap a successful or error tool result |
| `is_immutable(file)` | Return `True` for `MEMORY.md` and `daily/*.md` |
| `sync_after_edit(session, resolved, is_relations, base_payload)` | Re-index a file after an in-place edit and return `ok(payload)`. Calls `sync_file` (and, when `is_relations=True`, `sync_relations_from_file`) non-fatally - sync failures add a `warning` key rather than raising. |

#### 11. LLM Adapters (`GroundMemory/adapters/`)
Thin schema-conversion + agentic-loop helpers:
- **`adapters/openai.py`** - converts schemas to OpenAI function-calling format; `handle_tool_calls` dispatches tool calls and appends results to the message list; `run_agent_loop` iterates until the model stops calling tools.
- **`adapters/anthropic.py`** - same for Anthropic's `tool_use` / `tool_result` block format.

#### 12. Session (`GroundMemory/session.py`)
`MemorySession` is the composition root that holds references to `Workspace`, `MemoryIndex`, and `EmbeddingProvider`. It exposes `execute_tool`, `bootstrap`, and `sync` as the primary API surface.

#### 12 (note). Session vs Workspace - not two different things
`MemorySession` **is** the workspace session - there is no meaningful distinction between the two concepts in GroundMemory. `Workspace` is the low-level filesystem handle; `MemorySession` is the high-level runtime object that wraps it together with the index and embedding provider. When you call `MemorySession.create("my-project")`, it resolves the path as `~/.groundmemory/my-project` - a single directory, not a nested one.

---

## Data Flow

```
User message
     │
     ▼
MemorySession.bootstrap()
     │  Reads MEMORY.md, USER.md, AGENTS.md, RELATIONS.md, daily logs
     │  → assembled into system prompt block
     ▼
LLM receives system prompt + tool schemas + user message
     │
     │  Model may call memory tools:
     │
     ├─► memory_read(query, ...)          ← SEARCH mode
     │       └─► provider.embed(query)                  → query vector
     │       └─► index.vector_search                    → cosine top-k
     │       └─► index.keyword_search                   → FTS5 BM25 top-k
     │       └─► merge + decay + graph expansion        → ranked SearchResult list
     │
     ├─► memory_read(file, ...)           ← GET mode
     │       └─► storage.read_file                      → raw Markdown slice (1-indexed)
     │
     ├─► memory_write(file, content)      ← APPEND mode
     │       └─► storage.write_long_term / write_daily  → appends to Markdown
     │       └─► sync.sync_file                         → chunk → embed → upsert SQLite
     │
     ├─► memory_write(file, search, content)   ← REPLACE_TEXT mode
     │       └─► is_immutable(file) check               → reject if MEMORY.md or daily/*.md
     │       └─► storage.replace_text                   → first-match replacement in Markdown
     │       └─► sync.sync_file                         → re-index
     │
     ├─► memory_write(file, start, end, content)  ← REPLACE_LINES mode
     │       └─► is_immutable(file) check               → reject if MEMORY.md or daily/*.md
     │       └─► storage.replace_lines                  → line-range replacement in Markdown
     │       └─► sync.sync_file                         → re-index
     │
     ├─► memory_write(file, start, end, content="")  ← DELETE mode
     │       └─► is_immutable(file) check               → reject if MEMORY.md or daily/*.md
     │       └─► storage.hard_delete_lines              → physically removes lines (no tombstone)
     │       └─► sync.sync_file                         → re-index
     │
     └─► memory_relate(subject, predicate, object)
             └─► relations._find_semantic_duplicate      → cosine dedup check
             └─► index.insert_relation                  → SQLite relations table
             └─► storage._atomic_write                  → append to RELATIONS.md
     │
     ▼
Agent response returned to user
     │
     ▼
Next session: session.bootstrap() reloads persisted facts
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.10+ |
| Configuration | Pydantic Settings (`.env` file + env vars) |
| Database | SQLite via `sqlite3` stdlib, WAL mode (`PRAGMA journal_mode=WAL`) |
| Full-text search | SQLite FTS5 (BM25) with auto-sync triggers |
| Vector search | NumPy cosine similarity (pure Python; no native extension required) |
| Embeddings - local | `sentence-transformers` (optional extra; not installed by default) |
| Embeddings - remote | Any OpenAI-compatible HTTP endpoint via `httpx` (core dependency) |
| HTTP client | `httpx` |
| Packaging | `hatchling` build backend (`pyproject.toml`), installable via `uv` or `pip` |
| Tests | `pytest` |

---

## Configuration

### Minimum Config

No configuration file is required. With no config, GroundMemory uses BM25-only search backed by SQLite - no API key, no GPU, no extra packages.

**Finding the example config file**

`.env.example` is in the repository root and contains every available setting with descriptions and defaults. Copy it to get started:

```bash
cp .env.example ~/.groundmemory/.env   # pip install
cp .env.example ./data/.env            # Docker
```

---

### Environment Variables

All settings are available as environment variables using the `GROUNDMEMORY_` prefix. Nested keys use double-underscore (`__`) as a separator. Environment variables take priority over `.env` file values.

**Embedding**

| Variable | Description | Default |
|---|---|---|
| `GROUNDMEMORY_EMBEDDING__PROVIDER` | `none` / `local` / `openai` | `none` |
| `GROUNDMEMORY_EMBEDDING__BASE_URL` | OpenAI-compatible endpoint URL | - |
| `GROUNDMEMORY_EMBEDDING__API_KEY` | API key for the endpoint | - |
| `GROUNDMEMORY_EMBEDDING__MODEL` | Embedding model name (provider: openai) | `text-embedding-3-small` |
| `GROUNDMEMORY_EMBEDDING__LOCAL_MODEL` | sentence-transformers model name | `all-MiniLM-L6-v2` |
| `GROUNDMEMORY_EMBEDDING__BATCH_SIZE` | Texts per embedding API call | `64` |

**Search**

| Variable | Description | Default |
|---|---|---|
| `GROUNDMEMORY_SEARCH__TOP_K` | Results returned per query | `6` |
| `GROUNDMEMORY_SEARCH__CANDIDATE_MULTIPLIER` | Oversampling factor per path | `4` |
| `GROUNDMEMORY_SEARCH__VECTOR_WEIGHT` | Vector list weight in RRF (0.0–1.0); keyword weight = 1 - vector_weight | `0.7` |
| `GROUNDMEMORY_SEARCH__RRF_K` | Reciprocal Rank Fusion k constant. Higher values flatten rank differences. Standard default is 60. | `60` |
| `GROUNDMEMORY_SEARCH__TEMPORAL_DECAY_RATE` | Score decay per day of age | `0.0` |
| `GROUNDMEMORY_SEARCH__MMR_LAMBDA` | MMR diversity (0 = disabled) | `0.0` |

**Chunking**

| Variable | Description | Default |
|---|---|---|
| `GROUNDMEMORY_CHUNKING__TOKENS` | Target chunk size in tokens | `400` |
| `GROUNDMEMORY_CHUNKING__OVERLAP` | Overlap between chunks in tokens | `80` |

**Relations**

| Variable | Description | Default |
|---|---|---|
| `GROUNDMEMORY_RELATIONS__DEDUP_THRESHOLD` | Cosine similarity threshold for dedup | `0.92` |

**Bootstrap**

| Variable | Description | Default |
|---|---|---|
| `GROUNDMEMORY_BOOTSTRAP__MAX_CHARS_PER_FILE` | Max chars per injected file before truncation | `10000` |
| `GROUNDMEMORY_BOOTSTRAP__MAX_TOTAL_CHARS` | Max total chars across all injected files | `50000` |
| `GROUNDMEMORY_BOOTSTRAP__INJECT_LONG_TERM_MEMORY` | Inject MEMORY.md | `true` |
| `GROUNDMEMORY_BOOTSTRAP__INJECT_USER_PROFILE` | Inject USER.md | `true` |
| `GROUNDMEMORY_BOOTSTRAP__INJECT_AGENTS` | Inject AGENTS.md | `true` |
| `GROUNDMEMORY_BOOTSTRAP__INJECT_DAILY_LOGS` | Enable/disable daily log injection entirely | `true` |
| `GROUNDMEMORY_BOOTSTRAP__INJECT_RELATIONS` | Inject RELATIONS.md | `true` |
| `GROUNDMEMORY_BOOTSTRAP__DAILY_LOG_DAYS` | Number of daily log files to inject counting back from today. `1` = today only, `2` = today + yesterday, `0` = none. | `1` |
| `GROUNDMEMORY_BOOTSTRAP__SYNC_MEMORY_ON_BOOTSTRAP` | Re-index all workspace files at session start. Enable when you edit memory files outside the agent so the index stays consistent with disk. | `false` |
| `GROUNDMEMORY_BOOTSTRAP__COMPACTION_TOKEN_THRESHOLD` | Token count of compactable tiers (MEMORY.md, USER.md, AGENTS.md) above which a backup is taken and a compaction notice is injected. RELATIONS.md and daily logs are excluded from this count. `0` = disabled. | `0` |
| `GROUNDMEMORY_BOOTSTRAP__COMPACTION_TOKEN_COUNTER` | Token counting method: `"approx"` (`len // 4`) or `"tiktoken"` (accurate BPE, requires `pip install groundmemory[local]`). | `"approx"` |
| `GROUNDMEMORY_BOOTSTRAP__COMPACTION_TIERS` | Memory files the agent is allowed to compact. Daily logs are never compacted. | `["MEMORY.md"]` |

**Custom Files**

| Variable | Description | Default |
|---|---|---|
| `GROUNDMEMORY_CUSTOM_FILES` | JSON array of custom file configs (see below) | `[]` |

**General**

| Variable | Description | Default |
|---|---|---|
| `GROUNDMEMORY_ROOT_DIR` | Base directory for all workspaces | `~/.groundmemory` |
| `GROUNDMEMORY_WORKSPACE` | Default workspace name | `default` |

**MCP Server**

| Variable | Description | Default |
|---|---|---|
| `GROUNDMEMORY_MCP__HOST` | Host address the server binds to. Set to `0.0.0.0` for LAN access (see [Network Access](#network-access)). | `127.0.0.1` |
| `GROUNDMEMORY_MCP__PORT` | TCP port the server listens on | `4242` |
| `GROUNDMEMORY_MCP__ALLOWED_HOSTS` | Comma-separated list of `Host:` header values to allow (DNS-rebinding protection). `localhost` and `127.0.0.1` are always allowed. Required when `HOST=0.0.0.0`. | `` |
| `GROUNDMEMORY_MCP__FORWARDED_ALLOW_IPS` | IPs uvicorn trusts to pass `X-Forwarded-For` headers. Not needed for plain LAN access - only set when a reverse proxy sits in front of GroundMemory. | `127.0.0.1` |
| `GROUNDMEMORY_MCP__API_KEY` | Static bearer token required on every request. When unset (default), no authentication is enforced. Set when exposing the server beyond localhost. Clients must send `Authorization: Bearer <token>`. | *(unset)* |

### Custom Files

Custom files extend the standard memory tier set (MEMORY.md, USER.md, AGENTS.md, RELATIONS.md, daily/) with arbitrary Markdown files you define. Each custom file is fully integrated: it appears in bootstrap injection, participates in hybrid search, can be read and written via the standard tools, and is listed by `memory_list`.

Declare custom files via the `GROUNDMEMORY_CUSTOM_FILES` environment variable (JSON array):

```env
GROUNDMEMORY_CUSTOM_FILES='[
  {"name": "RESEARCH.md", "description": "Research notes and literature findings"},
  {"name": "DECISIONS.md", "description": "Architectural decisions and their rationale", "compactable": true},
  {"name": "GLOSSARY.md", "description": "Domain terminology", "inject": false, "max_chars": 3000}
]'
```

**Per-file options:**

| Field | Type | Default | Description |
|---|---|---|---|
| `name` | string | required | Filename (e.g. `RESEARCH.md`). Flat — no subdirectories. |
| `description` | string | `""` | Short purpose text shown in tool schemas and as the bootstrap section header. Tell the agent what goes here. |
| `inject` | bool | `true` | Include file content in the bootstrap system prompt at session start. |
| `max_chars` | int or null | `null` | Max chars injected for this file. `null` uses the global `MAX_CHARS_PER_FILE` limit. |
| `searchable` | bool | `true` | Index this file for vector and keyword search. Set to `false` for reference files you only read directly. |
| `compactable` | bool | `false` | Include in `memory_compact` targets. The file will also appear in the compaction notice when memory is large. |

**Behaviour:**
- Custom files are created on first write — they do not need to exist before the agent runs.
- If `inject=true` but the file does not exist or is empty, the bootstrap section is silently skipped.
- Custom files support all four `memory_write` modes: append, replace_text, replace_lines, and delete. Unlike MEMORY.md and daily logs, they are not append-only.
- In `memory_read` SEARCH mode, pass `file="RESEARCH.md"` to scope results to that file. In GET mode, pass `file="RESEARCH.md"` to read it directly.
- Non-searchable custom files still appear in `memory_list`.

**Agent guidance:** The agent learns about custom files through two channels — the `memory_write` tool schema (which lists each file's name and description) and the bootstrap header (which shows the file's content with the description as a section title). You do not need to write extensive system prompt instructions for each file; the `description` field is sufficient for most cases.

**Backup and Restore**

GroundMemory automatically takes a zip backup of the workspace when compaction is triggered. You can also create a backup manually at any time. Backups are stored in `<workspace>/backups/YYYY-MM-DD_HHmmss.zip` and contain all Markdown files, the `daily/` directory, and `.index/memory.db`.

Use the `groundmemory` CLI to manage backups:

```bash
# Create a manual backup
groundmemory --backup

# List all backups for the current workspace
groundmemory --list-backups

# Restore the most recent backup
groundmemory --restore -1

# Restore the second-most-recent backup
groundmemory --restore -2

# Restore by exact date (error if multiple backups on that date)
groundmemory --restore 2026-04-08

# Restore by exact timestamp
groundmemory --restore 2026-04-08_165530
```

The workspace is always resolved from your environment / config (same as `groundmemory-mcp`). Set `GROUNDMEMORY_WORKSPACE` in your `.env` or as an environment variable to target a specific workspace before running these commands.

If a date matches multiple backups, the command prints the list and exits - you can then use the full timestamp to disambiguate. After restoring, restart the MCP server if it is running.

**Configuration priority (highest wins):**

```
constructor kwargs  >  environment variables  >  $GROUNDMEMORY_ROOT_DIR/.env / ./.env  >  built-in defaults
```
