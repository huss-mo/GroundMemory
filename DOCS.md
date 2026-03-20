# OpenMemory — Documentation

This document covers integration guides, configuration reference, and environment variables.
For a project overview, see [README.md](README.md).

---

## Table of Contents

- [OpenMemory — Documentation](#openmemory--documentation)
  - [Table of Contents](#table-of-contents)
  - [Quickstart](#quickstart)
  - [Connecting to Your AI Agent](#connecting-to-your-ai-agent)
    - [OpenAI](#openai)
    - [Anthropic](#anthropic)
  - [Configuration](#configuration)
    - [Minimum Config](#minimum-config)
    - [openmemory.yaml Reference](#openmemoryyaml-reference)
    - [Environment Variables](#environment-variables)

---

## Quickstart

```python
from openmemory.session import MemorySession

# Create (or reopen) a named workspace
session = MemorySession.create("my-project")

# Write a long-term memory
session.execute_tool("memory_write", content="User prefers concise answers.", tier="long_term")

# Write a daily log entry
session.execute_tool("memory_write", content="Working on the auth service refactor.", tier="daily")

# Record a relationship between entities
session.execute_tool("memory_relate", subject="Alice", predicate="works_at", object="Acme Corp")

# Search across all memory tiers
result = session.execute_tool("memory_search", query="communication preferences")
for item in result["data"]["results"]:
    print(item["content"])

# Build the system prompt context block for your agent
system_prompt = session.bootstrap()
```

`MemorySession.create("my-project")` creates a workspace directory at `~/.openmemory/my-project/` on first run, seeding all required files. Subsequent calls reopen the same workspace.

---

## Connecting to Your AI Agent

OpenMemory exposes standard JSON schemas compatible with OpenAI function calling and Anthropic tool use. The `ALL_TOOLS` export is a list of `(schema, run)` pairs — the schemas go to the model, the run functions handle dispatch.

### OpenAI

```python
from openai import OpenAI
from openmemory.session import MemorySession
from openmemory.tools import ALL_TOOLS
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

# Dispatch tool calls the model makes back to OpenMemory
for call in response.choices[0].message.tool_calls or []:
    result = session.execute_tool(call.function.name, **json.loads(call.function.arguments))
    print(result)
```

### Anthropic

```python
from anthropic import Anthropic
from openmemory.session import MemorySession
from openmemory.tools import ALL_TOOLS

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

# Dispatch tool use blocks back to OpenMemory
for block in response.content:
    if block.type == "tool_use":
        result = session.execute_tool(block.name, **block.input)
        print(result)
```

---

## Configuration

### Minimum Config

No configuration file is required. With no config, OpenMemory uses BM25-only search backed by SQLite — no API key, no GPU, no extra packages:

```python
from openmemory.session import MemorySession

session = MemorySession.create()  # stores in ~/.openmemory/default/
```

To make this explicit in a config file:

```yaml
# openmemory.yaml
embedding:
  provider: none
```

### openmemory.yaml Reference

Place `openmemory.yaml` in your project root (or cwd). Settings here are overridden by environment variables, which in turn are overridden by constructor kwargs.

```yaml
# ---------------------------------------------------------------------------
# General
# ---------------------------------------------------------------------------

# Root directory for all workspaces (default: ~/.openmemory)
# root_dir: ~/.openmemory

# Default workspace name
# workspace: default

# ---------------------------------------------------------------------------
# Embedding provider
# ---------------------------------------------------------------------------
embedding:
  # provider options:
  #   "none"   — BM25-only, no extra deps, no API key needed
  #   "local"  — sentence-transformers (install with: pip install -e ".[local]")
  #   "openai" — any OpenAI-compatible HTTP endpoint
  provider: none

  # --- sentence-transformers (provider: local) ---
  # Any model from https://www.sbert.net/docs/pretrained_models.html
  # local_model: all-MiniLM-L6-v2      # fast, 384-dim, good general quality
  # local_model: all-mpnet-base-v2     # slower, 768-dim, higher quality

  # --- OpenAI-compatible API (provider: openai) ---
  # Supports OpenAI, Ollama, LM Studio, vLLM, LiteLLM, Mistral, Together, etc.

  # Real OpenAI:
  # base_url: ~        # leave blank to use api.openai.com
  # api_key: sk-...
  # model: text-embedding-3-small

  # Ollama (local):
  # base_url: http://localhost:11434/v1
  # api_key: ollama          # required by the client but ignored by Ollama
  # model: nomic-embed-text  # pull with: ollama pull nomic-embed-text

  # LM Studio:
  # base_url: http://localhost:1234/v1
  # api_key: lm-studio
  # model: nomic-ai/nomic-embed-text-v1.5-GGUF

  # LiteLLM proxy:
  # base_url: http://localhost:4000/v1
  # api_key: sk-...
  # model: text-embedding-3-small

  # Number of texts sent per embedding API call
  # batch_size: 64

# ---------------------------------------------------------------------------
# Hybrid search
# ---------------------------------------------------------------------------
search:
  # Number of results returned per memory_search call
  top_k: 6

  # Oversampling factor: top_k * candidate_multiplier candidates are fetched
  # from each path (vector + keyword) before merging and re-ranking
  candidate_multiplier: 4

  # Weight for vector similarity score (0.0–1.0)
  # keyword_weight = 1.0 - vector_weight
  # Set to 0.0 for pure BM25 (recommended when provider: none)
  vector_weight: 0.7

  # Temporal decay: score *= exp(-decay_rate * days_old)
  # 0.0 disables decay; 0.01 halves relevance after ~70 days
  temporal_decay_rate: 0.0

  # MMR (Maximal Marginal Relevance) diversity re-ranking
  # 0.0 = disabled (pure relevance ranking), 1.0 = maximum result diversity
  mmr_lambda: 0.0

# ---------------------------------------------------------------------------
# Text chunking
# ---------------------------------------------------------------------------
chunking:
  # Target chunk size in approximate tokens (1 token ≈ 4 chars)
  tokens: 400

  # Overlap between consecutive chunks in approximate tokens
  overlap: 80

# ---------------------------------------------------------------------------
# Relation graph
# ---------------------------------------------------------------------------
relations:
  # Cosine similarity threshold above which two relation triples are considered
  # semantic duplicates and merged. Set to 1.0 to disable semantic dedup.
  dedup_threshold: 0.92

# ---------------------------------------------------------------------------
# Bootstrap — system-prompt injection at session start
# ---------------------------------------------------------------------------
bootstrap:
  # Maximum characters per file before a truncation warning is appended
  max_chars_per_file: 20000

  # Maximum total characters injected across all files
  max_total_chars: 150000

  # Which memory files to inject into the system prompt
  inject_long_term_memory: true  # MEMORY.md
  inject_user_profile: true      # USER.md
  inject_agents: true            # AGENTS.md
  inject_daily_logs: true        # daily/YYYY-MM-DD.md (today + yesterday)
  inject_relations: true         # RELATIONS.md

# ---------------------------------------------------------------------------
# Compaction — pre-context-window-flush hooks
# ---------------------------------------------------------------------------
compaction:
  # Enable compaction detection
  enabled: true

  # Remaining tokens in the context window that cause should_compact() to return True
  soft_threshold_tokens: 4000

  # Minimum tokens always kept free for the model's response
  reserve_floor_tokens: 20000

  # Messages injected at the flush turn (override to customise wording)
  # system_prompt: "Session nearing compaction. Store durable memories now."
  # user_prompt: "Review the conversation and write lasting facts to memory. Reply DONE when finished."
```

### Environment Variables

All settings are available as environment variables using the `OPENMEMORY_` prefix. Nested keys use double-underscore (`__`) as a separator. Environment variables take priority over `openmemory.yaml`.

**Embedding**

| Variable | Description | Default |
|---|---|---|
| `OPENMEMORY_EMBEDDING__PROVIDER` | `none` / `local` / `openai` | `local` |
| `OPENMEMORY_EMBEDDING__BASE_URL` | OpenAI-compatible endpoint URL | — |
| `OPENMEMORY_EMBEDDING__API_KEY` | API key for the endpoint | — |
| `OPENMEMORY_EMBEDDING__MODEL` | Embedding model name (provider: openai) | `text-embedding-3-small` |
| `OPENMEMORY_EMBEDDING__LOCAL_MODEL` | sentence-transformers model name | `all-MiniLM-L6-v2` |
| `OPENMEMORY_EMBEDDING__BATCH_SIZE` | Texts per embedding API call | `64` |

**Search**

| Variable | Description | Default |
|---|---|---|
| `OPENMEMORY_SEARCH__TOP_K` | Results returned per query | `6` |
| `OPENMEMORY_SEARCH__CANDIDATE_MULTIPLIER` | Oversampling factor per path | `4` |
| `OPENMEMORY_SEARCH__VECTOR_WEIGHT` | Vector score weight (0.0–1.0) | `0.7` |
| `OPENMEMORY_SEARCH__TEMPORAL_DECAY_RATE` | Score decay per day of age | `0.0` |
| `OPENMEMORY_SEARCH__MMR_LAMBDA` | MMR diversity (0 = disabled) | `0.0` |

**Chunking**

| Variable | Description | Default |
|---|---|---|
| `OPENMEMORY_CHUNKING__TOKENS` | Target chunk size in tokens | `400` |
| `OPENMEMORY_CHUNKING__OVERLAP` | Overlap between chunks in tokens | `80` |

**Relations**

| Variable | Description | Default |
|---|---|---|
| `OPENMEMORY_RELATIONS__DEDUP_THRESHOLD` | Cosine similarity threshold for dedup | `0.92` |

**General**

| Variable | Description | Default |
|---|---|---|
| `OPENMEMORY_ROOT_DIR` | Base directory for all workspaces | `~/.openmemory` |
| `OPENMEMORY_WORKSPACE` | Default workspace name | `default` |

**Configuration priority (highest wins):**

```
constructor kwargs  >  environment variables  >  .env file  >  openmemory.yaml  >  built-in defaults
```

You can also use a `.env` file in your project root — all `OPENMEMORY_*` variables are picked up automatically.