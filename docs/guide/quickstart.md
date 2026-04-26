# Quickstart: TardigradeDB in 5 Minutes

## What is TardigradeDB?

A persistent memory engine for AI agents. Store memories as KV cache tensors, retrieve them using the model's own latent space, inject them directly into generation at zero prompt token cost.

## Setup

```bash
git clone https://github.com/Eldriss-Studio/tardigrade-db.git
cd tardigrade-db
./scripts/setup.sh
```

This creates a virtual environment, installs dependencies, and downloads the default model (Qwen3-0.6B, ~600MB).

## Option A: Use with Claude Code (MCP)

The MCP server delivers memories as text in tool responses. This works with any LLM client but consumes normal prompt tokens. For zero-token KV injection, see Option B.

Add to your Claude Code MCP settings (the setup script prints the exact config):

```json
{
  "mcpServers": {
    "tardigrade": {
      "command": "/path/to/tardigrade-db/.venv/bin/python",
      "args": ["-m", "tardigrade_mcp"],
      "env": {
        "PYTHONPATH": "/path/to/tardigrade-db/python",
        "TARDIGRADE_DB_PATH": "./tardigrade-memory",
        "TARDIGRADE_MODEL": "Qwen/Qwen3-0.6B"
      }
    }
  }
}
```

Then in any conversation, the agent has memory tools:

- `tardigrade_store` — remember a fact
- `tardigrade_store_and_link` — attach a detail to an existing memory
- `tardigrade_recall` — find relevant memories
- `tardigrade_recall_with_trace` — follow links for multi-hop queries
- `tardigrade_list_links` — see connected memories
- `tardigrade_list_all` — list all stored memories
- `tardigrade_forget` — delete a memory permanently

## Option B: Use with Python

The Python API injects KV cache directly into the model's attention — zero prompt tokens for retrieved memories. Requires running a HuggingFace model locally.

```python
import torch
from tardigrade_db import Engine
from tardigrade_hooks.kp_injector import KnowledgePackStore
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-0.6B", dtype=torch.float32, attn_implementation="eager"
)
model.eval()

# Create engine + memory store
engine = Engine("./my-agent-memory")
kps = KnowledgePackStore(engine, model, tokenizer, owner=1)

# Store a memory
kps.store("User prefers morning meetings before 10am")

# Retrieve + generate (zero prompt tokens for the memory)
text, tokens, had_memory = kps.generate(
    "When should we schedule the review?",
    max_new_tokens=50, do_sample=False,
)
print(text)  # The model recalls the morning preference
```

## Linking Related Memories

When the agent learns a new detail about something it already remembers:

```python
# Store initial fact
existing = kps.store("Went to a bookstore in Pilsen")

# Later, learn the name
kps.store_and_link("The bookstore is called Casa Azul", existing)

# Query follows the link automatically
text, _, _ = kps.generate_with_trace("What is the bookstore called?")
```

## What Happens Under the Hood

1. **Store:** Text wrapped in chat template, model forward pass, KV cache extracted, Q4 quantized, persisted to disk
2. **Retrieve:** Query hidden states scored against stored memories via per-token Top5Avg
3. **Inject:** KV cache reconstructed as DynamicCache, injected into model.generate() at zero prompt tokens
4. **Trace links:** Related memories connected via durable graph edges, retrieval follows connections

## Next Steps

- [MCP Setup Guide](mcp-setup.md) — detailed configuration for different clients
- [Python API Reference](python-api.md) — full KnowledgePackStore interface
- [Concepts](concepts.md) — KV injection, trace links, governance explained
