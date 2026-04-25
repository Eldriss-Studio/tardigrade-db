#!/usr/bin/env bash
# TardigradeDB setup — from zero to working memory agent.
#
# Usage:
#   ./scripts/setup.sh              # defaults (Qwen3-0.6B)
#   ./scripts/setup.sh --model Qwen/Qwen2.5-3B  # custom model
#
# What it does:
#   1. Creates a Python virtual environment
#   2. Installs tardigrade-db + dependencies
#   3. Downloads the model
#   4. Prints MCP configuration for Claude Code / Cursor

set -euo pipefail

MODEL="${TARDIGRADE_MODEL:-Qwen/Qwen3-0.6B}"
DB_PATH="${TARDIGRADE_DB_PATH:-./tardigrade-memory}"
VENV_DIR=".venv"

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --model) MODEL="$2"; shift 2 ;;
        --db-path) DB_PATH="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "================================================"
echo "  TardigradeDB Setup"
echo "================================================"
echo ""
echo "  Model:   $MODEL"
echo "  DB path: $DB_PATH"
echo ""

# 1. Virtual environment
if [ ! -d "$VENV_DIR" ]; then
    echo "[1/4] Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
else
    echo "[1/4] Virtual environment exists."
fi
source "$VENV_DIR/bin/activate"

# 2. Install dependencies
echo "[2/4] Installing dependencies..."
pip install --quiet numpy torch --index-url https://download.pytorch.org/whl/cpu
pip install --quiet transformers mcp

# Build and install tardigrade-db from source
if [ -f "crates/tdb-python/Cargo.toml" ]; then
    pip install --quiet maturin
    PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 maturin develop -m crates/tdb-python/Cargo.toml --quiet
    echo "  tardigrade-db installed from source."
else
    pip install --quiet tardigrade-db
    echo "  tardigrade-db installed from PyPI."
fi

# 3. Download model
echo "[3/4] Downloading model: $MODEL"
python3 -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
print('  Downloading tokenizer...')
AutoTokenizer.from_pretrained('$MODEL')
print('  Downloading model weights...')
AutoModelForCausalLM.from_pretrained('$MODEL')
print('  Model cached.')
"

# 4. Print configuration
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
echo ""
echo "[4/4] Setup complete!"
echo ""
echo "================================================"
echo "  Quick Start"
echo "================================================"
echo ""
echo "  # Python API:"
echo "  source $VENV_DIR/bin/activate"
echo "  PYTHONPATH=$SCRIPT_DIR/python python examples/agent_memory.py"
echo ""
echo "  # MCP Server (for Claude Code / Cursor):"
echo "  Add this to your MCP settings:"
echo ""
echo '  {'
echo '    "mcpServers": {'
echo '      "tardigrade": {'
echo "        \"command\": \"$SCRIPT_DIR/$VENV_DIR/bin/python\","
echo "        \"args\": [\"-m\", \"tardigrade_mcp\"],"
echo '        "env": {'
echo "          \"PYTHONPATH\": \"$SCRIPT_DIR/python\","
echo "          \"TARDIGRADE_DB_PATH\": \"$DB_PATH\","
echo "          \"TARDIGRADE_MODEL\": \"$MODEL\""
echo '        }'
echo '      }'
echo '    }'
echo '  }'
echo ""
echo "  See docs/guide/quickstart.md for more."
echo "================================================"
