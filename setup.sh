#!/bin/bash
set -e

echo "=== FuncGemma Setup ==="

# 1. Install Ollama via Homebrew
if ! command -v ollama &> /dev/null; then
    echo "[1/5] Installing Ollama..."
    brew install ollama
else
    echo "[1/5] Ollama already installed: $(ollama --version)"
fi

# 2. Start Ollama service if not running
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "[2/5] Starting Ollama service..."
    ollama serve &
    sleep 5
else
    echo "[2/5] Ollama service already running"
fi

# 3. Pull models (sequentially to avoid memory issues on 8GB)
echo "[3/5] Pulling models..."

echo "  - functiongemma (301MB)..."
ollama pull functiongemma

echo "  - gemma3:4b..."
ollama pull gemma3:4b

echo "  - qwen2.5:7b..."
ollama pull qwen2.5:7b

# 4. Create Python virtual environment
echo "[4/5] Creating Python virtual environment..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi
source .venv/bin/activate

# 5. Install Python dependencies
echo "[5/5] Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "=== Setup Complete ==="
echo "Activate venv: source .venv/bin/activate"
echo "Run conversation: python -m src.conversation"
echo ""
ollama list
