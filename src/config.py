"""Configuration for FuncGemma system."""

# Model identifiers for Ollama
CONVERSATION_MODEL = "gemma3:4b"
ROUTER_MODEL = "functiongemma"
TEST_DATA_MODEL = "qwen2.5:7b"

# Ollama API
OLLAMA_HOST = "http://localhost:11434"

# Model parameters
CONVERSATION_OPTIONS = {
    "temperature": 0.7,
    "num_ctx": 4096,
}

ROUTER_OPTIONS = {
    "temperature": 0.0,
    "num_ctx": 4096,
}

TEST_DATA_OPTIONS = {
    "temperature": 0.8,
    "num_ctx": 8192,
}

# Memory management (keep_alive duration)
CONVERSATION_KEEP_ALIVE = "10m"
ROUTER_KEEP_ALIVE = "5m"

# Router settings
MAX_HISTORY_MESSAGES = 6  # Last 3 turns for routing context

# Test data generation
TEST_DATA_PER_FUNCTION = 100
TEST_DATA_NO_FUNCTION = 250
TEST_DATA_BATCH_SIZE = 10
