"""
Shared configuration helpers for the Day 22 lab.
Loads environment variables from .env and provides defaults.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ── Load .env ──────────────────────────────────────────────────────────────────
dotenv_path = Path(__file__).parent / ".env"
if dotenv_path.exists():
    load_dotenv(dotenv_path)
else:
    print("⚠️  .env file not found — make sure to create it from .env.example")

# ── LangSmith ─────────────────────────────────────────────────────────────────
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY", "")
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "true").lower() == "true"
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "day22-langsmith-lab")
LANGCHAIN_ENDPOINT = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")

# ── OpenAI / Compatible endpoint ─────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# ── Derived paths ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
EVIDENCE_DIR = PROJECT_ROOT / "evidence"

DATA_DIR.mkdir(exist_ok=True)
EVIDENCE_DIR.mkdir(exist_ok=True)


def verify_config() -> bool:
    """Check that required environment variables are present."""
    missing = []
    if not LANGSMITH_API_KEY:
        missing.append("LANGSMITH_API_KEY")
    if not OPENAI_API_KEY:
        missing.append("OPENAI_API_KEY")

    if missing:
        print(f"❌ Missing required env vars: {', '.join(missing)}")
        print("   Copy .env.example to .env and fill in your keys.")
        return False

    print("✅ Config loaded successfully")
    print(f"   LangSmith project  : {LANGCHAIN_PROJECT}")
    print(f"   OpenAI endpoint    : {OPENAI_BASE_URL}")
    print(f"   Default LLM model : {OPENAI_MODEL}")
    print(f"   Embedding model    : {EMBEDDING_MODEL}")
    return True


if __name__ == "__main__":
    verify_config()
