import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

PROVIDER = os.getenv("PROVIDER", "anthropic").lower()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

MAX_TOKENS = int(os.getenv("MAX_TOKENS", "2048"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0"))

# Maximum parallel threads for the fan-out executor
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "4"))


def active_model() -> str:
    return ANTHROPIC_MODEL if PROVIDER == "anthropic" else OPENAI_MODEL


def model_tier() -> str:
    model = active_model().lower()
    high_signals = ("gpt-4", "claude", "opus", "sonnet", "o1", "o3")
    return "high" if any(s in model for s in high_signals) else "low"
