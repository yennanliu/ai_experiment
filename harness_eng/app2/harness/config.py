import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

PROVIDER = os.getenv("PROVIDER", "anthropic").lower()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Raise CODE_MAX_TOKENS when generating large source files
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "2048"))
CODE_MAX_TOKENS = int(os.getenv("CODE_MAX_TOKENS", "4096"))

# temperature=0 maximises JSON/tool-call reliability on weaker models
TEMPERATURE = float(os.getenv("TEMPERATURE", "0"))


def active_model() -> str:
    return ANTHROPIC_MODEL if PROVIDER == "anthropic" else OPENAI_MODEL


def model_tier() -> str:
    """
    Rough capability tier — 'high' for GPT-4-class / Claude models,
    'low' for anything older (gpt-3.5, o3-mini, etc.).
    Callers can use this to log a warning or adjust prompts.
    """
    model = active_model().lower()
    high_signals = ("gpt-4", "claude", "opus", "sonnet", "o1", "o3")
    return "high" if any(s in model for s in high_signals) else "low"
