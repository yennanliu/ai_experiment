"""Simple chat — basic conversation with GPT-4o-mini."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core import make_chain, chat_loop

chain = make_chain()

if __name__ == "__main__":
    chat_loop(chain)
