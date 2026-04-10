"""Entry point — run any example by name.

Usage: uv run main.py [example_name]
"""

import subprocess
import sys
from pathlib import Path

EXAMPLES_DIR = Path(__file__).parent / "examples"

EXAMPLES = {p.stem: p for p in EXAMPLES_DIR.glob("*.py") if p.stem != "__init__"}


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "chat"

    if name not in EXAMPLES:
        print(f"Unknown example: {name}")
        print(f"Available: {', '.join(sorted(EXAMPLES))}")
        sys.exit(1)

    subprocess.run([sys.executable, str(EXAMPLES[name])])


if __name__ == "__main__":
    main()
