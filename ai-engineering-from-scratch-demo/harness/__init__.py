"""Runtime harness for ai-engineering-from-scratch-demo.

The core of this package (`manifest`, `tiers`, `runner`, `coverage`) is
deliberately **stdlib-only**. It has to be, for two reasons stated in DESIGN.md:

  * D6 requires `run.py --explain` to work with zero dependencies installed, so
    a learner on a bare Python can read what a demo proves before paying for a
    `uv sync`.
  * `demo coverage` and `demo verify` must run in CI jobs that install nothing.

Only `parity` (numpy/torch) and demo `run.py` files may import third-party
packages, and `parity` imports them lazily.
"""

__version__ = "0.1.0"
