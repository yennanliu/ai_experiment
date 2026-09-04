"""Zero-dependency harness for the ai-engineering-from-scratch solution repo.

Everything in this package imports only the standard library, so `demo list`,
`demo coverage` and `--explain` work on a bare Python with nothing installed
(`DESIGN §4`). Solutions may use their phase's `deps_group`; the harness may not.
"""

__all__ = ["cassette", "coverage", "explain", "manifest", "parity", "practice",
           "runner", "tiers", "yamlite"]
__version__ = "0.1.0"
