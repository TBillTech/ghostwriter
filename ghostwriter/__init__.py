"""Ghostwriter package.

Planned modularization entry point. Current pipelines are orchestrated from scripts/driver.py.
This package will gradually absorb functionality behind stable APIs.
"""

# Ensure PyYAML uses the pure-Python implementation everywhere to avoid rare
# libyaml C-extension segfaults observed under test. This must execute before
# any submodule imports `yaml`.
import os as _gw_os
_gw_os.environ.setdefault("YAML_CEXT_DISABLED", "1")

__all__ = [
    "context",
    "env",
    "llm",
    "utils",
    "validation",
    "templates",
    "artifacts",
    "resume",
    "characters",
    "mock_support",
]
