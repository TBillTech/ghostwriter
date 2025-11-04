"""
Test session bootstrap tweaks.

This file is imported automatically by Python (if found on sys.path)
very early in interpreter startup, before most imports.

We use it to force PyYAML to run in pure-Python mode to avoid
rare segfaults seen with the C extension on some systems.
"""
import os
import sys

# Ensure PyYAML pure mode before any yaml import
os.environ.setdefault("YAML_FORCE_PURE", "1")


class _BlockYamlCExt:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "yaml._yaml":
            raise ImportError("blocked yaml._yaml (pure mode)")
        return None


# Install the import hook at the very front
sys.meta_path.insert(0, _BlockYamlCExt())
