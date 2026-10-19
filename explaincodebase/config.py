"""Load ``KEY=value`` settings from a ``.env`` file without overriding the real environment."""

from __future__ import annotations

import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_env_file(paths: tuple[str, ...] | None = None) -> str | None:
    """Apply the first ``.env`` found; variables already set in the environment win."""
    candidates = paths or (os.path.join(os.getcwd(), ".env"), os.path.join(PROJECT_ROOT, ".env"))
    for path in candidates:
        if not os.path.isfile(path):
            continue
        with open(path, encoding="utf-8") as handle:
            for raw in handle:
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                if line.startswith("export "):
                    line = line[len("export "):]
                key, value = line.split("=", 1)
                key, value = key.strip(), value.strip()
                if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
                    value = value[1:-1]
                if key and key not in os.environ:
                    os.environ[key] = value
        return path
    return None
