"""Runtime configuration.

Lenient with the built-in default; strict with an explicitly supplied config
path (missing -> hard error, not a silent fall-through). Portfolio rule 7.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, replace
from pathlib import Path

_DEFAULT_MAX_BYTES = 25 * 1024 * 1024


@dataclass(frozen=True)
class Settings:
    storage_dir: Path = Path("data/corpus")
    max_upload_bytes: int = _DEFAULT_MAX_BYTES

    def with_overrides(self, **kwargs: object) -> "Settings":
        return replace(self, **{k: v for k, v in kwargs.items() if v is not None})  # type: ignore[arg-type]


def _coerce(raw: dict[str, object]) -> Settings:
    max_bytes = raw.get("max_upload_bytes")
    if max_bytes is not None:
        max_bytes = int(max_bytes)
        if max_bytes <= 0:
            raise ValueError("max_upload_bytes must be positive")
    storage_dir = raw.get("storage_dir")
    return Settings().with_overrides(
        storage_dir=Path(storage_dir) if storage_dir is not None else None,
        max_upload_bytes=max_bytes,
    )


def load_settings(config_path: str | os.PathLike[str] | None = None) -> Settings:
    if config_path is not None:
        path = Path(config_path)
        if not path.is_file():
            raise FileNotFoundError(f"config file not found: {path}")
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("config file must contain a JSON object")
        return _coerce(raw)

    env: dict[str, object] = {}
    if "RAGEL_STORAGE_DIR" in os.environ:
        env["storage_dir"] = os.environ["RAGEL_STORAGE_DIR"]
    if "RAGEL_MAX_UPLOAD_BYTES" in os.environ:
        env["max_upload_bytes"] = os.environ["RAGEL_MAX_UPLOAD_BYTES"]
    return _coerce(env)
