#!/usr/bin/env python3
"""Atomically set one secret in a dotenv file without printing its value."""

from __future__ import annotations

import os
import re
import sys
import tempfile
from pathlib import Path


KEY_PATTERN = re.compile(r"^[A-Z][A-Z0-9_]*$")


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit("usage: set_secret_env.py <env-file> <KEY>")

    env_path = Path(sys.argv[1]).resolve()
    key = sys.argv[2].strip()
    if not KEY_PATTERN.fullmatch(key):
        raise SystemExit("invalid environment variable key")

    value = sys.stdin.read().strip()
    if not value or "\n" in value or "\r" in value or "\x00" in value:
        raise SystemExit("secret value must be one non-empty line")

    lines = (
        env_path.read_text(encoding="utf-8").splitlines()
        if env_path.exists()
        else []
    )
    prefix = f"{key}="
    replacement = f"{key}={value}"
    output: list[str] = []
    replaced = False
    for line in lines:
        if line.startswith(prefix):
            if not replaced:
                output.append(replacement)
                replaced = True
            continue
        output.append(line)
    if not replaced:
        output.append(replacement)

    env_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{env_path.name}.",
        dir=env_path.parent,
        text=True,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write("\n".join(output).rstrip() + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary_name, 0o600)
        os.replace(temporary_name, env_path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
