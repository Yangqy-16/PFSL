"""Exact-file SHA-256 manifest verification shared by prerequisite loaders."""

from __future__ import annotations

import hashlib
from pathlib import Path
import re


_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_manifest_target(
    manifest: Path,
    target: Path,
    *,
    code_root: Path,
    label: str,
) -> str:
    """Require one exact target entry and verify its current file content."""

    relative_target = target.resolve(strict=False).relative_to(
        code_root.resolve(strict=False)
    )
    matches: list[str] = []
    with manifest.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            fields = line.rstrip("\n").split("  ", 1)
            if len(fields) != 2 or _SHA256_PATTERN.fullmatch(fields[0]) is None:
                raise ValueError(
                    f"{label} manifest has invalid line {line_number}"
                )
            if Path(fields[1]) == relative_target:
                matches.append(fields[0])
    if len(matches) != 1:
        raise ValueError(f"{label} manifest must contain exactly one target")
    if not target.is_file() or file_sha256(target) != matches[0]:
        raise ValueError(f"{label} hash does not match manifest")
    return matches[0]
