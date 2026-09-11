"""Strict deterministic and atomic I/O for the ProTrek SAE feature atlas."""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import tempfile
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import BinaryIO, TextIO

import torch


class AtlasIOError(ValueError):
    """Raised when an atlas input or output violates the strict I/O contract."""


def _reject_symlink_components(path: Path) -> None:
    path = Path(path)
    if path.is_absolute():
        current = Path(path.anchor)
        raw_components = path.parts[1:]
    else:
        current = Path.cwd()
        raw_components = path.parts
    for component in raw_components:
        if component in ("", "."):
            continue
        if component == "..":
            current = current.parent
            continue
        current = current / component
        try:
            metadata = os.lstat(current)
        except (FileNotFoundError, NotADirectoryError):
            continue
        except OSError as error:
            raise AtlasIOError(f"Cannot inspect path component: {current}") from error
        if stat.S_ISLNK(metadata.st_mode):
            raise AtlasIOError(f"Symlink path component is not allowed: {current}")


def _require_regular_file(path: Path) -> Path:
    path = Path(path)
    _reject_symlink_components(path)
    try:
        if not path.is_file():
            raise AtlasIOError(f"Input is not a regular file: {path}")
    except OSError as error:
        raise AtlasIOError(f"Cannot inspect input file: {path}") from error
    return path


def _prepare_output(path: Path) -> Path:
    path = Path(path)
    _reject_symlink_components(path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise AtlasIOError(f"Cannot create output directory: {path.parent}") from error
    _reject_symlink_components(path)
    if not path.parent.is_dir():
        raise AtlasIOError(f"Output parent is not a directory: {path.parent}")
    if path.exists() and not path.is_file():
        raise AtlasIOError(f"Output target is not a regular file: {path}")
    return path


def sha256_file(path: Path) -> str:
    """Return the SHA-256 checksum of one non-symlink regular file."""

    path = _require_regular_file(path)
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise AtlasIOError(f"Cannot read input file: {path}") from error
    return digest.hexdigest()


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    keys = [key for key, _ in pairs]
    if len(keys) != len(set(keys)):
        duplicates = sorted(key for key in set(keys) if keys.count(key) > 1)
        raise AtlasIOError(f"JSON object contains duplicate key: {duplicates[0]}")
    if keys != sorted(keys):
        raise AtlasIOError("JSON object keys are not sorted")
    return dict(pairs)


def _reject_constant(value: str) -> object:
    raise AtlasIOError(f"Invalid non-finite JSON constant: {value}")


def _parse_finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise AtlasIOError(f"JSON number is not finite: {value}")
    return parsed


def _strict_json_text(text: str, path: Path, label: str) -> dict[str, object]:
    try:
        payload = json.loads(
            text,
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
            parse_float=_parse_finite_float,
        )
    except AtlasIOError:
        raise
    except (json.JSONDecodeError, UnicodeError, TypeError, ValueError) as error:
        raise AtlasIOError(f"Invalid {label}: {path}") from error
    if not isinstance(payload, dict):
        raise AtlasIOError(f"{label} must contain a JSON object: {path}")
    return payload


def strict_json_load(path: Path) -> dict[str, object]:
    """Load one sorted, duplicate-free, finite JSON object."""

    path = _require_regular_file(path)
    try:
        data = path.read_bytes()
        if not data.endswith(b"\n") or data.endswith(b"\n\n"):
            raise AtlasIOError(f"JSON must end with exactly one newline: {path}")
        text = data.decode("utf-8")
    except AtlasIOError:
        raise
    except (OSError, UnicodeError) as error:
        raise AtlasIOError(f"Cannot read JSON: {path}") from error
    return _strict_json_text(text, path, "JSON")


def strict_json_lines(path: Path) -> Iterator[dict[str, object]]:
    """Yield sorted, duplicate-free, finite JSON objects from strict JSONL."""

    path = _require_regular_file(path)
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            line_number = 0
            saw_row = False
            while True:
                line = handle.readline()
                if line == "":
                    if not saw_row:
                        raise AtlasIOError(f"JSONL must contain at least one row: {path}")
                    raise AtlasIOError(f"JSONL must end with exactly one newline: {path}")
                line_number += 1
                if line == "\n":
                    raise AtlasIOError(f"Blank JSONL line {line_number}: {path}")
                if not line.endswith("\n"):
                    raise AtlasIOError(f"JSONL must end with exactly one newline: {path}")
                payload = line[:-1]
                if not payload:
                    raise AtlasIOError(f"Blank JSONL line {line_number}: {path}")
                try:
                    next_char = handle.read(1)
                except UnicodeError as error:
                    raise AtlasIOError(f"Cannot read JSONL: {path}") from error
                if next_char == "":
                    try:
                        yield _strict_json_text(payload, path, f"JSONL line {line_number}")
                    except AtlasIOError as error:
                        raise AtlasIOError(f"Invalid JSONL line {line_number}: {path}: {error}") from error
                    return
                saw_row = True
                handle.seek(handle.tell() - 1)
                try:
                    yield _strict_json_text(payload, path, f"JSONL line {line_number}")
                except AtlasIOError as error:
                    raise AtlasIOError(f"Invalid JSONL line {line_number}: {path}: {error}") from error
    except AtlasIOError:
        raise
    except (OSError, UnicodeError) as error:
        raise AtlasIOError(f"Cannot read JSONL: {path}") from error


def _validate_json_value(value: object, location: str = "payload") -> None:
    value_type = type(value)
    if value is None or value_type in (str, bool, int):
        return
    if value_type is float:
        if not math.isfinite(value):
            raise AtlasIOError(f"Non-finite float in {location}")
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            if type(key) is not str:
                raise AtlasIOError(f"JSON object key in {location} must be a built-in str")
            _validate_json_value(child, f"{location}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _validate_json_value(child, f"{location}[{index}]")
        return
    raise AtlasIOError(
        f"Unsupported non-built-in JSON value in {location}: {value_type.__name__}"
    )


def _atomic_write(
    path: Path,
    mode: str,
    writer: object,
) -> None:
    path = _prepare_output(path)
    descriptor: int | None = None
    temporary_path: Path | None = None
    try:
        descriptor, raw_path = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        temporary_path = Path(raw_path)
        if mode == "wb":
            handle: BinaryIO | TextIO = os.fdopen(descriptor, "wb")
        else:
            handle = os.fdopen(descriptor, "w", encoding="utf-8", newline="\n")
        descriptor = None
        with handle:
            writer(handle)  # type: ignore[operator]
            handle.flush()
            os.fsync(handle.fileno())
        _reject_symlink_components(path)
        os.replace(temporary_path, path)
        temporary_path = None
    except AtlasIOError:
        raise
    except Exception as error:
        raise AtlasIOError(f"Atomic write failed for {path}: {error}") from error
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass


def _json_bytes(payload: object) -> str:
    _validate_json_value(payload)
    try:
        return json.dumps(
            payload,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError, OSError) as error:
        raise AtlasIOError(f"Cannot serialize strict JSON: {error}") from error


def write_json_atomic(path: Path, payload: object) -> None:
    """Atomically publish one deterministic JSON value."""

    def writer(handle: TextIO) -> None:
        handle.write(_json_bytes(payload))
        handle.write("\n")

    _atomic_write(path, "w", writer)


def write_jsonl_atomic(
    path: Path, rows: Iterable[Mapping[str, object]]
) -> None:
    """Atomically publish deterministic strict JSONL objects."""

    def writer(handle: TextIO) -> None:
        wrote_row = False
        for row_number, row in enumerate(rows, start=1):
            if not isinstance(row, Mapping):
                raise AtlasIOError(f"JSONL row {row_number} must be a mapping")
            handle.write(_json_bytes(row))
            handle.write("\n")
            wrote_row = True
        if not wrote_row:
            raise AtlasIOError("JSONL output must contain at least one row")

    _atomic_write(path, "w", writer)


def write_torch_atomic(path: Path, payload: object) -> None:
    """Atomically publish a PyTorch artifact through a sibling file."""

    def writer(handle: BinaryIO) -> None:
        torch.save(payload, handle)

    _atomic_write(path, "wb", writer)


def file_metadata(path: Path) -> dict[str, object]:
    """Return absolute path, byte size, and checksum for a regular file."""

    path = _require_regular_file(path)
    try:
        size = path.stat().st_size
    except OSError as error:
        raise AtlasIOError(f"Cannot stat input file: {path}") from error
    return {
        "path": str(path.resolve()),
        "bytes": size,
        "sha256": sha256_file(path),
    }
