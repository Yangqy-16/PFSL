"""Reproducible acquisition and validation of official annotation snapshots."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import re
import shutil
import stat
import sys
import tempfile
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import BinaryIO

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pfsl_process.sae.feature_atlas_io import (
    AtlasIOError,
    sha256_file,
    strict_json_load,
    write_json_atomic,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "data/sae_analysis/protrek_final_1024/topk_w8_k32_seed42/feature_atlas_v1/external"
)
DOWNLOAD_USER_AGENT = "ProTrek-SAE-Feature-Atlas/1.0"


class SnapshotError(ValueError):
    """Raised when an external snapshot cannot be acquired or validated."""


class _SnapshotRecoveryError(SnapshotError):
    def __init__(
        self, backup_dir: Path, stage_dir: Path, rollback_errors: Sequence[str]
    ) -> None:
        self.backup_dir = backup_dir
        self.stage_dir = stage_dir
        details = "; ".join(rollback_errors)
        super().__init__(
            "Snapshot recovery required after incomplete rollback; "
            f"backup recovery path: {backup_dir}; "
            f"staging recovery path: {stage_dir}; errors: {details}"
        )


@dataclass(frozen=True)
class SnapshotSpec:
    database: str
    name: str
    url: str
    relative_path: str
    license_name: str
    release_parser: str


SNAPSHOT_SPECS = (
    SnapshotSpec(
        "go",
        "human_gaf",
        "https://ftp.ebi.ac.uk/pub/databases/GO/goa/HUMAN/goa_human.gaf.gz",
        "go/goa_human.gaf.gz",
        "CC BY 4.0",
        "gaf_header",
    ),
    SnapshotSpec(
        "go",
        "go_basic",
        "https://current.geneontology.org/ontology/go-basic.obo",
        "go/go-basic.obo",
        "CC BY 4.0",
        "obo_header",
    ),
    SnapshotSpec(
        "reactome",
        "database_version",
        "https://reactome.org/ContentService/data/database/version",
        "reactome/database_version.json",
        "CC BY 4.0",
        "reactome_json",
    ),
    SnapshotSpec(
        "reactome",
        "uniprot_leaf",
        "https://reactome.org/download/current/UniProt2Reactome.txt",
        "reactome/UniProt2Reactome.txt",
        "CC BY 4.0",
        "reactome_companion",
    ),
    SnapshotSpec(
        "reactome",
        "uniprot_all",
        "https://reactome.org/download/current/UniProt2Reactome_All_Levels.txt",
        "reactome/UniProt2Reactome_All_Levels.txt",
        "CC BY 4.0",
        "reactome_companion",
    ),
    SnapshotSpec(
        "reactome",
        "pathways",
        "https://reactome.org/download/current/ReactomePathways.txt",
        "reactome/ReactomePathways.txt",
        "CC BY 4.0",
        "reactome_companion",
    ),
    SnapshotSpec(
        "reactome",
        "relations",
        "https://reactome.org/download/current/ReactomePathwaysRelation.txt",
        "reactome/ReactomePathwaysRelation.txt",
        "CC BY 4.0",
        "reactome_companion",
    ),
    SnapshotSpec(
        "interpro",
        "protein2ipr",
        "https://ftp.ebi.ac.uk/pub/databases/interpro/releases/109.0/protein2ipr.dat.gz",
        "interpro/protein2ipr.dat.gz",
        "Apache 2.0",
        "interpro_resolved_url",
    ),
    SnapshotSpec(
        "interpro",
        "entry_list",
        "https://ftp.ebi.ac.uk/pub/databases/interpro/releases/109.0/entry.list",
        "interpro/entry.list",
        "Apache 2.0",
        "interpro_resolved_url",
    ),
    SnapshotSpec(
        "corum",
        "database_version",
        "https://mips.helmholtz-muenchen.de/fastapi-corum/public/releases/current",
        "corum/database_version.json",
        "CC BY-NC 4.0",
        "corum_json",
    ),
    SnapshotSpec(
        "corum",
        "all_complexes",
        "https://mips.helmholtz-muenchen.de/fastapi-corum/public/file/download_current_file?file_id=complete&file_format=txt",
        "corum/allComplexes.txt",
        "CC BY-NC 4.0",
        "corum_companion",
    ),
)


_RELEASE_PARSERS_BY_DATABASE = {
    "go": frozenset({"gaf_header", "obo_header"}),
    "reactome": frozenset({"reactome_json", "reactome_companion"}),
    "interpro": frozenset({"interpro_resolved_url"}),
    "corum": frozenset({"corum_json", "corum_companion"}),
}
_RELEASE_PARSERS = frozenset().union(*_RELEASE_PARSERS_BY_DATABASE.values())
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
_OpenUrl = Callable[[str, float], BinaryIO]


def _open_url(url: str, timeout: float) -> BinaryIO:
    request = urllib.request.Request(
        url, headers={"User-Agent": DOWNLOAD_USER_AGENT}
    )
    return urllib.request.urlopen(request, timeout=timeout)  # type: ignore[return-value]


def _validate_specs(specs: Sequence[SnapshotSpec]) -> tuple[SnapshotSpec, ...]:
    if isinstance(specs, (str, bytes)) or not isinstance(specs, Sequence):
        raise TypeError("specs must be a sequence of SnapshotSpec values")
    result = tuple(specs)
    if not result:
        raise SnapshotError("At least one snapshot spec is required")
    identities: set[tuple[str, str]] = set()
    paths: set[str] = set()
    for index, spec in enumerate(result):
        if type(spec) is not SnapshotSpec:
            raise TypeError(f"specs[{index}] must be an exact SnapshotSpec")
        for field_name in (
            "database",
            "name",
            "url",
            "relative_path",
            "license_name",
            "release_parser",
        ):
            value = getattr(spec, field_name)
            if type(value) is not str:
                raise TypeError(f"SnapshotSpec.{field_name} must be a built-in str")
            if not value.strip():
                raise SnapshotError(f"SnapshotSpec.{field_name} must not be empty")
        relative = PurePosixPath(spec.relative_path)
        if relative.is_absolute() or ".." in relative.parts or "." in relative.parts:
            raise SnapshotError(f"Unsafe snapshot relative path: {spec.relative_path}")
        if str(relative) != spec.relative_path:
            raise SnapshotError(f"Snapshot path must use canonical POSIX syntax: {spec.relative_path}")
        if spec.release_parser not in _RELEASE_PARSERS:
            raise SnapshotError(f"Unknown release parser: {spec.release_parser}")
        if spec.release_parser not in _RELEASE_PARSERS_BY_DATABASE.get(
            spec.database, frozenset()
        ):
            raise SnapshotError(
                f"Release parser {spec.release_parser} is not allowed for database "
                f"{spec.database}"
            )
        identity = (spec.database, spec.name)
        if identity in identities:
            raise SnapshotError(f"Duplicate snapshot spec identity: {identity}")
        if spec.relative_path in paths:
            raise SnapshotError(f"Duplicate snapshot relative path: {spec.relative_path}")
        identities.add(identity)
        paths.add(spec.relative_path)
    return result


def _specs_sha256(specs: Sequence[SnapshotSpec]) -> str:
    payload = json.dumps(
        [asdict(spec) for spec in specs],
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _reject_symlink_components(path: Path) -> Path:
    absolute = Path(os.path.abspath(path))
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current = current / component
        try:
            metadata = os.lstat(current)
        except (FileNotFoundError, NotADirectoryError):
            continue
        except OSError as error:
            raise SnapshotError(f"Cannot inspect snapshot path: {current}") from error
        if stat.S_ISLNK(metadata.st_mode):
            raise SnapshotError(f"Symlink snapshot path is not allowed: {current}")
    return absolute


def _safe_registered_path(output_dir: Path, relative_path: str) -> Path:
    relative = PurePosixPath(relative_path)
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or not relative.parts
        or relative.parts[0] != "raw"
    ):
        raise SnapshotError(f"Unsafe registered raw path: {relative_path}")
    output_absolute = _reject_symlink_components(output_dir)
    candidate = output_absolute.joinpath(*relative.parts)
    current = output_absolute
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise SnapshotError(f"Symlink snapshot path is not allowed: {current}")
    return candidate


def _stream_one(
    spec: SnapshotSpec,
    destination: Path,
    timeout: float,
    retries: int,
    opener: _OpenUrl,
) -> tuple[Path, str, int, str]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    last_error: BaseException | None = None
    for _attempt in range(1, retries + 1):
        descriptor: int | None = None
        temporary: Path | None = None
        response: BinaryIO | None = None
        try:
            descriptor, raw_temporary = tempfile.mkstemp(
                prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
            )
            temporary = Path(raw_temporary)
            response = opener(spec.url, timeout)
            resolved_url = str(getattr(response, "geturl", lambda: spec.url)())
            if not resolved_url:
                raise SnapshotError(f"Empty resolved URL for {spec.name}")
            digest = hashlib.sha256()
            byte_count = 0
            with os.fdopen(descriptor, "wb") as handle:
                descriptor = None
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    if type(chunk) is not bytes:
                        raise SnapshotError(f"Non-byte response for {spec.name}")
                    handle.write(chunk)
                    digest.update(chunk)
                    byte_count += len(chunk)
                if byte_count == 0:
                    raise SnapshotError(f"zero-byte response for {spec.name}")
                handle.flush()
                os.fsync(handle.fileno())
            return temporary, resolved_url, byte_count, digest.hexdigest()
        except BaseException as error:
            if isinstance(error, (KeyboardInterrupt, SystemExit)):
                raise
            last_error = error
            if descriptor is not None:
                try:
                    os.close(descriptor)
                except OSError:
                    pass
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass
        finally:
            if response is not None:
                try:
                    response.close()
                except OSError:
                    pass
    detail = str(last_error) if last_error is not None else "unknown error"
    raise SnapshotError(
        f"Download failed for {spec.name} after {retries} attempts: {detail}"
    ) from last_error


def _unique_match(text: str, pattern: re.Pattern[str], label: str) -> str:
    values = {match.strip() for match in pattern.findall(text) if match.strip()}
    if len(values) != 1:
        raise SnapshotError(f"Missing or ambiguous release metadata in {label}")
    return next(iter(values))


def _parse_gaf_release(path: Path) -> str:
    try:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            header = "".join(line for _, line in zip(range(500), handle) if line.startswith("!"))
    except (OSError, UnicodeError, EOFError) as error:
        raise SnapshotError(f"Cannot read GAF release header: {path}") from error
    return _unique_match(
        header,
        re.compile(
            r"^!\s*date-generated\s*:\s*"
            r"(20\d{2}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01]))"
            r"(?: (?:[01]\d|2[0-3]):[0-5]\d(?::[0-5]\d)?)?\s*$",
            re.MULTILINE | re.IGNORECASE,
        ),
        "GAF header",
    )


def _parse_obo_release(path: Path) -> str:
    try:
        with path.open("rt", encoding="utf-8") as handle:
            header_lines = []
            for line in handle:
                if line.startswith("["):
                    break
                header_lines.append(line)
    except (OSError, UnicodeError) as error:
        raise SnapshotError(f"Cannot read OBO release header: {path}") from error
    value = _unique_match(
        "".join(header_lines),
        re.compile(r"^data-version\s*:\s*(\S+)\s*$", re.MULTILINE | re.IGNORECASE),
        "OBO header",
    )
    return value.rstrip("/").rsplit("/", 1)[-1]


def _release_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    keys = [key for key, _ in pairs]
    if len(keys) != len(set(keys)):
        duplicate = sorted(key for key in set(keys) if keys.count(key) > 1)[0]
        raise SnapshotError(f"Release JSON contains duplicate key: {duplicate}")
    return dict(pairs)


def _reject_release_json_constant(value: str) -> object:
    raise SnapshotError(f"Release JSON contains non-finite constant: {value}")


def _parse_release_json_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise SnapshotError(f"Release JSON contains non-finite number: {value}")
    return parsed


def _load_release_json(path: Path, database: str) -> object:
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_release_json_object,
            parse_constant=_reject_release_json_constant,
            parse_float=_parse_release_json_float,
        )
    except SnapshotError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SnapshotError(f"Invalid {database} version response: {path}") from error


def _parse_reactome_release(path: Path) -> str:
    payload = _load_release_json(path, "Reactome")
    candidates: list[object]
    if type(payload) in (str, int):
        candidates = [payload]
    elif type(payload) is dict:
        candidates = [
            payload[key]
            for key in ("release", "version", "databaseVersion", "database_version")
            if key in payload
        ]
    else:
        candidates = []
    values = {
        str(value).strip()
        for value in candidates
        if type(value) in (str, int) and str(value).strip()
    }
    if len(values) != 1:
        raise SnapshotError("Missing or ambiguous release metadata in Reactome JSON")
    return next(iter(values))


def _parse_interpro_release(resolved_url: str) -> str:
    patterns = (
        re.compile(r"(?:release|releases|interpro)[-_/](\d+(?:\.\d+)+)(?:/|\b)", re.IGNORECASE),
        re.compile(r"/(\d+\.\d+(?:\.\d+)*)/"),
    )
    values: set[str] = set()
    for pattern in patterns:
        values.update(pattern.findall(resolved_url))
    if len(values) != 1:
        raise SnapshotError("Missing or ambiguous release metadata in InterPro resolved URL")
    return next(iter(values))


def _parse_corum_release(path: Path) -> str:
    payload = _load_release_json(path, "CORUM")
    if type(payload) in (str, int, float):
        candidates = [payload]
    elif type(payload) is dict:
        candidates = [
            payload[key]
            for key in ("release", "version", "databaseVersion", "database_version")
            if key in payload
        ]
    else:
        candidates = []
    values: set[str] = set()
    for value in candidates:
        if type(value) not in (str, int, float):
            raise SnapshotError("CORUM release metadata must unambiguously identify 5.3")
        normalized = str(value).strip()
        if not normalized or (type(value) is float and not math.isfinite(value)):
            raise SnapshotError("CORUM release metadata must unambiguously identify 5.3")
        values.add(normalized)
    if values != {"5.3"}:
        raise SnapshotError("CORUM release metadata must unambiguously identify 5.3")
    return "5.3"


def _normalize_go_release(value: str) -> str:
    matches = {
        f"{year}-{month}-{day}"
        for year, month, day in re.findall(
            r"(?<!\d)(20\d{2})-?(0[1-9]|1[0-2])-?([0-2]\d|3[01])(?!\d)", value
        )
    }
    if len(matches) != 1:
        raise SnapshotError(f"Missing or ambiguous GO release date: {value}")
    return next(iter(matches))


def _resolve_releases(
    specs: Sequence[SnapshotSpec], downloads: Sequence[dict[str, object]]
) -> list[str]:
    reactome_versions = [
        _parse_reactome_release(Path(str(row["temporary_path"])))
        for spec, row in zip(specs, downloads)
        if spec.release_parser == "reactome_json"
    ]
    if any(spec.release_parser == "reactome_companion" for spec in specs) and len(reactome_versions) != 1:
        raise SnapshotError("Reactome companion files require exactly one version response")
    corum_versions = [
        _parse_corum_release(Path(str(row["temporary_path"])))
        for spec, row in zip(specs, downloads)
        if spec.release_parser == "corum_json"
    ]
    if any(spec.release_parser == "corum_companion" for spec in specs) and len(corum_versions) != 1:
        raise SnapshotError("CORUM companion files require exactly one version response")
    releases: list[str] = []
    for spec, row in zip(specs, downloads):
        path = Path(str(row["temporary_path"]))
        if spec.release_parser == "gaf_header":
            release = _normalize_go_release(_parse_gaf_release(path))
        elif spec.release_parser == "obo_header":
            release = _normalize_go_release(_parse_obo_release(path))
        elif spec.release_parser == "reactome_json":
            release = _parse_reactome_release(path)
        elif spec.release_parser == "reactome_companion":
            release = reactome_versions[0]
        elif spec.release_parser == "interpro_resolved_url":
            release = _parse_interpro_release(str(row["resolved_url"]))
        elif spec.release_parser == "corum_json":
            release = _parse_corum_release(path)
        elif spec.release_parser == "corum_companion":
            release = corum_versions[0]
        else:  # guarded by _validate_specs
            raise SnapshotError(f"Unknown release parser: {spec.release_parser}")
        if not release:
            raise SnapshotError(f"Missing release metadata for {spec.name}")
        releases.append(release)
    interpro_releases = {
        release
        for spec, release in zip(specs, releases)
        if spec.database == "interpro"
    }
    if len(interpro_releases) > 1:
        raise SnapshotError("InterPro files resolved to ambiguous releases")
    gaf_releases = {
        release
        for spec, release in zip(specs, releases)
        if spec.release_parser == "gaf_header"
    }
    obo_releases = {
        release
        for spec, release in zip(specs, releases)
        if spec.release_parser == "obo_header"
    }
    if (
        len(gaf_releases) > 1
        or len(obo_releases) > 1
    ):
        raise SnapshotError("GO snapshot files contain ambiguous release metadata")
    return releases


def _registered_relative_paths(manifest: dict[str, object]) -> list[str]:
    files = manifest.get("files")
    if type(files) is not list:
        raise SnapshotError("Snapshot manifest files must be a list")
    paths: list[str] = []
    for row in files:
        if type(row) is not dict or type(row.get("relative_path")) is not str:
            raise SnapshotError("Invalid registered snapshot file")
        paths.append(row["relative_path"])
    return paths


def _prune_empty_raw_directories(output_dir: Path) -> None:
    raw_dir = output_dir / "raw"
    if not raw_dir.is_dir() or raw_dir.is_symlink():
        return
    directories = sorted(
        (path for path in raw_dir.rglob("*") if path.is_dir() and not path.is_symlink()),
        key=lambda path: len(path.parts),
        reverse=True,
    )
    for path in directories + [raw_dir]:
        try:
            path.rmdir()
        except OSError:
            pass


def _acquire_into_empty_stage(
    stage_dir: Path,
    specs: Sequence[SnapshotSpec],
    timeout: float,
    retries: int,
    opener: _OpenUrl,
) -> dict[str, object]:
    downloads: list[dict[str, object]] = []
    published: list[Path] = []
    manifest_path = stage_dir / "manifest.json"
    manifest_published = False
    completed = False
    try:
        for spec in specs:
            registered_relative = str(PurePosixPath("raw") / spec.relative_path)
            destination = _safe_registered_path(stage_dir, registered_relative)
            temporary, resolved_url, byte_count, sha256 = _stream_one(
                spec, destination, timeout, retries, opener
            )
            downloads.append(
                {
                    "destination": destination,
                    "temporary_path": temporary,
                    "resolved_url": resolved_url,
                    "bytes": byte_count,
                    "sha256": sha256,
                }
            )
        releases = _resolve_releases(specs, downloads)
        for row in downloads:
            destination = Path(str(row["destination"]))
            temporary = Path(str(row["temporary_path"]))
            os.replace(temporary, destination)
            published.append(destination)
            row["temporary_path"] = None

        file_records = []
        for spec, row, release in zip(specs, downloads, releases):
            file_records.append(
                {
                    "bytes": row["bytes"],
                    "database": spec.database,
                    "license": spec.license_name,
                    "name": spec.name,
                    "relative_path": str(PurePosixPath("raw") / spec.relative_path),
                    "release": release,
                    "release_parser": spec.release_parser,
                    "requested_url": spec.url,
                    "resolved_url": row["resolved_url"],
                    "sha256": row["sha256"],
                }
            )
        manifest: dict[str, object] = {
            "databases": sorted({spec.database for spec in specs}),
            "files": file_records,
            "retrieved_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "schema_version": 1,
            "specs_sha256": _specs_sha256(specs),
        }
        write_json_atomic(manifest_path, manifest)
        manifest_published = True
        validated = validate_external_snapshots(stage_dir)
        completed = True
        return validated
    except SnapshotError:
        raise
    except (OSError, AtlasIOError) as error:
        raise SnapshotError(f"Atomic snapshot publication failed: {error}") from error
    finally:
        for row in downloads:
            raw_path = row.get("temporary_path")
            if raw_path is not None:
                try:
                    Path(str(raw_path)).unlink(missing_ok=True)
                except OSError:
                    pass
        if not completed:
            if manifest_published:
                try:
                    manifest_path.unlink(missing_ok=True)
                except OSError:
                    pass
            for path in reversed(published):
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass
            _prune_empty_raw_directories(stage_dir)


def _commit_staged_snapshot(
    output_dir: Path,
    stage_dir: Path,
    staged_manifest: dict[str, object],
    previous_manifest: dict[str, object] | None,
) -> dict[str, object]:
    new_relative_paths = _registered_relative_paths(staged_manifest)
    old_relative_paths = (
        _registered_relative_paths(previous_manifest)
        if previous_manifest is not None
        else []
    )
    old_destinations = {
        _safe_registered_path(output_dir, relative_path)
        for relative_path in old_relative_paths
    }
    new_pairs: list[tuple[Path, Path]] = []
    for relative_path in new_relative_paths:
        staged_path = _safe_registered_path(stage_dir, relative_path)
        destination = _safe_registered_path(output_dir, relative_path)
        if (destination.exists() or destination.is_symlink()) and destination not in old_destinations:
            raise SnapshotError(f"Refusing to replace existing unregistered output: {destination}")
        new_pairs.append((staged_path, destination))

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    backup_dir = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.snapshot-backup-", dir=output_dir.parent)
    )
    old_pairs: list[tuple[Path, Path]] = []
    if previous_manifest is not None:
        for relative_path in old_relative_paths:
            source = _safe_registered_path(output_dir, relative_path)
            backup = backup_dir.joinpath(*PurePosixPath(relative_path).parts)
            old_pairs.append((backup, source))
        old_pairs.append((backup_dir / "manifest.json", output_dir / "manifest.json"))
    staged_manifest_path = stage_dir / "manifest.json"
    destination_manifest_path = output_dir / "manifest.json"
    validated: dict[str, object]
    try:
        for backup, source in old_pairs:
            backup.parent.mkdir(parents=True, exist_ok=True)
            os.replace(source, backup)

        for staged_path, destination in new_pairs:
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staged_path, destination)
        os.replace(staged_manifest_path, destination_manifest_path)

        validated = validate_external_snapshots(output_dir)
        _prune_empty_raw_directories(output_dir)
    except BaseException as error:
        rollback_errors: list[str] = []
        new_recovery_pairs = [
            (destination_manifest_path, staged_manifest_path),
            *((destination, staged_path) for staged_path, destination in reversed(new_pairs)),
        ]
        for destination, staged_path in new_recovery_pairs:
            try:
                if (
                    (destination.exists() or destination.is_symlink())
                    and not (staged_path.exists() or staged_path.is_symlink())
                ):
                    staged_path.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(destination, staged_path)
            except BaseException as rollback_error:
                rollback_errors.append(
                    f"cannot recover staged output {destination} -> {staged_path}: {rollback_error}"
                )
        for backup, destination in old_pairs:
            try:
                if backup.exists() or backup.is_symlink():
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(backup, destination)
            except BaseException as rollback_error:
                rollback_errors.append(
                    f"cannot restore previous output {backup} -> {destination}: {rollback_error}"
                )
        _prune_empty_raw_directories(output_dir)
        if rollback_errors:
            raise _SnapshotRecoveryError(
                backup_dir, stage_dir, rollback_errors
            ) from error
        try:
            shutil.rmtree(backup_dir)
        except OSError:
            pass
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            raise
        if isinstance(error, SnapshotError):
            raise
        raise SnapshotError(f"Atomic snapshot publication failed: {error}") from error

    try:
        shutil.rmtree(backup_dir)
    except OSError as error:
        raise SnapshotError(
            "Snapshot committed but previous-stage backup retirement failed; "
            f"valid new stage: {output_dir}; retained backup path: {backup_dir}"
        ) from error
    return validated


def download_external_snapshots(
    output_dir: Path,
    specs: Sequence[SnapshotSpec] = SNAPSHOT_SPECS,
    *,
    timeout: float = 60.0,
    retries: int = 3,
    overwrite: bool = False,
    opener: _OpenUrl | None = None,
) -> dict[str, object]:
    """Stream, release-check, atomically publish, and manifest raw snapshots."""

    if not isinstance(output_dir, Path):
        raise TypeError("output_dir must be a pathlib.Path")
    specs = _validate_specs(specs)
    if type(timeout) not in (int, float) or type(timeout) is bool:
        raise TypeError("timeout must be a built-in number")
    timeout = float(timeout)
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be finite and positive")
    if type(retries) is not int:
        raise TypeError("retries must be a built-in int")
    if retries <= 0:
        raise ValueError("retries must be positive")
    if type(overwrite) is not bool:
        raise TypeError("overwrite must be a built-in bool")
    opener = _open_url if opener is None else opener
    if not callable(opener):
        raise TypeError("opener must be callable")

    output_dir = _reject_symlink_components(output_dir)
    manifest_path = output_dir / "manifest.json"
    previous_manifest: dict[str, object] | None = None
    if manifest_path.exists():
        previous_manifest = validate_external_snapshots(output_dir)
        if not overwrite:
            if previous_manifest.get("specs_sha256") != _specs_sha256(specs):
                raise SnapshotError("Existing snapshot manifest does not match requested specs")
            return previous_manifest
    elif output_dir.exists() and any(path.is_file() or path.is_symlink() for path in output_dir.rglob("*")):
        raise SnapshotError("Refusing to replace existing unregistered output")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink_components(output_dir.parent)
    stage_dir = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.snapshot-stage-", dir=output_dir.parent)
    )
    preserve_stage = False
    try:
        staged_manifest = _acquire_into_empty_stage(
            stage_dir, specs, timeout, retries, opener
        )
        return _commit_staged_snapshot(
            output_dir, stage_dir, staged_manifest, previous_manifest
        )
    except _SnapshotRecoveryError:
        preserve_stage = True
        raise
    finally:
        if not preserve_stage:
            try:
                shutil.rmtree(stage_dir)
            except OSError:
                pass


def validate_external_snapshots(output_dir: Path) -> dict[str, object]:
    """Validate one frozen manifest and every registered raw file without network."""

    if not isinstance(output_dir, Path):
        raise TypeError("output_dir must be a pathlib.Path")
    output_dir = _reject_symlink_components(output_dir)
    try:
        manifest = strict_json_load(output_dir / "manifest.json")
    except AtlasIOError as error:
        raise SnapshotError(f"Invalid snapshot manifest: {error}") from error
    if type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1:
        raise SnapshotError("Snapshot manifest schema_version must be 1")
    retrieved_at = manifest.get("retrieved_at")
    if type(retrieved_at) is not str or not retrieved_at:
        raise SnapshotError("Snapshot manifest retrieved_at is missing")
    try:
        parsed_retrieved_at = datetime.fromisoformat(
            retrieved_at[:-1] + "+00:00" if retrieved_at.endswith("Z") else retrieved_at
        )
    except ValueError as error:
        raise SnapshotError("Snapshot manifest retrieved_at is not canonical UTC ISO-8601") from error
    canonical_retrieved_at = parsed_retrieved_at.isoformat().replace("+00:00", "Z")
    if (
        parsed_retrieved_at.utcoffset() != timezone.utc.utcoffset(None)
        or canonical_retrieved_at != retrieved_at
        or "T" not in retrieved_at
        or not retrieved_at.endswith("Z")
    ):
        raise SnapshotError("Snapshot manifest retrieved_at is not canonical UTC ISO-8601")
    if type(manifest.get("specs_sha256")) is not str or not _SHA256_PATTERN.fullmatch(manifest["specs_sha256"]):
        raise SnapshotError("Snapshot manifest spec checksum is invalid")
    databases = manifest.get("databases")
    files = manifest.get("files")
    if type(databases) is not list or not databases or any(type(item) is not str or not item for item in databases):
        raise SnapshotError("Snapshot manifest databases must be non-empty strings")
    if databases != sorted(set(databases)):
        raise SnapshotError("Snapshot manifest databases must be sorted and unique")
    if type(files) is not list or not files:
        raise SnapshotError("Snapshot manifest files must be a non-empty list")

    identities: set[tuple[str, str]] = set()
    relative_paths: set[str] = set()
    reconstructed_specs: list[SnapshotSpec] = []
    validation_downloads: list[dict[str, object]] = []
    for index, row in enumerate(files):
        if type(row) is not dict:
            raise SnapshotError(f"Snapshot file record {index} must be an object")
        string_fields = (
            "database",
            "license",
            "name",
            "relative_path",
            "release",
            "release_parser",
            "requested_url",
            "resolved_url",
            "sha256",
        )
        for field in string_fields:
            if type(row.get(field)) is not str or not row[field]:
                raise SnapshotError(f"Snapshot file record {index} has invalid {field}")
        if type(row.get("bytes")) is not int or row["bytes"] <= 0:
            raise SnapshotError(f"Snapshot file record {index} has invalid byte size")
        if not _SHA256_PATTERN.fullmatch(row["sha256"]):
            raise SnapshotError(f"Snapshot file record {index} has invalid SHA-256")
        identity = (row["database"], row["name"])
        if identity in identities or row["relative_path"] in relative_paths:
            raise SnapshotError("Snapshot manifest contains duplicate registered files")
        identities.add(identity)
        relative_paths.add(row["relative_path"])
        if row["database"] not in databases:
            raise SnapshotError("Snapshot file database is absent from database list")
        path = _safe_registered_path(output_dir, row["relative_path"])
        try:
            size = path.stat().st_size
            checksum = sha256_file(path)
        except (OSError, AtlasIOError) as error:
            raise SnapshotError(f"Registered snapshot file is missing or invalid: {path}") from error
        if size != row["bytes"] or checksum != row["sha256"]:
            raise SnapshotError(f"Registered snapshot file was changed: {path}")
        raw_relative = PurePosixPath(row["relative_path"])
        reconstructed_specs.append(
            SnapshotSpec(
                row["database"],
                row["name"],
                row["requested_url"],
                str(PurePosixPath(*raw_relative.parts[1:])),
                row["license"],
                row["release_parser"],
            )
        )
        validation_downloads.append(
            {"resolved_url": row["resolved_url"], "temporary_path": path}
        )
    if set(databases) != {row["database"] for row in files}:
        raise SnapshotError("Snapshot database list does not match file records")
    try:
        reconstructed_specs = list(_validate_specs(reconstructed_specs))
    except (TypeError, ValueError) as error:
        raise SnapshotError(f"Snapshot manifest contains invalid specs: {error}") from error
    if _specs_sha256(reconstructed_specs) != manifest["specs_sha256"]:
        raise SnapshotError("Snapshot manifest spec checksum does not match file records")
    releases = _resolve_releases(reconstructed_specs, validation_downloads)
    for row, release in zip(files, releases):
        if row["release"] != release:
            raise SnapshotError(
                f"Snapshot release metadata does not match raw file: {row['relative_path']}"
            )
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path, nargs="?", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--retries", type=int, default=3)
    arguments = parser.parse_args(argv)
    if arguments.validate_only:
        manifest = validate_external_snapshots(arguments.output_dir)
    else:
        manifest = download_external_snapshots(
            arguments.output_dir,
            timeout=arguments.timeout,
            retries=arguments.retries,
            overwrite=arguments.overwrite,
        )
    print(
        json.dumps(
            {
                "databases": manifest["databases"],
                "files": len(manifest["files"]),
                "status": "valid",
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
