"""Cache-first access to official ESM Atlas feature descriptions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Mapping
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from esmc_sae.calibrate import _publish_temp_no_replace


FEATURE_COUNT = 16384
FEATURE_API_ROOT = (
    "https://biohub.ai/esm/protein/api/v1alpha1/features"
)
_CACHE_ROOT = Path(__file__).resolve().parent / "cache" / "feature_metadata"
_TIMEOUT_SECONDS = 15.0
_TOKEN_ENV_VARS = ("ESM_ATLAS_API_TOKEN", "BIOHUB_API_TOKEN")
_LOCAL_SNAPSHOT_PATH = (
    Path(__file__).resolve().parent / "interpret_data" / "features.json"
)


@dataclass(frozen=True)
class FeatureMetadata:
    feature_id: int
    description: str
    source_url: str
    retrieved_at: str


@dataclass(frozen=True)
class FeatureSnapshot:
    features: Mapping[int, Mapping[str, object]]
    source_path: Path
    sha256: str
    retrieved_at: str


def _validate_feature_id(feature_id: object) -> int:
    if type(feature_id) is not int or not 0 <= feature_id < FEATURE_COUNT:
        raise ValueError("feature_id must be an integer in [0, 16384)")
    return feature_id


def _source_url(feature_id: int) -> str:
    return f"{FEATURE_API_ROOT}/{feature_id}"


def _request_headers() -> dict[str, str]:
    headers = {
        "Accept": "application/json",
        "User-Agent": "esmc-sae-shortlist-metadata/1",
    }
    token = next(
        (
            os.environ[name]
            for name in _TOKEN_ENV_VARS
            if os.environ.get(name)
        ),
        None,
    )
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _fetch_json(
    url: str,
    *,
    headers: Mapping[str, str],
    timeout: float,
) -> object:
    request = Request(url, headers=dict(headers), method="GET")
    try:
        with urlopen(request, timeout=timeout) as response:
            status = getattr(response, "status", 200)
            if status != 200:
                raise RuntimeError(
                    f"official feature metadata request returned HTTP {status}"
                )
            raw = response.read()
    except HTTPError as error:
        raise RuntimeError(
            f"official feature metadata request returned HTTP {error.code}"
        ) from error
    except (URLError, TimeoutError, OSError) as error:
        raise RuntimeError("official feature metadata request failed") from error
    try:
        return json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(
            "official feature metadata response is not valid JSON"
        ) from error


def _official_description(payload: object, feature_id: int) -> str:
    if not isinstance(payload, Mapping):
        raise RuntimeError("official feature metadata response must be an object")
    response_id = payload.get(
        "feature_index",
        payload.get("feature_id", payload.get("id")),
    )
    if response_id is not None and response_id != feature_id:
        raise RuntimeError("official feature metadata response identity mismatch")
    description = payload.get("description")
    if description is None:
        feature = payload.get("feature")
        if isinstance(feature, Mapping):
            description = feature.get("description")
    if not isinstance(description, str) or not description.strip():
        return "unavailable"
    return description.strip()




def load_feature_snapshot(
    path: Path,
    *,
    require_complete: bool = False,
) -> FeatureSnapshot:
    """Load and validate one official all-feature API snapshot."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        raw = path.read_bytes()
        payload = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("local feature metadata snapshot is unreadable") from error
    if not isinstance(payload, Mapping) or set(payload) != {"data"}:
        raise RuntimeError("local feature metadata snapshot must be an object")
    data = payload.get("data")
    if not isinstance(data, list):
        raise RuntimeError("local feature metadata snapshot lacks a data list")
    features: dict[int, Mapping[str, object]] = {}
    for item in data:
        if not isinstance(item, Mapping):
            raise RuntimeError("local feature metadata snapshot item is malformed")
        response_id = item.get(
            "feature_index",
            item.get("feature_id", item.get("id")),
        )
        if type(response_id) is not int or not 0 <= response_id < FEATURE_COUNT:
            raise RuntimeError("local feature metadata snapshot has invalid identity")
        if response_id in features:
            raise RuntimeError("local feature metadata snapshot has duplicate identity")
        features[response_id] = item
    if require_complete and set(features) != set(range(FEATURE_COUNT)):
        raise RuntimeError("local feature metadata snapshot is incomplete")
    retrieved_at = datetime.fromtimestamp(
        path.stat().st_mtime,
        timezone.utc,
    ).isoformat()
    return FeatureSnapshot(
        features=features,
        source_path=path,
        sha256=hashlib.sha256(raw).hexdigest(),
        retrieved_at=retrieved_at,
    )

def _payload_to_metadata(
    payload: object,
    *,
    expected_feature_id: int,
    expected_source_url: str,
) -> FeatureMetadata:
    required = {
        "schema_version",
        "feature_id",
        "description",
        "source_url",
        "retrieved_at",
    }
    if not isinstance(payload, Mapping) or set(payload) != required:
        raise ValueError("metadata cache has an unexpected schema")
    if payload["schema_version"] != 1:
        raise ValueError("metadata cache has an unsupported schema version")
    if payload["feature_id"] != expected_feature_id:
        raise ValueError("metadata cache feature identity mismatch")
    if payload["source_url"] != expected_source_url:
        raise ValueError("metadata cache source identity mismatch")
    description = payload["description"]
    retrieved_at = payload["retrieved_at"]
    if (
        not isinstance(description, str)
        or not description
        or not isinstance(retrieved_at, str)
        or not retrieved_at
    ):
        raise ValueError("metadata cache contains invalid required fields")
    try:
        parsed_time = datetime.fromisoformat(retrieved_at)
    except ValueError as error:
        raise ValueError("metadata cache retrieval timestamp is invalid") from error
    if parsed_time.tzinfo is None:
        raise ValueError("metadata cache retrieval timestamp lacks a timezone")
    return FeatureMetadata(
        feature_id=expected_feature_id,
        description=description,
        source_url=expected_source_url,
        retrieved_at=retrieved_at,
    )


def _read_cache(path: Path, feature_id: int) -> FeatureMetadata:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("metadata cache is unreadable or malformed") from error
    return _payload_to_metadata(
        payload,
        expected_feature_id=feature_id,
        expected_source_url=_source_url(feature_id),
    )


def _write_cache_create_only(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                payload,
                handle,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        _publish_temp_no_replace(Path(temporary), path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def get_feature_metadata(
    feature_id: int,
    *,
    snapshot: FeatureSnapshot | None = None,
) -> FeatureMetadata:
    """Return one official description, fetching only on a validated miss."""

    feature_id = _validate_feature_id(feature_id)
    cache_path = _CACHE_ROOT / f"{feature_id}.json"
    if cache_path.exists():
        return _read_cache(cache_path, feature_id)

    url = _source_url(feature_id)
    if snapshot is None and _LOCAL_SNAPSHOT_PATH.is_file():
        snapshot = load_feature_snapshot(_LOCAL_SNAPSHOT_PATH)
    if snapshot is not None:
        try:
            response = snapshot.features[feature_id]
        except KeyError as error:
            raise RuntimeError(
                "local feature metadata snapshot lacks requested feature"
            ) from error
        retrieved_at = snapshot.retrieved_at
    else:
        response = _fetch_json(
            url,
            headers=_request_headers(),
            timeout=_TIMEOUT_SECONDS,
        )
        retrieved_at = datetime.now(timezone.utc).isoformat()
    payload = {
        "schema_version": 1,
        "feature_id": feature_id,
        "description": _official_description(response, feature_id),
        "source_url": url,
        "retrieved_at": retrieved_at,
    }
    try:
        _write_cache_create_only(cache_path, payload)
    except FileExistsError:
        return _read_cache(cache_path, feature_id)
    return _payload_to_metadata(
        payload,
        expected_feature_id=feature_id,
        expected_source_url=url,
    )
