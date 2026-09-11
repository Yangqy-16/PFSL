"""Two-phase, independently resumable feature descriptions."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from .codex_cli_transport import (
    GPT_MODEL,
    CodexCLITransportError,
    codex_exec_argv,
    parse_codex_exec_jsonl,
)
from .feature_atlas_config import FeatureAtlasConfig
from .feature_atlas_io import (
    AtlasIOError,
    sha256_file,
    strict_json_lines,
    strict_json_load,
    write_json_atomic,
)
from .feature_atlas_selection import CLASSIFICATIONS, validate_selection_manifest
from .span_attribution import (
    EVIDENCE_GRADES,
    build_protrek_model_binding,
    validate_span_attribution,
)


SCHEMA_VERSION = 3
CLAUDE_MODEL = "claude-sonnet-4-6"
GEMINI_MODEL = "gemini-2.5-pro"
MAX_ATTEMPTS = 3
MAX_CALLS = 800
_PROVIDERS = ("claude", "gemini")
_KNOWN_PROVIDERS = (*_PROVIDERS, "gpt")
PROVIDER_MODE_DUAL = "claude_gemini"
PROVIDER_MODE_CLAUDE_ONLY = "claude_only"
PROVIDER_MODE_GPT_ONLY = "gpt_only"
_SHA = re.compile(r"^[0-9a-f]{64}$")
_UTC = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z$")
_RESERVED = re.compile(
    r"^(?:train_feature_|validation_feature_|draft_|verdict_|agreement_feature_|"
    r"feature_|train_freeze_|draft_freeze_|validation_freeze_).*\.json$"
)
_LEAKAGE = re.compile(
    r"validation|synthetic[\s_-]+lethal|sl_label|(?<![a-z0-9])sl(?![a-z0-9])",
    flags=re.IGNORECASE,
)

_DRAFT_KEYS = frozenset({
    "alternative_descriptions", "candidate_description", "confidence",
    "evidence_against", "evidence_for", "falsification_test",
    "field_detector_risk", "short_label", "supported_concept_ids",
})
_VERDICT_KEYS = frozenset({
    "calibrated_confidence", "contradictions",
    "supporting_validation_evidence", "verdict",
})
_TRAIN_PACKET_KEYS = frozenset({
    "candidate_concepts", "diagnostics", "feature_index",
    "per_concept_sample_uncertainty", "source_excluded_training_enrichment",
    "train_contrasts", "train_spans", "train_top_texts",
})
_VALIDATION_PACKET_KEYS = frozenset({
    "feature_index", "heldout_concept_checks", "validation_contrasts",
    "validation_spans", "validation_top_texts",
})
_IDENTITY_KEYS = frozenset({
    "feature_index", "masking_evidence_grade", "statistical_classification",
    "statistical_evidence_grade", "validated_concept_ids",
})
_BINDING_BASE_KEYS = frozenset({
    "cli_version", "executable", "executable_sha256", "model", "params", "provider",
    "reported_model", "reported_version", "resolved_executable",
    "timeout_seconds",
})
_BINDING_KEYS = _BINDING_BASE_KEYS | {"binding_sha256"}
_ATTEMPT_BASE_KEYS = frozenset({
    "cli_version", "command", "duration_seconds", "ended_at", "error_sha256",
    "executable", "executable_sha256", "model", "params", "parse_error", "parsed_response",
    "parsed_response_sha256", "prompt", "prompt_sha256", "provider",
    "raw_response", "raw_response_sha256", "reported_model",
    "reported_version", "resolved_executable", "returncode", "started_at",
    "stderr", "stderr_sha256", "stdout", "stdout_sha256", "timeout_seconds",
})
_ATTEMPT_KEYS = _ATTEMPT_BASE_KEYS | {
    "artifact_kind", "feature_index", "phase", "round", "sequence",
}
_STATUS_KEYS = frozenset({
    "artifact_kind", "attempt_paths", "feature_index", "parsed", "phase",
    "provider", "response_sha256", "round", "status",
})
_ARTIFACT_KEYS = frozenset({
    "bytes", "feature_index", "kind", "path", "provider", "round",
    "sequence", "sha256",
})
_MANIFEST_KEYS = frozenset({
    "artifacts", "complete", "draft_freeze", "family_order",
    "input_fingerprint", "label_binding", "max_attempts_per_phase", "max_calls", "output_fingerprint",
    "phase", "planned_calls", "provider_bindings", "provider_mode",
    "provider_statuses", "schema_version",
    "source_binding", "statistical_identities", "train_freeze",
    "validation_freeze",
})
_CONFIDENCE = frozenset({"low", "medium", "high"})
_VERDICTS = frozenset({
    "supported", "partially_supported", "refuted", "insufficient",
})
_SUPPORTING = frozenset({"supported", "partially_supported"})


class LLMFeatureDescriptionError(ValueError):
    """Raised when the Task 10 closed protocol is violated."""


def _clone(value: object) -> object:
    try:
        return json.loads(json.dumps(value, allow_nan=False, sort_keys=True))
    except (TypeError, ValueError) as error:
        raise LLMFeatureDescriptionError("payload is not finite strict JSON") from error


def canonical_json(value: object) -> str:
    return json.dumps(
        _clone(value), allow_nan=False, sort_keys=True, separators=(",", ":")
    )


def text_sha256(text: str) -> str:
    if type(text) is not str:
        raise TypeError("text must be a built-in str")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_sha256(value: object) -> str:
    return text_sha256(canonical_json(value))


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _parse_raw_object(raw: str) -> dict[str, object]:
    if type(raw) is not str:
        raise TypeError("LLM response must be text")
    try:
        value = json.loads(
            raw.strip(),
            object_pairs_hook=_unique_object,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant: {token}")
            ),
        )
    except (json.JSONDecodeError, TypeError, ValueError) as error:
        raise LLMFeatureDescriptionError("response must be one bare strict JSON object") from error
    if not isinstance(value, dict):
        raise LLMFeatureDescriptionError("response must be one bare strict JSON object")
    return value


def _string_list(value: object, label: str) -> list[str]:
    if (
        not isinstance(value, list)
        or any(type(item) is not str or not item.strip() for item in value)
        or len(value) != len(set(value))
    ):
        raise LLMFeatureDescriptionError(f"{label} must be unique nonempty strings")
    return list(value)


def parse_draft(raw: str) -> dict[str, object]:
    value = _parse_raw_object(raw)
    if set(value) != _DRAFT_KEYS:
        raise LLMFeatureDescriptionError("draft has invalid exact schema")
    for name in (
        "candidate_description", "falsification_test", "field_detector_risk", "short_label"
    ):
        if type(value.get(name)) is not str or not str(value[name]).strip():
            raise LLMFeatureDescriptionError(f"draft {name} must be nonempty text")
    if value.get("confidence") not in _CONFIDENCE:
        raise LLMFeatureDescriptionError("draft confidence is invalid")
    for name in (
        "alternative_descriptions", "evidence_against", "evidence_for",
        "supported_concept_ids",
    ):
        _string_list(value.get(name), f"draft {name}")
    return value


def parse_verdict(raw: str) -> dict[str, object]:
    value = _parse_raw_object(raw)
    if set(value) != _VERDICT_KEYS:
        raise LLMFeatureDescriptionError("verdict has invalid exact schema")
    if value.get("verdict") not in _VERDICTS:
        raise LLMFeatureDescriptionError("verdict domain is invalid")
    if value.get("calibrated_confidence") not in _CONFIDENCE:
        raise LLMFeatureDescriptionError("verdict calibrated confidence is invalid")
    _string_list(value.get("contradictions"), "verdict contradictions")
    _string_list(
        value.get("supporting_validation_evidence"),
        "verdict supporting validation evidence",
    )
    return value


def _validate_train_packet(packet: object) -> dict[str, object]:
    if not isinstance(packet, dict) or set(packet) != _TRAIN_PACKET_KEYS:
        raise LLMFeatureDescriptionError("train packet has invalid closed schema")
    if type(packet.get("feature_index")) is not int or packet["feature_index"] < 0:
        raise LLMFeatureDescriptionError("train packet feature identity is invalid")
    list_fields = (
        "candidate_concepts", "per_concept_sample_uncertainty",
        "source_excluded_training_enrichment", "train_contrasts", "train_spans",
        "train_top_texts",
    )
    if any(not isinstance(packet.get(name), list) for name in list_fields):
        raise LLMFeatureDescriptionError("train packet evidence lists are invalid")
    diagnostics = packet.get("diagnostics")
    if not isinstance(diagnostics, dict) or set(diagnostics) != {
        "absorption", "field", "splitting"
    }:
        raise LLMFeatureDescriptionError("train packet diagnostics have invalid schema")
    serialized = canonical_json(packet)
    if _LEAKAGE.search(serialized):
        raise LLMFeatureDescriptionError("train packet leaks validation/SL material")
    return packet


def _validate_validation_packet(packet: object) -> dict[str, object]:
    if not isinstance(packet, dict) or set(packet) != _VALIDATION_PACKET_KEYS:
        raise LLMFeatureDescriptionError("validation packet has invalid closed schema")
    if type(packet.get("feature_index")) is not int or packet["feature_index"] < 0:
        raise LLMFeatureDescriptionError("validation packet feature identity is invalid")
    if any(
        not isinstance(packet.get(name), list)
        for name in _VALIDATION_PACKET_KEYS - {"feature_index"}
    ):
        raise LLMFeatureDescriptionError("validation packet evidence lists are invalid")
    return packet


def build_draft_prompt(train_packet: Mapping[str, object]) -> str:
    packet = _validate_train_packet(dict(train_packet))
    prompt = (
        "Independently draft one mechanistic SAE feature description using only the "
        "frozen training evidence. Return one bare JSON object with exactly these keys: "
        + ",".join(sorted(_DRAFT_KEYS))
        + ". confidence must be low, medium, or high.\ntrain_packet="
        + canonical_json(packet)
    )
    if _LEAKAGE.search(prompt):
        raise LLMFeatureDescriptionError("complete draft prompt leaks validation/SL material")
    return prompt


def build_verdict_prompt(
    immutable_draft: Mapping[str, object],
    draft_sha256: str,
    validation_packet: Mapping[str, object],
) -> str:
    parsed = parse_draft(canonical_json(immutable_draft))
    if draft_sha256 != canonical_sha256(parsed):
        raise LLMFeatureDescriptionError("immutable draft hash mismatch")
    packet = _validate_validation_packet(dict(validation_packet))
    return (
        "Judge, but never rewrite, this immutable same-provider draft using only the "
        "frozen validation packet. Return one bare JSON object with exactly keys "
        "calibrated_confidence,contradictions,supporting_validation_evidence,verdict.\n"
        "calibrated_confidence must be low, medium, or high.\n"
        "verdict must be supported, partially_supported, refuted, or insufficient.\n"
        f"draft_sha256={draft_sha256}\nimmutable_draft={canonical_json(parsed)}\n"
        f"validation_packet={canonical_json(packet)}"
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _binding_hash(binding: Mapping[str, object]) -> str:
    return canonical_sha256({key: binding[key] for key in sorted(_BINDING_BASE_KEYS)})


class LLMProvider:
    """Explicit provider identity and bounded argv-only invocation adapter."""

    def __init__(
        self,
        *,
        provider: str,
        executable: str,
        model: str,
        params: Sequence[str] = (),
        timeout_seconds: float = 120.0,
        cli_version: str = "unrecorded",
        reported_model: str | None = None,
        reported_version: str = "unrecorded",
        resolved_executable: str | None = None,
    ) -> None:
        if any(type(value) is not str or not value for value in (provider, executable, model)):
            raise TypeError("provider identity fields must be nonempty strings")
        if isinstance(params, (str, bytes)) or any(type(value) is not str for value in params):
            raise TypeError("provider params must be a sequence of strings")
        if (
            type(timeout_seconds) not in (int, float)
            or not math.isfinite(float(timeout_seconds))
            or timeout_seconds <= 0
        ):
            raise TypeError("provider timeout must be finite and positive")
        self.provider = provider
        self.executable = executable
        self.resolved_executable = resolved_executable or shutil.which(executable) or executable
        self.model = model
        self.params = tuple(params)
        self.timeout_seconds = float(timeout_seconds)
        self.cli_version = cli_version
        self.reported_model = reported_model or model
        self.reported_version = reported_version
        resolved = Path(self.resolved_executable)
        if self.resolved_executable.startswith("fixture://"):
            self.executable_sha256 = "fixture-unbound"
        elif resolved.is_file():
            self.executable_sha256 = sha256_file(resolved)
        else:
            self.executable_sha256 = None

    def command(self, prompt: str) -> list[str]:
        return [self.executable, "--model", self.model, *self.params, prompt]

    def binding(self) -> dict[str, object]:
        binding: dict[str, object] = {
            "cli_version": self.cli_version,
            "executable": self.executable,
            "executable_sha256": self.executable_sha256,
            "model": self.model,
            "params": list(self.params),
            "provider": self.provider,
            "reported_model": self.reported_model,
            "reported_version": self.reported_version,
            "resolved_executable": self.resolved_executable,
            "timeout_seconds": self.timeout_seconds,
        }
        binding["binding_sha256"] = _binding_hash(binding)
        return binding

    def attempt_record(
        self,
        *,
        prompt: str,
        command: Sequence[str],
        stdout: str,
        stderr: str,
        returncode: int,
        started_at: str,
        ended_at: str,
        duration_seconds: float,
    ) -> dict[str, object]:
        error = None if returncode == 0 else f"nonzero return code: {returncode}"
        return {
            **{key: self.binding()[key] for key in _BINDING_BASE_KEYS},
            "command": list(command),
            "duration_seconds": float(duration_seconds),
            "ended_at": ended_at,
            "error_sha256": text_sha256(error) if error is not None else None,
            "parse_error": error,
            "parsed_response": None,
            "parsed_response_sha256": None,
            "prompt": prompt,
            "prompt_sha256": text_sha256(prompt),
            "raw_response": str(stdout),
            "raw_response_sha256": text_sha256(str(stdout)),
            "returncode": int(returncode),
            "started_at": started_at,
            "stderr": str(stderr),
            "stderr_sha256": text_sha256(str(stderr)),
            "stdout": str(stdout),
            "stdout_sha256": text_sha256(str(stdout)),
        }

    def invoke(self, prompt: str) -> dict[str, object]:
        command = self.command(prompt)
        started_at, start = _utc_now(), time.monotonic()
        try:
            completed = subprocess.run(
                command,
                shell=False,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                check=False,
            )
            stdout, stderr, returncode = (
                completed.stdout, completed.stderr, completed.returncode
            )
        except subprocess.TimeoutExpired as error:
            stdout = error.stdout.decode() if isinstance(error.stdout, bytes) else (error.stdout or "")
            stderr = error.stderr.decode() if isinstance(error.stderr, bytes) else (error.stderr or "")
            stderr = f"{stderr}\ntimeout after {self.timeout_seconds:g}s".lstrip()
            returncode = 124
        except OSError as error:
            stdout, stderr, returncode = "", str(error), 127
        return self.attempt_record(
            prompt=prompt, command=command, stdout=stdout, stderr=stderr,
            returncode=returncode, started_at=started_at, ended_at=_utc_now(),
            duration_seconds=max(0.0, time.monotonic() - start),
        )


class ClaudeCLIProvider(LLMProvider):
    def __init__(
        self,
        *,
        executable: str = "claude",
        resolved_executable: str | None = None,
        params: Sequence[str] = ("--print",),
        timeout_seconds: float = 120.0,
        cli_version: str = "unrecorded",
        reported_version: str = "unrecorded",
    ) -> None:
        super().__init__(
            provider="claude", executable=executable,
            resolved_executable=resolved_executable, model=CLAUDE_MODEL,
            params=params, timeout_seconds=timeout_seconds, cli_version=cli_version,
            reported_model=CLAUDE_MODEL, reported_version=reported_version,
        )


class GeminiCLIProvider(LLMProvider):
    def __init__(
        self,
        *,
        executable: str = "gemini",
        resolved_executable: str | None = None,
        params: Sequence[str] = ("--prompt",),
        timeout_seconds: float = 120.0,
        cli_version: str = "unrecorded",
        reported_version: str = "unrecorded",
    ) -> None:
        super().__init__(
            provider="gemini", executable=executable,
            resolved_executable=resolved_executable, model=GEMINI_MODEL,
            params=params, timeout_seconds=timeout_seconds, cli_version=cli_version,
            reported_model=GEMINI_MODEL, reported_version=reported_version,
        )


class CodexCLIProvider(LLMProvider):
    def __init__(
        self,
        *,
        executable: str = "codex",
        resolved_executable: str | None = None,
        timeout_seconds: float = 120.0,
        cli_version: str = "unrecorded",
        reported_version: str = "unrecorded",
    ) -> None:
        super().__init__(
            provider="gpt", executable=executable,
            resolved_executable=resolved_executable, model=GPT_MODEL,
            params=tuple(codex_exec_argv(executable, GPT_MODEL)[1:]),
            timeout_seconds=timeout_seconds, cli_version=cli_version,
            reported_model=GPT_MODEL, reported_version=reported_version,
        )

    def command(self, prompt: str) -> list[str]:
        return codex_exec_argv(self.executable, self.model)

    def invoke(self, prompt: str) -> dict[str, object]:
        command = self.command(prompt)
        started_at, start = _utc_now(), time.monotonic()
        raw_jsonl = ""
        stderr = ""
        returncode = 0
        try:
            completed = subprocess.run(
                command,
                input=prompt,
                shell=False,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                check=False,
            )
            raw_jsonl = completed.stdout
            stderr = completed.stderr
            returncode = completed.returncode
        except subprocess.TimeoutExpired as error:
            raw_jsonl = (
                error.stdout.decode()
                if isinstance(error.stdout, bytes)
                else (error.stdout or "")
            )
            stderr = (
                error.stderr.decode()
                if isinstance(error.stderr, bytes)
                else (error.stderr or "")
            )
            stderr = f"{stderr}\ntimeout after {self.timeout_seconds:g}s".lstrip()
            returncode = 124
        except OSError as error:
            stderr, returncode = str(error), 127

        final_response = ""
        if returncode == 0:
            try:
                final_response = parse_codex_exec_jsonl(raw_jsonl)
            except CodexCLITransportError as error:
                stderr = f"{stderr}\ntransport error: {error}".lstrip()
                returncode = 65
        audit = (
            f"[codex-jsonl-audit]\nsha256={text_sha256(raw_jsonl)}\n{raw_jsonl}"
        )
        stderr = f"{stderr.rstrip()}\n{audit}".lstrip()
        return self.attempt_record(
            prompt=prompt, command=command, stdout=final_response, stderr=stderr,
            returncode=returncode, started_at=started_at, ended_at=_utc_now(),
            duration_seconds=max(0.0, time.monotonic() - start),
        )


def _validate_binding(binding: object, *, allow_test_fixture_unbound: bool) -> dict[str, object]:
    if not isinstance(binding, dict) or set(binding) != _BINDING_KEYS:
        raise LLMFeatureDescriptionError("provider binding has invalid exact schema")
    if binding["binding_sha256"] != _binding_hash(binding):
        raise LLMFeatureDescriptionError("provider binding hash changed")
    provider = binding.get("provider")
    expected_model = (
        CLAUDE_MODEL if provider == "claude"
        else GEMINI_MODEL if provider == "gemini"
        else GPT_MODEL if provider == "gpt"
        else None
    )
    if expected_model is None or binding.get("model") != expected_model or binding.get("reported_model") != expected_model:
        raise LLMFeatureDescriptionError("provider/model identity is not the exact frozen pair")
    if (
        type(binding.get("cli_version")) is not str
        or type(binding.get("reported_version")) is not str
        or type(binding.get("resolved_executable")) is not str
        or not isinstance(binding.get("params"), list)
    ):
        raise LLMFeatureDescriptionError("provider provenance fields are invalid")
    fixture = str(binding["resolved_executable"]).startswith("fixture://")
    if fixture and not allow_test_fixture_unbound:
        raise LLMFeatureDescriptionError("fake provider requires explicit fixture gate")
    if fixture and binding.get("executable_sha256") != "fixture-unbound":
        raise LLMFeatureDescriptionError("fixture executable binding is invalid")
    if not fixture and (
        binding["cli_version"] == "unrecorded"
        or binding["reported_version"] == "unrecorded"
        or not Path(str(binding["resolved_executable"])).is_absolute()
        or not Path(str(binding["resolved_executable"])).is_file()
        or not _SHA.fullmatch(str(binding.get("executable_sha256")))
        or sha256_file(Path(str(binding["resolved_executable"]))) != binding["executable_sha256"]
    ):
        raise LLMFeatureDescriptionError("production provider provenance is incomplete")
    return binding


def validate_provider_family(
    providers: Sequence[LLMProvider], *, allow_test_fixture_unbound: bool,
    return_mode: bool = False,
) -> list[dict[str, object]] | tuple[list[dict[str, object]], str]:
    names = [provider.provider for provider in providers]
    if names == ["gpt"]:
        mode = PROVIDER_MODE_GPT_ONLY
    elif names == ["claude"] or set(names) == {"claude"} and len(names) == 1:
        mode = PROVIDER_MODE_CLAUDE_ONLY
    elif len(names) == 2 and set(names) == {"claude", "gemini"}:
        mode = PROVIDER_MODE_DUAL
    else:
        raise LLMFeatureDescriptionError(
            "providers must be exact GPT, exact Claude, or exact Claude+Gemini"
        )
    bindings = sorted(
        (_validate_binding(provider.binding(), allow_test_fixture_unbound=allow_test_fixture_unbound)
         for provider in providers),
        key=lambda row: str(row["provider"]),
    )
    return (bindings, mode) if return_mode else bindings


def _provider_mode_from_names(names: Sequence[object]) -> str:
    if list(names) == ["gpt"]:
        return PROVIDER_MODE_GPT_ONLY
    if list(names) == ["claude"]:
        return PROVIDER_MODE_CLAUDE_ONLY
    if list(names) == list(_PROVIDERS):
        return PROVIDER_MODE_DUAL
    raise LLMFeatureDescriptionError("provider binding family is invalid")


def unavailable_agreement() -> dict[str, object]:
    return {
        "available": False,
        "llm_consensus": None,
        "reason": "gemini_skipped_by_user",
    }


def _gpt_only_agreement() -> dict[str, object]:
    return {
        "available": False,
        "llm_consensus": None,
        "reason": "single_gpt_provider",
    }


def _initial_provider_statuses(mode: str) -> dict[str, str]:
    if mode == PROVIDER_MODE_GPT_ONLY:
        return {
            "gpt": "pending",
            "claude": "skipped_by_user",
            "gemini": "skipped_by_user",
        }
    return {
        "claude": "pending",
        "gemini": (
            "skipped_by_user"
            if mode == PROVIDER_MODE_CLAUDE_ONLY else "pending"
        ),
    }


def _completed_provider_statuses(mode: str) -> dict[str, str]:
    if mode == PROVIDER_MODE_GPT_ONLY:
        return {
            "gpt": "completed",
            "claude": "skipped_by_user",
            "gemini": "skipped_by_user",
        }
    return {
        "claude": "completed",
        "gemini": (
            "skipped_by_user"
            if mode == PROVIDER_MODE_CLAUDE_ONLY else "completed"
        ),
    }


def normalized_label_tokens(label: str) -> tuple[str, ...]:
    if type(label) is not str:
        raise TypeError("label must be text")
    return tuple(re.findall(r"[a-z0-9]+", label.casefold()))


def _label_binding(embedder: object, *, allow_test_fixture_unbound: bool) -> dict[str, object]:
    try:
        model = build_protrek_model_binding(
            embedder, allow_test_fixture_unbound=allow_test_fixture_unbound
        )
    except Exception as error:
        raise LLMFeatureDescriptionError(f"frozen ProTrek label binding failed: {error}") from error
    fixture = model == {"fingerprint": "fixture-unbound"}
    if fixture and not allow_test_fixture_unbound:
        raise LLMFeatureDescriptionError("fake label embedder requires explicit fixture gate")
    binding = {"kind": "frozen_protrek", "model_binding": model}
    binding["binding_sha256"] = canonical_sha256(binding)
    return binding


def _cosine_pair(encoded: object) -> float:
    if not isinstance(encoded, torch.Tensor) or encoded.ndim != 2 or encoded.shape[0] != 2:
        raise LLMFeatureDescriptionError("label embedder returned invalid shape")
    vectors = encoded.detach().to(dtype=torch.float64, device="cpu")
    if not bool(torch.isfinite(vectors).all()):
        raise LLMFeatureDescriptionError("label embeddings must be finite")
    norms = torch.linalg.vector_norm(vectors, dim=1)
    if bool((norms <= 0).any()):
        raise LLMFeatureDescriptionError("label embedding norm must be positive")
    return float(torch.dot(vectors[0], vectors[1]) / (norms[0] * norms[1]))


def _claim(value: str) -> str:
    return " ".join(normalized_label_tokens(value))


def compute_llm_agreement(
    left: Mapping[str, object],
    right: Mapping[str, object],
    *,
    validated_concept_ids: set[str] | Sequence[str],
    label_embedder: object,
    allow_test_fixture_unbound: bool = False,
) -> dict[str, object]:
    _label_binding(label_embedder, allow_test_fixture_unbound=allow_test_fixture_unbound)
    drafts = [parse_draft(canonical_json(item.get("draft"))) for item in (left, right)]
    verdicts = [parse_verdict(canonical_json(item.get("verdict"))) for item in (left, right)]
    validated = set(validated_concept_ids)
    supported = [
        set(value["supported_concept_ids"]) & validated
        if decision["verdict"] in _SUPPORTING else set()
        for value, decision in zip(drafts, verdicts)
    ]
    labels = [str(value["short_label"]) for value in drafts]
    token_sets = [set(normalized_label_tokens(label)) for label in labels]
    union = token_sets[0] | token_sets[1]
    encoded = label_embedder.encode(labels)
    pairs: list[dict[str, object]] = []
    positives = [
        [("evidence_for", text) for text in drafts[index]["evidence_for"]]
        for index in range(2)
    ]
    negatives = [
        [("evidence_against", text) for text in drafts[index]["evidence_against"]]
        + [("verdict_contradiction", text) for text in verdicts[index]["contradictions"]]
        for index in range(2)
    ]
    for positive_provider, negative_provider in ((0, 1), (1, 0)):
        for positive_source, positive in positives[positive_provider]:
            for negative_source, negative in negatives[negative_provider]:
                normalized = _claim(positive)
                if normalized and normalized == _claim(negative):
                    pairs.append({
                        "negative_claim": negative,
                        "negative_provider": _PROVIDERS[negative_provider],
                        "negative_source": negative_source,
                        "normalized_claim": normalized,
                        "positive_claim": positive,
                        "positive_provider": _PROVIDERS[positive_provider],
                        "positive_source": positive_source,
                    })
    direct = (
        verdicts[0]["verdict"] in _SUPPORTING and verdicts[1]["verdict"] == "refuted"
    ) or (
        verdicts[1]["verdict"] in _SUPPORTING and verdicts[0]["verdict"] == "refuted"
    )
    flags = []
    if direct:
        flags.append("direct_verdict_contradiction")
    if pairs:
        flags.append("claim_contradiction")
    result = {
        "contradiction_flags": flags,
        "contradictory_claim_pairs": pairs,
        "field_risk_agreement": (
            str(drafts[0]["field_detector_risk"]).casefold()
            == str(drafts[1]["field_detector_risk"]).casefold()
        ),
        "frozen_protrek_label_cosine": _cosine_pair(encoded),
        "label_binding_sha256": _label_binding(
            label_embedder, allow_test_fixture_unbound=allow_test_fixture_unbound
        )["binding_sha256"],
        "llm_consensus": bool(supported[0]) and supported[0] == supported[1] and not flags,
        "normalized_label_jaccard": len(token_sets[0] & token_sets[1]) / len(union) if union else 1.0,
        "supported_validated_concept_jaccard": (
            len(supported[0] & supported[1]) / len(supported[0] | supported[1])
            if supported[0] | supported[1] else 0.0
        ),
        "supported_validated_concept_overlap": sorted(supported[0] & supported[1]),
        "verdict_agreement": verdicts[0]["verdict"] == verdicts[1]["verdict"],
    }
    return result


def derive_feature_family(selection: Mapping[str, object]) -> list[int]:
    selected = selection.get("selected")
    if not isinstance(selected, list):
        raise LLMFeatureDescriptionError("Task 8 selected family is invalid")
    features = []
    for row in selected:
        feature = row.get("feature_index") if isinstance(row, Mapping) else None
        if type(feature) is not int or feature < 0:
            raise LLMFeatureDescriptionError("Task 8 selected identity is invalid")
        features.append(feature)
    if len(features) != len(set(features)):
        raise LLMFeatureDescriptionError("Task 8 selected identities are duplicated")
    return sorted(features)


def build_train_packet(row: Mapping[str, object], spans: Mapping[str, object]) -> dict[str, object]:
    feature = row.get("feature_index")
    results = row.get("train_concept_results", [])
    selected = spans.get("selected_records", {})
    packet = {
        "candidate_concepts": [
            {"concept_id": item.get("concept_id"), "concept_name": item.get("concept_name"), "database": item.get("database")}
            for item in results if isinstance(item, Mapping)
        ],
        "diagnostics": {
            "absorption": row.get("absorption_diagnostics"),
            "field": {
                "preferred_aspect": row.get("preferred_aspect"),
                "source_effect_loss": row.get("source_effect_loss"),
                "source_lift": row.get("source_lift"),
                "source_purity": row.get("source_purity"),
            },
            "splitting": row.get("splitting_diagnostics"),
        },
        "feature_index": feature,
        "per_concept_sample_uncertainty": [
            {
                "concept_id": item.get("concept_id"), "database": item.get("database"),
                "eligible_accession_count": item.get("eligible_accession_count"),
                "firing_count": item.get("source_excluded", {}).get("firing_count") if isinstance(item.get("source_excluded"), Mapping) else None,
                "uncertainty": {
                    "fisher_q_value": item.get("source_excluded", {}).get("fisher_q_value") if isinstance(item.get("source_excluded"), Mapping) else None,
                    "rank_q_value": item.get("source_excluded", {}).get("rank_q_value") if isinstance(item.get("source_excluded"), Mapping) else None,
                },
            }
            for item in results if isinstance(item, Mapping)
        ],
        "source_excluded_training_enrichment": [
            {"concept_id": item.get("concept_id"), "database": item.get("database"), "source_excluded": item.get("source_excluded")}
            for item in results if isinstance(item, Mapping)
        ],
        "train_contrasts": [
            dict(item) for item in row.get("contrasts", [])
            if isinstance(item, Mapping) and item.get("split") == "train"
        ],
        "train_spans": list(selected.get("train", [])) if isinstance(selected, Mapping) else [],
        "train_top_texts": list(row.get("top_texts", {}).get("train", [])) if isinstance(row.get("top_texts"), Mapping) else [],
    }
    return _validate_train_packet(_clone(packet))


def build_validation_packet(row: Mapping[str, object], spans: Mapping[str, object]) -> dict[str, object]:
    selected = spans.get("selected_records", {})
    packet = {
        "feature_index": row.get("feature_index"),
        "heldout_concept_checks": list(row.get("heldout_concept_results", [])),
        "validation_contrasts": [
            dict(item) for item in row.get("contrasts", [])
            if isinstance(item, Mapping) and item.get("split") == "validation"
        ],
        "validation_spans": list(selected.get("validation", [])) if isinstance(selected, Mapping) else [],
        "validation_top_texts": list(row.get("top_texts", {}).get("validation", [])) if isinstance(row.get("top_texts"), Mapping) else [],
    }
    return _validate_validation_packet(_clone(packet))


def _statistical_identity(row: Mapping[str, object], spans: Mapping[str, object]) -> dict[str, object]:
    heldout = row.get("heldout_concept_results", [])
    confirmed = [item for item in heldout if isinstance(item, Mapping) and item.get("confirmed") is True]
    tiers = [str(item.get("evidence_tier")) for item in confirmed]
    rank = {"unvalidated": 0, "insufficient": 1, "descriptive": 2, "confirmatory": 3}
    grade = max(tiers, key=lambda value: rank.get(value, -1)) if tiers else "unvalidated"
    return {
        "feature_index": row.get("feature_index"),
        "masking_evidence_grade": spans.get("evidence_grade"),
        "statistical_classification": row.get("classification"),
        "statistical_evidence_grade": grade,
        "validated_concept_ids": sorted({str(item["concept_id"]) for item in confirmed}),
    }


def _validate_identity(identity: object) -> dict[str, object]:
    if not isinstance(identity, dict) or set(identity) != _IDENTITY_KEYS:
        raise LLMFeatureDescriptionError("statistical identity has invalid exact schema")
    if type(identity.get("feature_index")) is not int or identity["feature_index"] < 0:
        raise LLMFeatureDescriptionError("statistical identity feature is invalid")
    if identity.get("statistical_classification") not in CLASSIFICATIONS:
        raise LLMFeatureDescriptionError("statistical classification is invalid")
    if identity.get("statistical_evidence_grade") not in {
        "unvalidated", "insufficient", "descriptive", "confirmatory"
    }:
        raise LLMFeatureDescriptionError("statistical evidence grade is invalid")
    if identity.get("masking_evidence_grade") not in EVIDENCE_GRADES:
        raise LLMFeatureDescriptionError("masking evidence grade is invalid")
    ids = _string_list(identity.get("validated_concept_ids"), "validated concept IDs")
    if ids != sorted(ids):
        raise LLMFeatureDescriptionError("validated concept IDs are not canonical")
    return identity


def validate_task9_family(
    manifest: Mapping[str, object],
    *,
    selected_features: Sequence[int],
    current_bindings: Mapping[str, object] | None,
    sae: object | None,
    plans_by_feature: Mapping[int, object] | None,
) -> None:
    if current_bindings is None or sae is None or plans_by_feature is None:
        raise LLMFeatureDescriptionError("Task 9 binding/SAE/frozen plans are required")
    selected = sorted(selected_features)
    finals = manifest.get("final_outputs")
    final_features = [item.get("feature_index") for item in finals] if isinstance(finals, list) else None
    if manifest.get("features") != selected or final_features != selected or sorted(plans_by_feature) != selected:
        raise LLMFeatureDescriptionError("Task 9 feature/final/plan family differs from Task 8")


def capture_materializer_source_binding(
    selection_path: Path,
    atlas_path: Path,
    span_output_dir: Path,
    inference_path: Path,
) -> dict[str, object]:
    span_manifest_path = span_output_dir / "manifest.json"
    span_manifest = strict_json_load(span_manifest_path)
    final_outputs = span_manifest.get("final_outputs")
    if not isinstance(final_outputs, list):
        raise LLMFeatureDescriptionError("Task 9 final output registry is invalid")
    checked_finals = []
    for item in final_outputs:
        if (
            not isinstance(item, dict)
            or set(item) != {"bytes", "feature_index", "path", "sha256"}
            or type(item.get("path")) is not str
            or type(item.get("feature_index")) is not int
        ):
            raise LLMFeatureDescriptionError("Task 9 final output metadata is invalid")
        path = span_output_dir / str(item["path"])
        actual = {
            "bytes": path.stat().st_size,
            "feature_index": item["feature_index"],
            "path": item["path"],
            "sha256": sha256_file(path),
        }
        if actual != item:
            raise LLMFeatureDescriptionError("Task 9 registered final source changed")
        checked_finals.append(actual)
    return {
        "atlas_sha256": sha256_file(atlas_path),
        "inference_sha256": sha256_file(inference_path),
        "selection_sha256": sha256_file(selection_path),
        "span_final_outputs": checked_finals,
        "span_manifest_sha256": sha256_file(span_manifest_path),
    }


def verify_materializer_sources(context: Mapping[str, object]) -> None:
    required = {
        "selection_path", "atlas_path", "span_output_dir", "inference_path",
        "source_binding",
    }
    if not required <= set(context):
        raise LLMFeatureDescriptionError("materializer source context is incomplete")
    observed = capture_materializer_source_binding(
        Path(context["selection_path"]), Path(context["atlas_path"]),
        Path(context["span_output_dir"]), Path(context["inference_path"]),
    )
    expected = context["source_binding"]
    if not isinstance(expected, Mapping):
        raise LLMFeatureDescriptionError("materializer source binding is invalid")
    comparable = {key: expected.get(key) for key in observed}
    if observed != comparable:
        raise LLMFeatureDescriptionError("materializer source hash/registry changed")


def _read_sources(context: Mapping[str, object]) -> tuple[dict[int, dict[str, object]], dict[int, dict[str, object]]]:
    verify_materializer_sources(context)
    rows = {int(row["feature_index"]): row for row in strict_json_lines(Path(context["atlas_path"]))}
    span_manifest = strict_json_load(Path(context["span_output_dir"]) / "manifest.json")
    spans = {
        int(item["feature_index"]): strict_json_load(Path(context["span_output_dir"]) / str(item["path"]))
        for item in span_manifest["final_outputs"]
    }
    verify_materializer_sources(context)
    return rows, spans


def materialize_train_packets(context: Mapping[str, object]) -> list[dict[str, object]]:
    rows, spans = _read_sources(context)
    return [build_train_packet(rows[feature], spans[feature]) for feature in context["family_order"]]


def materialize_validation_packets(context: Mapping[str, object]) -> dict[int, dict[str, object]]:
    rows, spans = _read_sources(context)
    return {feature: build_validation_packet(rows[feature], spans[feature]) for feature in context["family_order"]}


def _prepare_production_context(
    selection_path: Path,
    atlas_path: Path,
    span_output_dir: Path,
    inference_path: Path,
    *,
    config: FeatureAtlasConfig,
    provenance: Mapping[str, object] | None,
    allow_test_fixture_unbound: bool,
    current_bindings: Mapping[str, object] | None,
    sae: object | None,
    plans_by_feature: Mapping[int, object] | None,
) -> dict[str, object]:
    if current_bindings is None or sae is None or plans_by_feature is None:
        raise LLMFeatureDescriptionError(
            "Task 9 current bindings, SAE, and frozen plans are required before validation"
        )
    selection = validate_selection_manifest(
        selection_path, atlas_path, inference_path, config,
        provenance=provenance,
        allow_test_fixture_unbound=allow_test_fixture_unbound,
    )
    family = derive_feature_family(selection)
    source_before = capture_materializer_source_binding(
        selection_path, atlas_path, span_output_dir, inference_path
    )
    rows = {int(row["feature_index"]): row for row in strict_json_lines(atlas_path)}
    span_manifest = validate_span_attribution(
        span_output_dir, current_bindings=current_bindings, atlas_rows=rows,
        sae=sae, plans_by_feature=plans_by_feature,
    )
    validate_task9_family(
        span_manifest, selected_features=family, current_bindings=current_bindings,
        sae=sae, plans_by_feature=plans_by_feature,
    )
    spans = {
        int(item["feature_index"]): strict_json_load(span_output_dir / str(item["path"]))
        for item in span_manifest["final_outputs"]
    }
    identities = [_validate_identity(_statistical_identity(rows[feature], spans[feature])) for feature in family]
    source_binding = capture_materializer_source_binding(
        selection_path, atlas_path, span_output_dir, inference_path
    )
    if source_binding != source_before:
        raise LLMFeatureDescriptionError(
            "Task 8/9 sources changed during production context validation"
        )
    source_binding["config_fingerprint"] = config.fingerprint()
    return {
        "atlas_path": atlas_path,
        "family_order": family,
        "identities": identities,
        "inference_path": inference_path,
        "selection_path": selection_path,
        "source_binding": source_binding,
        "span_output_dir": span_output_dir,
    }


def output_fingerprint(artifacts: Sequence[Mapping[str, object]]) -> str:
    return canonical_sha256(list(artifacts))


def file_registration(path: Path, descriptor: Mapping[str, object]) -> dict[str, object]:
    return {
        "bytes": path.stat().st_size,
        "feature_index": descriptor.get("feature_index"),
        "kind": descriptor.get("kind"),
        "path": path.name,
        "provider": descriptor.get("provider"),
        "round": descriptor.get("round"),
        "sequence": descriptor.get("sequence"),
        "sha256": sha256_file(path),
    }


def _new_manifest(
    family: Sequence[int], identities: Sequence[Mapping[str, object]],
    source_binding: Mapping[str, object], provider_bindings: Sequence[Mapping[str, object]],
    provider_mode: str, label_binding: Mapping[str, object], max_calls: int,
) -> dict[str, object]:
    active_names = tuple(str(row["provider"]) for row in provider_bindings)
    return {
        "artifacts": [], "complete": False, "draft_freeze": None,
        "family_order": list(family), "input_fingerprint": None,
        "label_binding": _clone(label_binding), "max_calls": max_calls,
        "max_attempts_per_phase": MAX_ATTEMPTS,
        "output_fingerprint": output_fingerprint([]), "phase": "train",
        "planned_calls": len(family) * len(active_names) * 2,
        "provider_bindings": _clone(provider_bindings),
        "provider_mode": provider_mode,
        "provider_statuses": _initial_provider_statuses(provider_mode),
        "schema_version": SCHEMA_VERSION, "source_binding": _clone(source_binding),
        "statistical_identities": _clone(identities), "train_freeze": None,
        "validation_freeze": None,
    }


def _publish_manifest(output_dir: Path, manifest: Mapping[str, object]) -> None:
    write_json_atomic(output_dir / "manifest.json", manifest)


def _event(hook: object | None, name: str, context: Mapping[str, object]) -> None:
    if hook is not None:
        hook(name, dict(context))  # type: ignore[operator]


def _register_artifact(
    output_dir: Path,
    manifest: dict[str, object],
    path_name: str,
    payload: Mapping[str, object],
    descriptor: Mapping[str, object],
    *,
    event_hook: object | None,
) -> dict[str, object]:
    path = output_dir / path_name
    registered = {row["path"]: row for row in manifest["artifacts"]}
    if path_name in registered:
        if strict_json_load(path) != dict(payload):
            raise LLMFeatureDescriptionError(f"registered artifact changed: {path_name}")
        return registered[path_name]
    if path.exists():
        raise LLMFeatureDescriptionError(f"unregistered artifact collision: {path_name}")
    metadata: dict[str, object] | None = None
    try:
        write_json_atomic(path, payload)
        _event(event_hook, "artifact_renamed", {**descriptor, "path": path_name})
        metadata = file_registration(path, descriptor)
        updated = copy.deepcopy(manifest)
        updated["artifacts"].append(metadata)
        updated["output_fingerprint"] = output_fingerprint(updated["artifacts"])
        _publish_manifest(output_dir, updated)
        _event(event_hook, "artifact_manifest_committed", descriptor)
        manifest.clear(); manifest.update(updated)
    except BaseException as error:
        if metadata is not None:
            try:
                installed = strict_json_load(output_dir / "manifest.json")
                if (
                    metadata in installed.get("artifacts", [])
                    and path.exists()
                    and file_registration(path, metadata) == metadata
                ):
                    manifest.clear(); manifest.update(installed)
                    return metadata
            except BaseException:
                pass
        if path.exists() and path_name not in {
            row.get("path") for row in manifest.get("artifacts", [])
        }:
            try:
                path.unlink()
            except BaseException as rollback_error:
                raise LLMFeatureDescriptionError(
                    f"post-rename publication rollback requires recovery at {path}: {rollback_error}"
                ) from rollback_error
        if isinstance(error, (KeyboardInterrupt, SystemExit, RuntimeError)):
            raise
        if isinstance(error, LLMFeatureDescriptionError):
            raise
        raise LLMFeatureDescriptionError(f"artifact publication failed after rename: {error}") from error
    event_name = {
        "draft_attempt": "draft_attempt_registered",
        "draft_status": "draft_status_registered",
        "verdict_attempt": "verdict_attempt_registered",
        "verdict_status": "verdict_status_registered",
    }.get(str(descriptor["kind"]), f"{descriptor['kind']}_registered")
    _event(event_hook, event_name, descriptor)
    return manifest["artifacts"][-1]


def _artifact_rows(
    manifest: Mapping[str, object], kind: str, *, feature: int | None = None,
    provider: str | None = None,
) -> list[dict[str, object]]:
    return [
        row for row in manifest["artifacts"]
        if row["kind"] == kind
        and (feature is None or row["feature_index"] == feature)
        and (provider is None or row["provider"] == provider)
    ]


def load_phase_history(
    output_dir: Path, feature: int, provider: str, phase: str
) -> dict[str, list[dict[str, object]]]:
    manifest = strict_json_load(output_dir / "manifest.json")
    attempts = [
        strict_json_load(output_dir / row["path"])
        for row in _artifact_rows(manifest, f"{phase}_attempt", feature=feature, provider=provider)
    ]
    statuses = [
        strict_json_load(output_dir / row["path"])
        for row in _artifact_rows(manifest, f"{phase}_status", feature=feature, provider=provider)
    ]
    attempts.sort(key=lambda row: (row["round"], row["sequence"]))
    statuses.sort(key=lambda row: row["round"])
    return {"attempts": attempts, "statuses": statuses}


def _parse_attempt(
    raw: Mapping[str, object], parser: Callable[[str], dict[str, object]],
    *, feature: int, provider: str, phase: str, round_number: int, sequence: int,
) -> dict[str, object]:
    attempt = dict(raw)
    if set(attempt) != _ATTEMPT_BASE_KEYS:
        raise LLMFeatureDescriptionError("provider returned invalid attempt schema")
    parsed = None
    error = attempt.get("parse_error")
    if attempt.get("returncode") == 0:
        try:
            parsed = parser(str(attempt["stdout"]))
            error = None
        except (LLMFeatureDescriptionError, TypeError, ValueError) as parse_error:
            error = str(parse_error)
    attempt.update({
        "artifact_kind": f"{phase}_attempt", "feature_index": feature,
        "parse_error": error, "parsed_response": parsed,
        "parsed_response_sha256": canonical_sha256(parsed) if parsed is not None else None,
        "error_sha256": text_sha256(str(error)) if error is not None else None,
        "phase": phase, "provider": provider, "round": round_number,
        "sequence": sequence,
    })
    return attempt


def _latest_success(statuses: Sequence[Mapping[str, object]]) -> Mapping[str, object] | None:
    return next((row for row in reversed(statuses) if row.get("status") == "ok"), None)


def _reconstruct_provider_statuses(
    family: Sequence[int], active_names: Sequence[str],
    status_lookup: Callable[[int, str, str], Sequence[Mapping[str, object]]],
) -> dict[str, str]:
    mode = _provider_mode_from_names(active_names)
    reconstructed = _initial_provider_statuses(mode)
    for provider in active_names:
        if all(
            _latest_success(status_lookup(feature, provider, phase)) is not None
            for feature in family
            for phase in ("draft", "verdict")
        ):
            reconstructed[provider] = "completed"
    return reconstructed


def _run_phase(
    output_dir: Path, manifest: dict[str, object], *, feature: int,
    provider: LLMProvider, phase: str, prompt: str,
    parser: Callable[[str], dict[str, object]], event_hook: object | None,
) -> dict[str, object]:
    history = load_phase_history(output_dir, feature, provider.provider, phase)
    attempts, statuses = history["attempts"], history["statuses"]
    if (
        any(row["round"] != 1 for row in attempts)
        or any(row["round"] != 1 for row in statuses)
        or len(attempts) > MAX_ATTEMPTS
        or len(statuses) > 1
    ):
        raise LLMFeatureDescriptionError("phase lifetime attempt/status family exceeds terminal policy")
    successful_attempts = [row for row in attempts if row["parsed_response"] is not None]
    if len(successful_attempts) > 1:
        raise LLMFeatureDescriptionError("phase contains multiple successful attempts")
    if successful_attempts:
        first_success = successful_attempts[0]
        if any(row["sequence"] > first_success["sequence"] for row in attempts):
            raise LLMFeatureDescriptionError("phase contains an attempt after first success")
    if statuses:
        return dict(statuses[0])
    if phase == "draft" and (
        manifest.get("validation_freeze") is not None
        or bool(_artifact_rows(manifest, "validation_packet"))
    ):
        raise LLMFeatureDescriptionError(
            "draft phase is nonterminal after validation materialization"
        )
    current = list(attempts)
    parsed = successful_attempts[0]["parsed_response"] if successful_attempts else None
    for sequence in range(len(current) + 1, MAX_ATTEMPTS + 1):
        if parsed is not None:
            break
        try:
            raw = provider.invoke(prompt)
        except BaseException as error:
            if isinstance(error, (KeyboardInterrupt, SystemExit)):
                raise
            now = _utc_now()
            raw = provider.attempt_record(
                prompt=prompt, command=provider.command(prompt), stdout="",
                stderr=f"provider exception: {type(error).__name__}: {error}",
                returncode=125, started_at=now, ended_at=now, duration_seconds=0.0,
            )
        attempt = _parse_attempt(
            raw, parser, feature=feature, provider=provider.provider, phase=phase,
            round_number=1, sequence=sequence,
        )
        name = f"{phase}_f{feature:04d}_{provider.provider}_r0001_a{sequence:02d}.json"
        _register_artifact(
            output_dir, manifest, name, attempt,
            {"kind": f"{phase}_attempt", "feature_index": feature,
             "provider": provider.provider, "round": 1, "sequence": sequence},
            event_hook=event_hook,
        )
        current.append(attempt)
        if attempt["parsed_response"] is not None:
            parsed = attempt["parsed_response"]
            break
    status = {
        "artifact_kind": f"{phase}_status",
        "attempt_paths": [
            f"{phase}_f{feature:04d}_{provider.provider}_r0001_a{row['sequence']:02d}.json"
            for row in current
        ],
        "feature_index": feature, "parsed": parsed, "phase": phase,
        "provider": provider.provider,
        "response_sha256": canonical_sha256(parsed) if parsed is not None else None,
        "round": 1, "status": "ok" if parsed is not None else "llm_failed",
    }
    name = f"{phase}_f{feature:04d}_{provider.provider}_r0001_status.json"
    _register_artifact(
        output_dir, manifest, name, status,
        {"kind": f"{phase}_status", "feature_index": feature,
         "provider": provider.provider, "round": 1, "sequence": None},
        event_hook=event_hook,
    )
    return status


def _freeze(
    output_dir: Path, manifest: dict[str, object], *, phase: str,
    records: Sequence[Mapping[str, object]], event_hook: object | None,
) -> None:
    generation = len(_artifact_rows(manifest, f"{phase}_freeze")) + 1
    if phase == "draft":
        frozen_records = []
        metadata_by_path = {str(row["path"]): row for row in manifest["artifacts"]}
        for row in records:
            status = strict_json_load(output_dir / str(row["path"]))
            if status["status"] == "ok":
                terminal_sha256 = status["response_sha256"]
            else:
                terminal_sha256 = canonical_sha256([
                    metadata_by_path[path]["sha256"] for path in status["attempt_paths"]
                ])
            frozen_records.append({
                "feature_index": row["feature_index"],
                "provider": row["provider"],
                "status": status["status"],
                "status_path": row["path"],
                "status_sha256": row["sha256"],
                "terminal_sha256": terminal_sha256,
            })
    else:
        frozen_records = [
            {"path": row["path"], "sha256": row["sha256"]} for row in records
        ]
    payload = {
        "artifact_kind": f"{phase}_freeze", "family_order": manifest["family_order"],
        "generation": generation,
        "records": frozen_records,
    }
    existing_path = manifest.get(f"{phase}_freeze")
    if existing_path:
        existing = strict_json_load(output_dir / str(existing_path))
        if existing.get("records") == payload["records"]:
            return
    for row in reversed(_artifact_rows(manifest, f"{phase}_freeze")):
        existing = strict_json_load(output_dir / str(row["path"]))
        if existing.get("records") == payload["records"]:
            manifest[f"{phase}_freeze"] = row["path"]
            manifest["phase"] = (
                "draft" if phase == "train" else
                "validation" if phase == "draft" else "verdict"
            )
            if phase == "validation":
                manifest["input_fingerprint"] = _input_fingerprint(manifest)
            manifest["output_fingerprint"] = output_fingerprint(manifest["artifacts"])
            _publish_manifest(output_dir, manifest)
            _event(event_hook, f"{phase}_family_frozen", {"generation": row["round"]})
            return
    name = f"{phase}_freeze_{generation:04d}.json"
    metadata = _register_artifact(
        output_dir, manifest, name, payload,
        {"kind": f"{phase}_freeze", "feature_index": None, "provider": None,
         "round": generation, "sequence": None}, event_hook=event_hook,
    )
    manifest[f"{phase}_freeze"] = metadata["path"]
    manifest["phase"] = "draft" if phase == "train" else "validation" if phase == "draft" else "verdict"
    if phase == "validation":
        manifest["input_fingerprint"] = _input_fingerprint(manifest)
    manifest["output_fingerprint"] = output_fingerprint(manifest["artifacts"])
    _publish_manifest(output_dir, manifest)
    _event(event_hook, f"{phase}_family_frozen", {"generation": generation})


def _input_fingerprint(manifest: Mapping[str, object]) -> str:
    packets = [
        {"kind": row["kind"], "feature_index": row["feature_index"], "sha256": row["sha256"]}
        for row in manifest["artifacts"] if row["kind"] in {"train_packet", "validation_packet"}
    ]
    return canonical_sha256({
        "family_order": manifest["family_order"], "label_binding": manifest["label_binding"],
        "packets": packets, "provider_bindings": manifest["provider_bindings"],
        "provider_mode": manifest["provider_mode"],
        "source_binding": manifest["source_binding"],
        "statistical_identities": manifest["statistical_identities"],
    })


def _artifact_descriptor(payload: Mapping[str, object]) -> dict[str, object]:
    return {
        "kind": payload.get("artifact_kind"), "feature_index": payload.get("feature_index"),
        "provider": payload.get("provider"), "round": payload.get("round"),
        "sequence": payload.get("sequence"),
    }


def _orphan_descriptor_and_name(
    payload: Mapping[str, object],
) -> tuple[dict[str, object], str]:
    kind = payload.get("artifact_kind")
    feature = payload.get("feature_index")
    provider = payload.get("provider")
    round_number = payload.get("round")
    sequence = payload.get("sequence")
    if kind == "train_packet" and type(feature) is int:
        expected = f"train_feature_{feature:04d}.json"
    elif kind == "validation_packet" and type(feature) is int:
        expected = f"validation_feature_{feature:04d}.json"
    elif kind in {"draft_attempt", "verdict_attempt"} and (
        type(feature) is int and provider in _KNOWN_PROVIDERS
        and type(round_number) is int and type(sequence) is int
    ):
        phase = str(kind).removesuffix("_attempt")
        expected = f"{phase}_f{feature:04d}_{provider}_r{round_number:04d}_a{sequence:02d}.json"
    elif kind in {"draft_status", "verdict_status"} and (
        type(feature) is int and provider in _KNOWN_PROVIDERS and type(round_number) is int
    ):
        phase = str(kind).removesuffix("_status")
        expected = f"{phase}_f{feature:04d}_{provider}_r{round_number:04d}_status.json"
        sequence = None
    elif kind in {"train_freeze", "draft_freeze", "validation_freeze"} and type(payload.get("generation")) is int:
        phase = str(kind).removesuffix("_freeze")
        round_number = int(payload["generation"])
        feature = provider = sequence = None
        expected = f"{phase}_freeze_{round_number:04d}.json"
    elif kind == "agreement" and type(feature) is int:
        provider = round_number = sequence = None
        expected = f"agreement_feature_{feature:04d}.json"
    elif kind == "feature" and type(feature) is int:
        provider = round_number = sequence = None
        expected = f"feature_{feature:04d}.json"
    else:
        raise LLMFeatureDescriptionError("unregistered artifact has invalid identity schema")
    return ({
        "kind": kind, "feature_index": feature, "provider": provider,
        "round": round_number, "sequence": sequence,
    }, expected)


def _adopt_orphans(output_dir: Path, manifest: dict[str, object]) -> None:
    registered = {str(row["path"]) for row in manifest["artifacts"]}
    orphan_names = sorted(
        path.name for path in output_dir.iterdir()
        if _RESERVED.match(path.name) and path.name not in registered
    )
    for name in orphan_names:
        path = output_dir / name
        payload = strict_json_load(path)
        descriptor, expected = _orphan_descriptor_and_name(payload)
        if expected != name:
            raise LLMFeatureDescriptionError(
                f"unregistered artifact identity/path collision: {name}"
            )
        metadata = file_registration(path, descriptor)
        manifest["artifacts"].append(metadata)
        manifest["output_fingerprint"] = output_fingerprint(manifest["artifacts"])
        _publish_manifest(output_dir, manifest)


def _extract_codex_jsonl_audit(stderr: object) -> str:
    delimiter = "[codex-jsonl-audit]"
    if type(stderr) is not str or stderr.count(delimiter) != 1:
        raise LLMFeatureDescriptionError(
            "Codex JSONL audit block is missing or duplicated"
        )
    prefix, suffix = stderr.split(delimiter, 1)
    if (prefix and not prefix.endswith("\n")) or not suffix.startswith("\nsha256="):
        raise LLMFeatureDescriptionError("Codex JSONL audit framing is invalid")
    digest_line, separator, raw_jsonl = suffix[1:].partition("\n")
    claimed = digest_line.removeprefix("sha256=")
    if (
        not separator
        or not digest_line.startswith("sha256=")
        or not _SHA.fullmatch(claimed)
        or claimed != text_sha256(raw_jsonl)
    ):
        raise LLMFeatureDescriptionError("Codex JSONL audit SHA-256 is invalid")
    return raw_jsonl


def _validate_codex_attempt_audit(payload: Mapping[str, object]) -> None:
    raw_jsonl = _extract_codex_jsonl_audit(payload.get("stderr"))
    returncode = payload.get("returncode")
    if type(returncode) is not int:
        raise LLMFeatureDescriptionError("Codex JSONL audit return code is invalid")
    if returncode == 0:
        try:
            message = parse_codex_exec_jsonl(raw_jsonl)
        except CodexCLITransportError as error:
            raise LLMFeatureDescriptionError(
                "Codex JSONL audit is invalid for a successful attempt"
            ) from error
        if message != payload.get("stdout"):
            raise LLMFeatureDescriptionError(
                "Codex JSONL audit message differs from successful attempt stdout"
            )
        return

    expected_error = f"nonzero return code: {returncode}"
    if (
        payload.get("stdout") != ""
        or payload.get("parsed_response") is not None
        or payload.get("parse_error") != expected_error
        or payload.get("error_sha256") != text_sha256(expected_error)
    ):
        raise LLMFeatureDescriptionError(
            "Codex JSONL audit differs from native nonzero attempt state"
        )

    if returncode == 65:
        try:
            parse_codex_exec_jsonl(raw_jsonl)
        except CodexCLITransportError:
            return
        raise LLMFeatureDescriptionError(
            "Codex JSONL audit code 65 did not record parser rejection"
        )

    try:
        parse_codex_exec_jsonl(raw_jsonl)
    except CodexCLITransportError:
        pass


def _validate_attempt(
    payload: Mapping[str, object], binding: Mapping[str, object], parser: Callable[[str], dict[str, object]],
) -> None:
    if set(payload) != _ATTEMPT_KEYS:
        raise LLMFeatureDescriptionError("attempt has invalid exact schema")
    for key in _BINDING_BASE_KEYS:
        if payload.get(key) != binding.get(key):
            raise LLMFeatureDescriptionError(f"attempt provider binding changed: {key}")
    expected_command = (
        codex_exec_argv(str(binding["executable"]), str(binding["model"]))
        if binding.get("provider") == "gpt"
        else [
            binding["executable"], "--model", binding["model"],
            *binding["params"], payload["prompt"],
        ]
    )
    if payload.get("command") != expected_command:
        raise LLMFeatureDescriptionError("attempt argv differs from provider binding")
    if (
        not _UTC.fullmatch(str(payload.get("started_at")))
        or not _UTC.fullmatch(str(payload.get("ended_at")))
        or type(payload.get("duration_seconds")) is not float
        or not math.isfinite(payload["duration_seconds"])
        or payload["duration_seconds"] < 0
    ):
        raise LLMFeatureDescriptionError("attempt UTC timing provenance is invalid")
    start_time = datetime.strptime(str(payload["started_at"]), "%Y-%m-%dT%H:%M:%S.%fZ")
    end_time = datetime.strptime(str(payload["ended_at"]), "%Y-%m-%dT%H:%M:%S.%fZ")
    if end_time < start_time:
        raise LLMFeatureDescriptionError("attempt UTC end precedes start")
    checks = {
        "prompt_sha256": text_sha256(str(payload["prompt"])),
        "stdout_sha256": text_sha256(str(payload["stdout"])),
        "stderr_sha256": text_sha256(str(payload["stderr"])),
        "raw_response_sha256": text_sha256(str(payload["raw_response"])),
        "parsed_response_sha256": canonical_sha256(payload["parsed_response"]) if payload["parsed_response"] is not None else None,
        "error_sha256": text_sha256(str(payload["parse_error"])) if payload["parse_error"] is not None else None,
    }
    if any(payload.get(key) != value for key, value in checks.items()) or payload["raw_response"] != payload["stdout"]:
        raise LLMFeatureDescriptionError("attempt raw/parsed/error provenance hash changed")
    if binding.get("provider") == "gpt":
        _validate_codex_attempt_audit(payload)
    if payload["parsed_response"] is not None:
        if parser(str(payload["stdout"])) != payload["parsed_response"] or payload["parse_error"] is not None:
            raise LLMFeatureDescriptionError("attempt parsed response differs from raw stdout")
    elif type(payload["parse_error"]) is not str or not payload["parse_error"]:
        raise LLMFeatureDescriptionError("failed attempt lacks exact error provenance")


def validate_llm_feature_descriptions(
    output_dir: Path,
    *,
    label_embedder: object,
    expected_statistical_identities: Sequence[Mapping[str, object]] | None = None,
    expected_source_binding: Mapping[str, object] | None = None,
    expected_provider_bindings: Sequence[Mapping[str, object]] | None = None,
    allow_test_fixture_unbound: bool = False,
    selection_path: Path | None = None,
    atlas_path: Path | None = None,
    span_output_dir: Path | None = None,
    inference_path: Path | None = None,
    config: FeatureAtlasConfig | None = None,
    provenance: Mapping[str, object] | None = None,
    current_bindings: Mapping[str, object] | None = None,
    sae: object | None = None,
    plans_by_feature: Mapping[int, object] | None = None,
    _canonicalize_provider_statuses: bool = False,
) -> dict[str, object]:
    supplied_paths = (selection_path, atlas_path, span_output_dir, inference_path)
    if any(path is not None for path in supplied_paths):
        if not all(isinstance(path, Path) for path in supplied_paths):
            raise TypeError("standalone production validation requires all Task 8/9 paths")
        context = _prepare_production_context(
            selection_path, atlas_path, span_output_dir, inference_path,  # type: ignore[arg-type]
            config=config or FeatureAtlasConfig(), provenance=provenance,
            allow_test_fixture_unbound=allow_test_fixture_unbound,
            current_bindings=current_bindings, sae=sae,
            plans_by_feature=plans_by_feature,
        )
        expected_statistical_identities = context["identities"]
        expected_source_binding = context["source_binding"]
    if not allow_test_fixture_unbound and (
        expected_statistical_identities is None or expected_source_binding is None
    ):
        raise LLMFeatureDescriptionError(
            "production validation requires canonical Task 8/9 statistical/source bindings"
        )
    try:
        manifest = strict_json_load(Path(output_dir) / "manifest.json")
    except (AtlasIOError, OSError, ValueError) as error:
        raise LLMFeatureDescriptionError(f"invalid Task 10 manifest: {error}") from error
    if set(manifest) != _MANIFEST_KEYS or manifest.get("schema_version") != SCHEMA_VERSION:
        raise LLMFeatureDescriptionError("Task 10 manifest has invalid exact schema")
    family = manifest.get("family_order")
    if not isinstance(family, list) or family != sorted(set(family)):
        raise LLMFeatureDescriptionError("derived feature family order is invalid")
    identities = manifest.get("statistical_identities")
    if not isinstance(identities, list) or [row.get("feature_index") for row in identities if isinstance(row, dict)] != family:
        raise LLMFeatureDescriptionError("statistical identity family is invalid")
    for row in identities:
        _validate_identity(row)
    if expected_statistical_identities is not None and identities != _clone(expected_statistical_identities):
        raise LLMFeatureDescriptionError("statistical identity differs from validated Task 8/7")
    if expected_source_binding is not None and manifest.get("source_binding") != dict(expected_source_binding):
        raise LLMFeatureDescriptionError("Task 8/9 source binding changed")
    bindings = manifest.get("provider_bindings")
    if not isinstance(bindings, list):
        raise LLMFeatureDescriptionError("provider binding family is invalid")
    bindings = [_validate_binding(row, allow_test_fixture_unbound=allow_test_fixture_unbound) for row in bindings]
    active_names = tuple(str(row["provider"]) for row in bindings)
    mode = _provider_mode_from_names(active_names)
    if manifest.get("provider_mode") != mode:
        raise LLMFeatureDescriptionError("manifest provider mode differs from bindings")
    provider_statuses = manifest.get("provider_statuses")
    if mode == PROVIDER_MODE_GPT_ONLY:
        statuses_valid = (
            isinstance(provider_statuses, dict)
            and set(provider_statuses) == {"gpt", "claude", "gemini"}
            and provider_statuses.get("gpt") in {"pending", "completed"}
            and provider_statuses.get("claude") == "skipped_by_user"
            and provider_statuses.get("gemini") == "skipped_by_user"
        )
    else:
        statuses_valid = (
            isinstance(provider_statuses, dict)
            and set(provider_statuses) == set(_PROVIDERS)
            and provider_statuses.get("claude") in {"pending", "completed"}
            and provider_statuses.get("gemini") in {
                "pending", "completed", "skipped_by_user"
            }
            and (
                mode != PROVIDER_MODE_CLAUDE_ONLY
                or provider_statuses.get("gemini") == "skipped_by_user"
            )
            and (
                mode != PROVIDER_MODE_DUAL
                or provider_statuses.get("gemini") != "skipped_by_user"
            )
        )
    if not statuses_valid:
        raise LLMFeatureDescriptionError("manifest provider statuses are invalid")
    if expected_provider_bindings is not None and bindings != _clone(expected_provider_bindings):
        raise LLMFeatureDescriptionError("provider binding differs from caller")
    binding_by_provider = {str(row["provider"]): row for row in bindings}
    if manifest.get("label_binding") != _label_binding(label_embedder, allow_test_fixture_unbound=allow_test_fixture_unbound):
        raise LLMFeatureDescriptionError("frozen ProTrek label binding changed")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or any(not isinstance(row, dict) or set(row) != _ARTIFACT_KEYS for row in artifacts):
        raise LLMFeatureDescriptionError("artifact journal has invalid exact schema")
    if len({row["path"] for row in artifacts}) != len(artifacts):
        raise LLMFeatureDescriptionError("artifact journal paths are duplicated")
    if manifest.get("output_fingerprint") != output_fingerprint(artifacts):
        raise LLMFeatureDescriptionError("output fingerprint does not reconstruct")
    payload_by_path: dict[str, dict[str, object]] = {}
    for metadata in artifacts:
        path = Path(output_dir) / str(metadata["path"])
        if file_registration(path, metadata) != metadata:
            raise LLMFeatureDescriptionError("registered artifact bytes/checksum changed")
        payload = strict_json_load(path)
        if payload.get("artifact_kind") != metadata["kind"]:
            raise LLMFeatureDescriptionError("artifact kind differs from journal")
        descriptor, expected_name = _orphan_descriptor_and_name(payload)
        if expected_name != metadata["path"] or any(
            metadata[name] != descriptor[name]
            for name in ("kind", "feature_index", "provider", "round", "sequence")
        ):
            raise LLMFeatureDescriptionError("artifact identity differs from journal/path")
        payload_by_path[str(metadata["path"])] = payload
    reserved = {path.name for path in Path(output_dir).iterdir() if _RESERVED.match(path.name)}
    if reserved != set(payload_by_path):
        raise LLMFeatureDescriptionError("unregistered/colliding Task 10 artifact exists")
    train_by_feature, validation_by_feature = {}, {}
    agreement_by_feature: dict[int, dict[str, object]] = {}
    feature_payloads: dict[int, dict[str, object]] = {}
    statuses: dict[tuple[int, str, str], list[dict[str, object]]] = {}
    for metadata in artifacts:
        payload = payload_by_path[str(metadata["path"])]
        kind = metadata["kind"]
        if kind == "train_packet":
            packet = _validate_train_packet(payload.get("packet"))
            if set(payload) != {"artifact_kind", "feature_index", "packet", "packet_sha256", "prompt", "prompt_sha256"}:
                raise LLMFeatureDescriptionError("train packet artifact schema is invalid")
            prompt = build_draft_prompt(packet)
            if payload["packet_sha256"] != canonical_sha256(packet) or payload["prompt"] != prompt or payload["prompt_sha256"] != text_sha256(prompt):
                raise LLMFeatureDescriptionError("train packet/prompt freeze changed")
            train_by_feature[int(payload["feature_index"])] = payload
        elif kind == "validation_packet":
            packet = _validate_validation_packet(payload.get("packet"))
            if set(payload) != {"artifact_kind", "feature_index", "packet", "packet_sha256"} or payload["packet_sha256"] != canonical_sha256(packet):
                raise LLMFeatureDescriptionError("validation packet freeze changed")
            validation_by_feature[int(payload["feature_index"])] = payload
        elif kind in {"draft_attempt", "verdict_attempt"}:
            phase = str(payload["phase"])
            expected_phase = str(kind).removesuffix("_attempt")
            if (
                phase != expected_phase
                or payload.get("feature_index") not in family
                or payload.get("provider") not in active_names
                or type(payload.get("round")) is not int or payload["round"] <= 0
                or type(payload.get("sequence")) is not int
                or not 1 <= payload["sequence"] <= MAX_ATTEMPTS
            ):
                raise LLMFeatureDescriptionError("attempt phase/identity/domain is invalid")
            _validate_attempt(payload, binding_by_provider[str(payload["provider"])], parse_draft if phase == "draft" else parse_verdict)
            expected_prompt = (
                train_by_feature[int(payload["feature_index"])]["prompt"]
                if phase == "draft" else None
            )
            if phase == "draft" and payload["prompt"] != expected_prompt:
                raise LLMFeatureDescriptionError("draft prompt differs from frozen train packet")
        elif kind in {"draft_status", "verdict_status"}:
            if set(payload) != _STATUS_KEYS:
                raise LLMFeatureDescriptionError("phase status has invalid exact schema")
            expected_phase = str(kind).removesuffix("_status")
            if (
                payload.get("phase") != expected_phase
                or payload.get("provider") not in active_names
                or payload.get("feature_index") not in family
                or type(payload.get("round")) is not int or payload["round"] <= 0
                or payload.get("status") not in {"ok", "llm_failed"}
                or not isinstance(payload.get("attempt_paths"), list)
                or not 1 <= len(payload["attempt_paths"]) <= MAX_ATTEMPTS
                or len(payload["attempt_paths"]) != len(set(payload["attempt_paths"]))
            ):
                raise LLMFeatureDescriptionError("phase status identity/domain is invalid")
            key = (int(payload["feature_index"]), str(payload["provider"]), str(payload["phase"]))
            statuses.setdefault(key, []).append(payload)
        elif kind == "agreement":
            if set(payload) != {"agreement", "artifact_kind", "feature_index"} or payload.get("feature_index") not in family or not isinstance(payload.get("agreement"), dict):
                raise LLMFeatureDescriptionError("agreement artifact has invalid exact schema")
            feature = int(payload["feature_index"])
            if feature in agreement_by_feature:
                raise LLMFeatureDescriptionError("agreement feature is duplicated")
            agreement_by_feature[feature] = payload
        elif kind == "feature":
            exact_feature = {
                "agreement", "agreement_path", "artifact_kind", "feature_index",
                "provider_status_paths", "provider_statuses", "statistical_identity",
                "train_packet_path", "validation_packet_path",
            }
            if set(payload) != exact_feature or payload.get("feature_index") not in family:
                raise LLMFeatureDescriptionError("feature artifact has invalid exact schema")
            feature = int(payload["feature_index"])
            if feature in feature_payloads:
                raise LLMFeatureDescriptionError("final feature is duplicated")
            feature_payloads[feature] = payload
        elif kind in {"train_freeze", "draft_freeze", "validation_freeze"}:
            if set(payload) != {"artifact_kind", "family_order", "generation", "records"}:
                raise LLMFeatureDescriptionError("freeze artifact has invalid exact schema")
        else:
            raise LLMFeatureDescriptionError(f"unknown registered artifact kind: {kind}")
    metadata_by_path = {str(row["path"]): row for row in artifacts}
    freeze_payloads: dict[str, list[dict[str, object]]] = {}
    for phase in ("train", "draft", "validation"):
        freeze_rows = [row for row in artifacts if row["kind"] == f"{phase}_freeze"]
        freeze_rows.sort(key=lambda row: int(row["round"]))
        pointer = manifest.get(f"{phase}_freeze")
        if bool(pointer) != bool(freeze_rows):
            raise LLMFeatureDescriptionError(
                f"{phase} freeze pointer/artifact bijection is invalid"
            )
        if freeze_rows:
            generations = [row["round"] for row in freeze_rows]
            if generations != list(range(1, len(freeze_rows) + 1)):
                raise LLMFeatureDescriptionError(
                    f"{phase} freeze generations are duplicated or non-contiguous"
                )
            if pointer != freeze_rows[-1]["path"]:
                raise LLMFeatureDescriptionError(
                    f"{phase} freeze pointer does not resolve to latest generation"
                )
            payloads = [payload_by_path[str(row["path"])] for row in freeze_rows]
            if any(
                payload["generation"] != row["round"]
                or payload["artifact_kind"] != f"{phase}_freeze"
                for row, payload in zip(freeze_rows, payloads)
            ):
                raise LLMFeatureDescriptionError(
                    f"{phase} freeze generation/kind differs from journal"
                )
            freeze_payloads[phase] = payloads
        else:
            freeze_payloads[phase] = []
    for phase, packet_kind in (
        ("train", "train_packet"), ("validation", "validation_packet"),
    ):
        freezes = freeze_payloads[phase]
        if freezes:
            freezes.sort(key=lambda row: int(row["generation"]))
            latest = freezes[-1]
            if (
                set(latest) != {"artifact_kind", "family_order", "generation", "records"}
                or latest["family_order"] != family
            ):
                raise LLMFeatureDescriptionError(f"{phase} freeze schema/family pointer is invalid")
            expected_records = [
                {"path": row["path"], "sha256": row["sha256"]}
                for row in artifacts if row["kind"] == packet_kind
            ]
            if latest["records"] != expected_records:
                raise LLMFeatureDescriptionError(f"{phase} freeze records do not match packet family")
    draft_freezes = freeze_payloads["draft"]
    if draft_freezes:
        draft_freezes.sort(key=lambda row: int(row["generation"]))
        latest = draft_freezes[-1]
        if (
            set(latest) != {"artifact_kind", "family_order", "generation", "records"}
            or latest["family_order"] != family
            or len(latest["records"]) != len(family) * len(active_names)
        ):
            raise LLMFeatureDescriptionError("draft freeze schema/family pointer is invalid")
        covered: set[tuple[int, str]] = set()
        for record in latest["records"]:
            if (
                not isinstance(record, dict)
                or set(record) != {
                    "feature_index", "provider", "status", "status_path",
                    "status_sha256", "terminal_sha256",
                }
                or record.get("status_path") not in metadata_by_path
                or metadata_by_path[record["status_path"]]["kind"] != "draft_status"
                or metadata_by_path[record["status_path"]]["sha256"]
                != record.get("status_sha256")
            ):
                raise LLMFeatureDescriptionError("draft freeze record is not a registered status")
            meta = metadata_by_path[record["status_path"]]
            status_payload = payload_by_path[record["status_path"]]
            if (
                record["feature_index"] != meta["feature_index"]
                or record["provider"] != meta["provider"]
                or record["status"] != status_payload["status"]
            ):
                raise LLMFeatureDescriptionError("draft freeze terminal identity/status changed")
            expected_terminal = (
                status_payload["response_sha256"]
                if status_payload["status"] == "ok"
                else canonical_sha256([
                    metadata_by_path[path]["sha256"]
                    for path in status_payload["attempt_paths"]
                ])
            )
            if record["terminal_sha256"] != expected_terminal:
                raise LLMFeatureDescriptionError("draft freeze terminal hash changed")
            covered.add((int(meta["feature_index"]), str(meta["provider"])))
        if covered != {(feature, provider) for feature in family for provider in active_names}:
            raise LLMFeatureDescriptionError("draft freeze does not cover the provider family")
    train_features = sorted(train_by_feature)
    validation_features = sorted(validation_by_feature)
    has_draft = any(row["kind"].startswith("draft_") for row in artifacts)
    has_verdict = any(row["kind"].startswith("verdict_") for row in artifacts)
    if manifest.get("train_freeze") is not None and train_features != family:
        raise LLMFeatureDescriptionError("train freeze does not cover the selected family")
    if has_draft and manifest.get("train_freeze") is None:
        raise LLMFeatureDescriptionError("draft evidence exists before train freeze")
    if validation_features and manifest.get("draft_freeze") is None:
        raise LLMFeatureDescriptionError("validation packets exist before draft freeze")
    if manifest.get("validation_freeze") is not None and validation_features != family:
        raise LLMFeatureDescriptionError("validation freeze does not cover the selected family")
    if has_verdict and manifest.get("validation_freeze") is None:
        raise LLMFeatureDescriptionError("verdict evidence exists before validation freeze")
    validation_positions = [
        index for index, row in enumerate(artifacts)
        if row["kind"] == "validation_packet"
    ]
    if validation_positions and any(
        index > min(validation_positions) and row["kind"] == "draft_attempt"
        for index, row in enumerate(artifacts)
    ):
        raise LLMFeatureDescriptionError("draft attempt was registered after validation materialization")
    attempt_groups: dict[tuple[int, str, str], list[dict[str, object]]] = {}
    for row in artifacts:
        if row["kind"] not in {"draft_attempt", "verdict_attempt"}:
            continue
        payload = payload_by_path[str(row["path"])]
        key = (int(row["feature_index"]), str(row["provider"]), str(payload["phase"]))
        attempt_groups.setdefault(key, []).append(payload)
    for key, attempts_for_phase in attempt_groups.items():
        attempts_for_phase.sort(key=lambda row: int(row["sequence"]))
        if (
            len(attempts_for_phase) > MAX_ATTEMPTS
            or any(row["round"] != 1 for row in attempts_for_phase)
            or [row["sequence"] for row in attempts_for_phase]
            != list(range(1, len(attempts_for_phase) + 1))
        ):
            raise LLMFeatureDescriptionError(
                f"phase lifetime attempt cap/order changed: {key}"
            )
        successes = [row for row in attempts_for_phase if row["parsed_response"] is not None]
        if len(successes) > 1 or (
            successes
            and any(row["sequence"] > successes[0]["sequence"] for row in attempts_for_phase)
        ):
            raise LLMFeatureDescriptionError("phase contains attempts after first success")
    for values in statuses.values():
        values.sort(key=lambda row: int(row["round"]))
        if len(values) != 1 or values[0]["round"] != 1:
            raise LLMFeatureDescriptionError("phase must have one lifetime terminal status")
        for status in values:
            expected_attempt_rows = sorted(
                (
                    row for row in artifacts
                    if row["kind"] == f"{status['phase']}_attempt"
                    and row["feature_index"] == status["feature_index"]
                    and row["provider"] == status["provider"]
                    and row["round"] == status["round"]
                ),
                key=lambda row: int(row["sequence"]),
            )
            expected_paths = [str(row["path"]) for row in expected_attempt_rows]
            if status["attempt_paths"] != expected_paths or [row["sequence"] for row in expected_attempt_rows] != list(range(1, len(expected_attempt_rows) + 1)):
                raise LLMFeatureDescriptionError("phase status attempt family does not reconstruct")
            attempt_payloads = [payload_by_path[path] for path in expected_paths]
            if any(
                row["round"] != status["round"]
                or row["feature_index"] != status["feature_index"]
                or row["provider"] != status["provider"]
                or row["phase"] != status["phase"]
                for row in attempt_payloads
            ):
                raise LLMFeatureDescriptionError("phase status links another round")
            parsed = next((row["parsed_response"] for row in reversed(attempt_payloads) if row["parsed_response"] is not None), None)
            expected = {"parsed": parsed, "response_sha256": canonical_sha256(parsed) if parsed is not None else None, "status": "ok" if parsed is not None else "llm_failed"}
            if any(status[name] != value for name, value in expected.items()):
                raise LLMFeatureDescriptionError("phase status does not reconstruct from attempts")
            if status["status"] == "llm_failed" and len(attempt_payloads) != MAX_ATTEMPTS:
                raise LLMFeatureDescriptionError(
                    "terminal failed phase does not contain exactly three lifetime attempts"
                )
    if validation_by_feature:
        for feature in family:
            for provider in active_names:
                verdict_values = statuses.get((feature, provider, "verdict"), [])
                for status in verdict_values:
                    for path in status["attempt_paths"]:
                        attempt = payload_by_path[path]
                        draft_status = _latest_success(statuses.get((feature, provider, "draft"), []))
                        if draft_status is None:
                            raise LLMFeatureDescriptionError("verdict exists without same-provider draft")
                        expected_prompt = build_verdict_prompt(
                            draft_status["parsed"], draft_status["response_sha256"],
                            validation_by_feature[feature]["packet"],
                        )
                        if attempt["prompt"] != expected_prompt:
                            raise LLMFeatureDescriptionError("verdict prompt/draft/evidence binding changed")
    for feature, payload in feature_payloads.items():
        expected_identity = identities[family.index(feature)] if feature in family else None
        if payload.get("statistical_identity") != expected_identity:
            raise LLMFeatureDescriptionError("feature statistical identity changed")
        if (
            payload.get("train_packet_path") != f"train_feature_{feature:04d}.json"
            or payload.get("validation_packet_path") != f"validation_feature_{feature:04d}.json"
            or payload.get("agreement_path") != f"agreement_feature_{feature:04d}.json"
            or feature not in agreement_by_feature
            or payload.get("agreement") != agreement_by_feature[feature].get("agreement")
        ):
            raise LLMFeatureDescriptionError("feature packet/agreement links changed")
        provider_results = {}
        expected_status_paths: dict[str, dict[str, str]] = {}
        for provider in active_names:
            draft_status = _latest_success(statuses.get((feature, provider, "draft"), []))
            verdict_status = _latest_success(statuses.get((feature, provider, "verdict"), []))
            if draft_status is None or verdict_status is None:
                raise LLMFeatureDescriptionError("final feature lacks successful provider phases")
            provider_results[provider] = {"draft": draft_status["parsed"], "verdict": verdict_status["parsed"]}
            expected_status_paths[provider] = {
                "draft": f"draft_f{feature:04d}_{provider}_r{draft_status['round']:04d}_status.json",
                "verdict": f"verdict_f{feature:04d}_{provider}_r{verdict_status['round']:04d}_status.json",
            }
        if payload.get("provider_status_paths") != expected_status_paths:
            raise LLMFeatureDescriptionError("feature provider status links changed")
        expected_feature_statuses = _completed_provider_statuses(mode)
        if payload.get("provider_statuses") != expected_feature_statuses:
            raise LLMFeatureDescriptionError("feature provider statuses changed")
        expected_agreement = (
            _gpt_only_agreement()
            if mode == PROVIDER_MODE_GPT_ONLY
            else unavailable_agreement()
            if mode == PROVIDER_MODE_CLAUDE_ONLY
            else compute_llm_agreement(
                provider_results["claude"], provider_results["gemini"],
                validated_concept_ids=set(expected_identity["validated_concept_ids"]),
                label_embedder=label_embedder,
                allow_test_fixture_unbound=allow_test_fixture_unbound,
            )
        )
        if payload.get("agreement") != expected_agreement:
            raise LLMFeatureDescriptionError("feature agreement does not recompute")
    if set(agreement_by_feature) != set(feature_payloads):
        raise LLMFeatureDescriptionError("agreement/final feature families disagree")
    if manifest.get("input_fingerprint") is not None and manifest["input_fingerprint"] != _input_fingerprint(manifest):
        raise LLMFeatureDescriptionError("input fingerprint does not reconstruct")
    actual_attempts = sum(
        row["kind"] in {"draft_attempt", "verdict_attempt"} for row in artifacts
    )
    if (
        type(manifest.get("max_calls")) is not int
        or not 0 < manifest["max_calls"] <= MAX_CALLS
        or manifest.get("planned_calls") != len(family) * len(active_names) * 2
        or manifest.get("max_attempts_per_phase") != MAX_ATTEMPTS
        or actual_attempts > int(manifest["planned_calls"]) * MAX_ATTEMPTS
    ):
        raise LLMFeatureDescriptionError("planned call budget is invalid")
    final_features = sorted(row["feature_index"] for row in artifacts if row["kind"] == "feature")
    if manifest.get("complete") is not (final_features == family):
        raise LLMFeatureDescriptionError("manifest completion does not match final features")
    expected_manifest_statuses = _reconstruct_provider_statuses(
        family,
        active_names,
        lambda feature, provider, phase: statuses.get(
            (feature, provider, phase), ()
        ),
    )
    if manifest["complete"] and manifest.get("phase") != "complete":
        raise LLMFeatureDescriptionError("manifest phase does not match reconstructable state")
    if (
        not manifest["complete"]
        and manifest.get("validation_freeze") is not None
        and manifest.get("phase") not in {"verdict", "agreement", "retryable"}
    ):
        raise LLMFeatureDescriptionError("manifest phase does not match reconstructable state")
    if manifest.get("validation_freeze") is not None and manifest.get("input_fingerprint") is None:
        raise LLMFeatureDescriptionError("validation freeze lacks closed input fingerprint")
    if manifest.get("complete") and any(
        manifest.get(name) is None
        for name in ("train_freeze", "draft_freeze", "validation_freeze", "input_fingerprint")
    ):
        raise LLMFeatureDescriptionError("complete manifest lacks required phase freezes")
    if provider_statuses != expected_manifest_statuses:
        if not _canonicalize_provider_statuses:
            raise LLMFeatureDescriptionError(
                "manifest provider statuses do not match completion state"
            )
        recovered = copy.deepcopy(manifest)
        recovered["provider_statuses"] = expected_manifest_statuses
        _publish_manifest(Path(output_dir), recovered)
        manifest = recovered
    return manifest


def _run_state_machine(
    train_packets: Sequence[Mapping[str, object]], output_dir: Path, *,
    identities: Sequence[Mapping[str, object]], source_binding: Mapping[str, object],
    providers: Sequence[LLMProvider], label_embedder: object,
    validation_materializer: Callable[[object], Mapping[int, Mapping[str, object]]],
    validation_context: object, allow_test_fixture_unbound: bool, max_calls: int,
    event_hook: object | None,
) -> dict[str, object]:
    packets = [_validate_train_packet(dict(packet)) for packet in train_packets]
    by_feature = {int(packet["feature_index"]): packet for packet in packets}
    family = sorted(by_feature)
    if len(by_feature) != len(packets):
        raise LLMFeatureDescriptionError("train packet feature identities are duplicated")
    checked_identities = [_validate_identity(dict(row)) for row in identities]
    checked_identities.sort(key=lambda row: int(row["feature_index"]))
    if [row["feature_index"] for row in checked_identities] != family:
        raise LLMFeatureDescriptionError("statistical/train feature families disagree")
    provider_bindings, mode = validate_provider_family(
        providers,
        allow_test_fixture_unbound=allow_test_fixture_unbound,
        return_mode=True,
    )
    active_names = tuple(str(row["provider"]) for row in provider_bindings)
    planned = len(family) * len(active_names) * 2
    if planned > max_calls:
        raise LLMFeatureDescriptionError(f"LLM preflight budget exceeded: {planned} > {max_calls} (max 800)")
    label_binding = _label_binding(
        label_embedder, allow_test_fixture_unbound=allow_test_fixture_unbound
    )
    provider_by_name = {provider.provider: provider for provider in providers}
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        manifest = strict_json_load(manifest_path)
        _adopt_orphans(output_dir, manifest)
        # A committed terminal status can precede the aggregate status publish.
        # Canonicalize only after the full journal and immutable inputs verify.
        validate_llm_feature_descriptions(
            output_dir, label_embedder=label_embedder,
            expected_statistical_identities=checked_identities,
            expected_source_binding=source_binding,
            expected_provider_bindings=provider_bindings,
            allow_test_fixture_unbound=allow_test_fixture_unbound,
            _canonicalize_provider_statuses=True,
        )
        manifest = validate_llm_feature_descriptions(
            output_dir, label_embedder=label_embedder,
            expected_statistical_identities=checked_identities,
            expected_source_binding=source_binding,
            expected_provider_bindings=provider_bindings,
            allow_test_fixture_unbound=allow_test_fixture_unbound,
        )
    else:
        collisions = [path.name for path in output_dir.iterdir() if path.name == "manifest.json" or _RESERVED.match(path.name)]
        if collisions:
            raise LLMFeatureDescriptionError(f"fresh registered-name collision: {collisions[0]}")
        manifest = _new_manifest(
            family, checked_identities, source_binding, provider_bindings,
            mode, label_binding, max_calls,
        )
        _publish_manifest(output_dir, manifest)
    # Freeze every train packet and prompt before the first draft call.
    train_rows = []
    for feature in family:
        packet, prompt = by_feature[feature], build_draft_prompt(by_feature[feature])
        payload = {
            "artifact_kind": "train_packet", "feature_index": feature,
            "packet": packet, "packet_sha256": canonical_sha256(packet),
            "prompt": prompt, "prompt_sha256": text_sha256(prompt),
        }
        metadata = _register_artifact(
            output_dir, manifest, f"train_feature_{feature:04d}.json", payload,
            {"kind": "train_packet", "feature_index": feature, "provider": None,
             "round": None, "sequence": None}, event_hook=event_hook,
        )
        train_rows.append(metadata)
    _freeze(output_dir, manifest, phase="train", records=train_rows, event_hook=event_hook)
    # Each provider/feature draft is independently journaled.
    draft_status_rows = []
    for feature in family:
        prompt = strict_json_load(output_dir / f"train_feature_{feature:04d}.json")["prompt"]
        for provider_name in active_names:
            status = _run_phase(
                output_dir, manifest, feature=feature,
                provider=provider_by_name[provider_name], phase="draft",
                prompt=str(prompt), parser=parse_draft, event_hook=event_hook,
            )
            status_meta = next(
                row for row in reversed(manifest["artifacts"])
                if row["kind"] == "draft_status" and row["feature_index"] == feature
                and row["provider"] == provider_name and row["round"] == status["round"]
            )
            draft_status_rows.append(status_meta)
    _freeze(output_dir, manifest, phase="draft", records=draft_status_rows, event_hook=event_hook)
    # The validation materializer is first invoked here, after the full draft freeze.
    existing_validation = _artifact_rows(manifest, "validation_packet")
    if not existing_validation:
        _event(event_hook, "validation_materializer_called", {})
        validation_raw = validation_materializer(validation_context)
        if not isinstance(validation_raw, Mapping) or sorted(validation_raw) != family:
            raise LLMFeatureDescriptionError("validation materializer family differs from train freeze")
        validation_rows = []
        for feature in family:
            packet = _validate_validation_packet(dict(validation_raw[feature]))
            payload = {
                "artifact_kind": "validation_packet", "feature_index": feature,
                "packet": packet, "packet_sha256": canonical_sha256(packet),
            }
            validation_rows.append(_register_artifact(
                output_dir, manifest, f"validation_feature_{feature:04d}.json", payload,
                {"kind": "validation_packet", "feature_index": feature, "provider": None,
                 "round": None, "sequence": None}, event_hook=event_hook,
            ))
        _freeze(output_dir, manifest, phase="validation", records=validation_rows, event_hook=event_hook)
        manifest["input_fingerprint"] = _input_fingerprint(manifest)
        manifest["output_fingerprint"] = output_fingerprint(manifest["artifacts"])
        _publish_manifest(output_dir, manifest)
    validation_by_feature = {
        feature: strict_json_load(output_dir / f"validation_feature_{feature:04d}.json")["packet"]
        for feature in family
    }
    # Verdicts run only for providers with successful immutable drafts.
    for feature in family:
        for provider_name in active_names:
            draft_history = load_phase_history(output_dir, feature, provider_name, "draft")
            draft_status = _latest_success(draft_history["statuses"])
            if draft_status is None:
                continue
            prompt = build_verdict_prompt(
                draft_status["parsed"], draft_status["response_sha256"],
                validation_by_feature[feature],
            )
            _run_phase(
                output_dir, manifest, feature=feature,
                provider=provider_by_name[provider_name], phase="verdict",
                prompt=prompt, parser=parse_verdict, event_hook=event_hook,
            )
    # Agreement/final feature files are written once all active providers pass.
    for feature in family:
        results: dict[str, dict[str, object]] = {}
        status_paths: dict[str, dict[str, str]] = {}
        for provider_name in active_names:
            draft_history = load_phase_history(output_dir, feature, provider_name, "draft")
            verdict_history = load_phase_history(output_dir, feature, provider_name, "verdict")
            draft_status = _latest_success(draft_history["statuses"])
            verdict_status = _latest_success(verdict_history["statuses"])
            if draft_status is None or verdict_status is None:
                break
            results[provider_name] = {"draft": draft_status["parsed"], "verdict": verdict_status["parsed"]}
            status_paths[provider_name] = {
                "draft": f"draft_f{feature:04d}_{provider_name}_r{draft_status['round']:04d}_status.json",
                "verdict": f"verdict_f{feature:04d}_{provider_name}_r{verdict_status['round']:04d}_status.json",
            }
        if len(results) != len(active_names):
            continue
        identity_row = checked_identities[family.index(feature)]
        agreement = (
            _gpt_only_agreement()
            if mode == PROVIDER_MODE_GPT_ONLY
            else unavailable_agreement()
            if mode == PROVIDER_MODE_CLAUDE_ONLY
            else compute_llm_agreement(
                results["claude"], results["gemini"],
                validated_concept_ids=set(identity_row["validated_concept_ids"]),
                label_embedder=label_embedder,
                allow_test_fixture_unbound=allow_test_fixture_unbound,
            )
        )
        agreement_payload = {
            "agreement": agreement, "artifact_kind": "agreement",
            "feature_index": feature,
        }
        agreement_meta = _register_artifact(
            output_dir, manifest, f"agreement_feature_{feature:04d}.json", agreement_payload,
            {"kind": "agreement", "feature_index": feature, "provider": None,
             "round": None, "sequence": None}, event_hook=event_hook,
        )
        feature_payload = {
            "agreement": agreement, "agreement_path": agreement_meta["path"],
            "artifact_kind": "feature", "feature_index": feature,
            "provider_status_paths": status_paths,
            "provider_statuses": _completed_provider_statuses(mode),
            "statistical_identity": identity_row,
            "train_packet_path": f"train_feature_{feature:04d}.json",
            "validation_packet_path": f"validation_feature_{feature:04d}.json",
        }
        _register_artifact(
            output_dir, manifest, f"feature_{feature:04d}.json", feature_payload,
            {"kind": "feature", "feature_index": feature, "provider": None,
             "round": None, "sequence": None}, event_hook=event_hook,
        )
    final_features = sorted(row["feature_index"] for row in manifest["artifacts"] if row["kind"] == "feature")
    manifest["complete"] = final_features == family
    manifest["phase"] = "complete" if manifest["complete"] else "retryable"
    manifest["provider_statuses"] = _reconstruct_provider_statuses(
        family,
        active_names,
        lambda feature, provider, phase: load_phase_history(
            output_dir, feature, provider, phase
        )["statuses"],
    )
    manifest["output_fingerprint"] = output_fingerprint(manifest["artifacts"])
    _publish_manifest(output_dir, manifest)
    return validate_llm_feature_descriptions(
        output_dir, label_embedder=label_embedder,
        expected_statistical_identities=checked_identities,
        expected_source_binding=source_binding,
        expected_provider_bindings=provider_bindings,
        allow_test_fixture_unbound=allow_test_fixture_unbound,
    )


def run_feature_descriptions(
    selection_or_train: Path | Sequence[Mapping[str, object]],
    atlas_or_output: Path,
    span_output_dir: Path | None = None,
    output_dir: Path | None = None,
    *,
    providers: Sequence[LLMProvider],
    label_embedder: object,
    inference_path: Path | None = None,
    config: FeatureAtlasConfig | None = None,
    provenance: Mapping[str, object] | None = None,
    statistical_identities: Sequence[Mapping[str, object]] | None = None,
    validation_materializer: Callable[[object], Mapping[int, Mapping[str, object]]] | None = None,
    current_bindings: Mapping[str, object] | None = None,
    sae: object | None = None,
    plans_by_feature: Mapping[int, object] | None = None,
    allow_test_fixture_unbound: bool = False,
    max_calls: int = MAX_CALLS,
    overwrite: bool = False,
    event_hook: object | None = None,
) -> dict[str, object]:
    if type(max_calls) is not int or not 0 < max_calls <= MAX_CALLS:
        raise LLMFeatureDescriptionError("max call budget must be in 1..800")
    if type(overwrite) is not bool or type(allow_test_fixture_unbound) is not bool:
        raise TypeError("run gates must be built-in bool values")
    if isinstance(selection_or_train, Path):
        if span_output_dir is None or output_dir is None or inference_path is None:
            raise TypeError("production run requires selection/atlas/span/output/inference paths")
        context = _prepare_production_context(
            selection_or_train, Path(atlas_or_output), Path(span_output_dir), Path(inference_path),
            config=config or FeatureAtlasConfig(), provenance=provenance,
            allow_test_fixture_unbound=allow_test_fixture_unbound,
            current_bindings=current_bindings, sae=sae, plans_by_feature=plans_by_feature,
        )
        trains = materialize_train_packets(context)
        identities = context["identities"]
        source_binding = context["source_binding"]
        destination = Path(output_dir)
        validation_fn = materialize_validation_packets
        validation_context: object = context
    else:
        if not allow_test_fixture_unbound:
            raise LLMFeatureDescriptionError("direct packets require explicit fixture gate")
        if span_output_dir is not None or output_dir is not None:
            raise TypeError("fixture adapter accepts train packets and output path only")
        if statistical_identities is None or validation_materializer is None:
            raise TypeError("fixture adapter requires identities and lazy validation materializer")
        trains = list(selection_or_train)
        identities = list(statistical_identities)
        source_binding = {
            "kind": "fixture-unbound",
            "train_source_sha256": canonical_sha256(trains),
        }
        destination = Path(atlas_or_output)
        validation_fn = validation_materializer
        validation_context = None
    # Overwrite isolation is a complete sibling stage; normal calls always resume.
    if overwrite and (destination / "manifest.json").exists():
        validate_provider_family(
            providers, allow_test_fixture_unbound=allow_test_fixture_unbound
        )
        old_manifest = strict_json_load(destination / "manifest.json")
        same_source = old_manifest.get("source_binding") == source_binding
        if not same_source and not allow_test_fixture_unbound:
            raise LLMFeatureDescriptionError(
                "production overwrite cannot replace a stage bound to different Task 8/9 inputs"
            )
        validate_llm_feature_descriptions(
            destination, label_embedder=label_embedder,
            expected_statistical_identities=(
                identities if same_source else old_manifest.get("statistical_identities")
            ),
            expected_source_binding=(
                source_binding if same_source else old_manifest.get("source_binding")
            ),
            expected_provider_bindings=old_manifest.get("provider_bindings"),
            allow_test_fixture_unbound=allow_test_fixture_unbound,
        )
        stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}.overwrite-stage-", dir=destination.parent))
        preserve_stage = False
        try:
            _run_state_machine(
                trains, stage, identities=identities, source_binding=source_binding,
                providers=providers, label_embedder=label_embedder,
                validation_materializer=validation_fn, validation_context=validation_context,
                allow_test_fixture_unbound=allow_test_fixture_unbound,
                max_calls=max_calls, event_hook=event_hook,
            )
            _install_overwrite(stage, destination)
        except LLMFeatureDescriptionError as error:
            preserve_stage = "requires recovery" in str(error)
            raise
        finally:
            if not preserve_stage:
                shutil.rmtree(stage, ignore_errors=True)
        return validate_llm_feature_descriptions(
            destination, label_embedder=label_embedder,
            expected_statistical_identities=identities,
            expected_source_binding=source_binding,
            allow_test_fixture_unbound=allow_test_fixture_unbound,
        )
    return _run_state_machine(
        trains, destination, identities=identities, source_binding=source_binding,
        providers=providers, label_embedder=label_embedder,
        validation_materializer=validation_fn, validation_context=validation_context,
        allow_test_fixture_unbound=allow_test_fixture_unbound,
        max_calls=max_calls, event_hook=event_hook,
    )


def _install_overwrite(stage: Path, destination: Path) -> None:
    old = strict_json_load(destination / "manifest.json")
    new = strict_json_load(stage / "manifest.json")
    old_names = {"manifest.json", *(str(row["path"]) for row in old["artifacts"])}
    new_names = {"manifest.json", *(str(row["path"]) for row in new["artifacts"])}
    collisions = [
        name for name in new_names - old_names
        if (destination / name).exists()
    ]
    if collisions:
        raise LLMFeatureDescriptionError(
            f"overwrite would replace unrelated registered-name collision: {collisions[0]}"
        )
    backup = Path(tempfile.mkdtemp(prefix=f".{destination.name}.overwrite-backup-", dir=destination.parent))
    preserve = False
    try:
        for name in old_names:
            shutil.copy2(destination / name, backup / name)
        try:
            for name in sorted(new_names - {"manifest.json"}):
                os.replace(stage / name, destination / name)
            os.replace(stage / "manifest.json", destination / "manifest.json")
            for name in old_names - new_names:
                (destination / name).unlink()
        except BaseException as publish_error:
            try:
                for name in new_names | old_names:
                    (destination / name).unlink(missing_ok=True)
                for path in backup.iterdir():
                    os.replace(path, destination / path.name)
            except BaseException as restore_error:
                preserve = True
                raise LLMFeatureDescriptionError(
                    "overwrite rollback requires recovery; "
                    f"stage: {stage}; backup: {backup}; publish: {publish_error}; restore: {restore_error}"
                ) from restore_error
            raise publish_error
    finally:
        if not preserve:
            shutil.rmtree(backup, ignore_errors=True)


def run_provider_phase(
    provider: LLMProvider, prompt: str,
    parser: Callable[[str], dict[str, object]],
) -> dict[str, object]:
    """Compatibility helper for one in-memory bounded round."""
    attempts = []
    parsed = None
    for sequence in range(1, MAX_ATTEMPTS + 1):
        raw = provider.invoke(prompt)
        attempt = _parse_attempt(
            raw, parser, feature=0, provider=provider.provider,
            phase="draft", round_number=1, sequence=sequence,
        )
        attempts.append(attempt)
        if attempt["parsed_response"] is not None:
            parsed = attempt["parsed_response"]
            break
    return {
        "attempts": attempts, "parsed": parsed,
        "response_sha256": canonical_sha256(parsed) if parsed is not None else None,
        "status": "ok" if parsed is not None else "llm_failed",
    }


__all__ = [
    "CLAUDE_MODEL", "GEMINI_MODEL", "GPT_MODEL", "PROVIDER_MODE_DUAL",
    "PROVIDER_MODE_CLAUDE_ONLY", "PROVIDER_MODE_GPT_ONLY",
    "LLMFeatureDescriptionError", "LLMProvider", "CodexCLITransportError",
    "ClaudeCLIProvider", "GeminiCLIProvider", "CodexCLIProvider",
    "codex_exec_argv", "parse_codex_exec_jsonl", "build_draft_prompt",
    "build_verdict_prompt", "parse_draft", "parse_verdict",
    "derive_feature_family", "build_train_packet", "build_validation_packet",
    "capture_materializer_source_binding", "verify_materializer_sources",
    "materialize_train_packets", "materialize_validation_packets",
    "validate_task9_family", "validate_provider_family", "compute_llm_agreement",
    "unavailable_agreement",
    "run_feature_descriptions", "validate_llm_feature_descriptions",
    "load_phase_history", "file_registration", "output_fingerprint",
    "run_provider_phase", "normalized_label_tokens", "canonical_sha256",
    "text_sha256",
]
