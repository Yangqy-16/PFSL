"""Strict, tool-free transport helpers for frozen Codex CLI execution."""

from __future__ import annotations

import json
import os
import selectors
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


GPT_MODEL = "gpt-5.6-sol"
CODEX_PATH_ALIAS_READ_ONLY_WARNING = (
    "WARNING: proceeding, even though we could not create PATH aliases: "
    "Read-only file system (os error 30)"
)


class CodexCLITransportError(ValueError):
    pass


@dataclass(frozen=True)
class CodexModelIdentity:
    cli_version: str
    model: str
    model_provider: str
    reported_version: str
    resolved_executable: str


def strict_jsonl_objects(raw: str) -> list[dict[str, object]]:
    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        value: dict[str, object] = {}
        for key, item in pairs:
            if key in value:
                raise CodexCLITransportError(f"duplicate JSON key: {key}")
            value[key] = item
        return value
    try:
        rows = [
            json.loads(
                line,
                object_pairs_hook=unique_object,
                parse_constant=lambda token: (_ for _ in ()).throw(
                    ValueError(f"non-finite JSON constant: {token}")
                ),
            )
            for line in raw.splitlines()
            if line.strip()
        ]
    except (json.JSONDecodeError, TypeError, ValueError) as error:
        raise CodexCLITransportError("Codex output is not strict JSONL") from error
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise CodexCLITransportError("Codex JSONL contains a non-object or is empty")
    return rows


def codex_exec_argv(executable: str, model: str) -> list[str]:
    if model != GPT_MODEL:
        raise CodexCLITransportError("Codex model is not the frozen GPT model")
    return [
        executable, "exec", "--model", model, "--ephemeral",
        "--skip-git-repo-check", "--sandbox", "read-only",
        "--ignore-user-config", "--ignore-rules", "--color", "never",
        "--json", "-",
    ]


def parse_codex_exec_jsonl(raw: str) -> str:
    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        value: dict[str, object] = {}
        for key, item in pairs:
            if key in value:
                raise CodexCLITransportError(f"duplicate JSON key: {key}")
            value[key] = item
        return value

    try:
        events = [
            json.loads(
                line,
                object_pairs_hook=unique_object,
                parse_constant=lambda token: (_ for _ in ()).throw(
                    ValueError(f"non-finite JSON constant: {token}")
                ),
            )
            for line in raw.splitlines()
            if line.strip()
        ]
    except (json.JSONDecodeError, TypeError, ValueError) as error:
        raise CodexCLITransportError("Codex output is not strict JSONL") from error
    if not events or any(not isinstance(event, dict) for event in events):
        raise CodexCLITransportError("Codex event stream is empty or invalid")

    thread_starts = 0
    turn_starts = 0
    turn_completions = 0
    messages: list[str] = []
    for event in events:
        event_type = event.get("type")
        if event_type == "thread.started":
            thread_starts += 1
        elif event_type == "turn.started":
            turn_starts += 1
        elif event_type == "turn.completed":
            turn_completions += 1
        elif event_type == "error":
            raise CodexCLITransportError("Codex emitted an error event")
        elif event_type in {"item.started", "item.completed"}:
            item = event.get("item")
            if not isinstance(item, dict):
                raise CodexCLITransportError("Codex item is invalid")
            item_type = item.get("type")
            if item_type == "reasoning":
                continue
            if item_type != "agent_message":
                raise CodexCLITransportError("Codex tool or unknown item is forbidden")
            if event_type == "item.completed":
                text = item.get("text")
                if type(text) is not str or not text.strip():
                    raise CodexCLITransportError("Codex final message is empty")
                messages.append(text)
        else:
            raise CodexCLITransportError("Codex emitted an unknown event type")
    if (thread_starts, turn_starts, turn_completions, len(messages)) != (1, 1, 1, 1):
        raise CodexCLITransportError("Codex event family is incomplete or duplicated")
    return messages[0]


def probe_codex_model_identity(
    executable: str,
    *,
    cwd: Path,
    model: str = GPT_MODEL,
    timeout_seconds: float = 120.0,
) -> CodexModelIdentity:
    if model != GPT_MODEL:
        raise CodexCLITransportError("Codex model is not the frozen GPT model")
    candidate = shutil.which(executable)
    if candidate is None:
        raise CodexCLITransportError("Codex executable is unavailable")
    candidate_path = Path(candidate)
    if candidate_path.is_symlink():
        raise CodexCLITransportError("Codex executable is not a real regular file")
    resolved = candidate_path.resolve(strict=True)
    if not resolved.is_file() or resolved.is_symlink():
        raise CodexCLITransportError("Codex executable is not a real regular file")

    def run(
        argv: list[str], *, input_text: str | None = None
    ) -> subprocess.CompletedProcess[str]:
        try:
            completed = subprocess.run(
                argv,
                input=input_text,
                shell=False,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as error:
            raise CodexCLITransportError(f"Codex preflight failed: {error}") from error
        if completed.returncode != 0:
            raise CodexCLITransportError(
                f"Codex preflight exited with code {completed.returncode}"
            )
        return completed

    version_call = run([str(resolved), "--version"])
    cli_version = version_call.stdout.strip()
    if not cli_version:
        raise CodexCLITransportError("Codex version is empty")
    login_call = run([str(resolved), "login", "status"])
    login_status = "Logged in using ChatGPT"
    login_lines = [
        line
        for stream in (login_call.stdout, login_call.stderr)
        for line in stream.splitlines()
        if line
    ]
    allowed_login_lines = {login_status, CODEX_PATH_ALIAS_READ_ONLY_WARNING}
    if (
        login_lines.count(login_status) != 1
        or login_lines.count(CODEX_PATH_ALIAS_READ_ONLY_WARNING) > 1
        or any(line not in allowed_login_lines for line in login_lines)
    ):
        raise CodexCLITransportError("Codex is not logged in using ChatGPT")

    initialize_request = {
        "id": 1,
        "method": "initialize",
        "params": {
            "clientInfo": {"name": "protrek-feature-atlas", "version": "1"},
            "capabilities": {"experimentalApi": True},
        },
    }
    initialized_notification = {"method": "initialized", "params": {}}
    model_list_request = {
        "id": 2,
        "method": "model/list",
        "params": {"includeHidden": True, "limit": 1000},
    }
    process: subprocess.Popen[bytes] | None = None
    app_selector: selectors.BaseSelector | None = None
    stdout_buffer = b""
    stderr_chunks: list[bytes] = []

    def send_app_row(row: dict[str, object]) -> None:
        if process is None or process.stdin is None:
            raise CodexCLITransportError("Codex app-server stdin is unavailable")
        try:
            process.stdin.write((json.dumps(row, sort_keys=True) + "\n").encode())
            process.stdin.flush()
        except (BrokenPipeError, OSError) as error:
            raise CodexCLITransportError("Codex app-server stdin failed") from error

    def read_app_row() -> dict[str, object]:
        nonlocal stdout_buffer
        if process is None or process.stdout is None or app_selector is None:
            raise CodexCLITransportError("Codex app-server stdout is unavailable")
        deadline = time.monotonic() + timeout_seconds
        while True:
            while b"\n" in stdout_buffer:
                line, stdout_buffer = stdout_buffer.split(b"\n", 1)
                if line.strip():
                    break
            else:
                line = None
            if line is not None:
                try:
                    rows = strict_jsonl_objects(line.decode("utf-8"))
                except UnicodeDecodeError as error:
                    raise CodexCLITransportError(
                        "Codex output is not strict JSONL"
                    ) from error
                if len(rows) != 1:
                    raise CodexCLITransportError("Codex app-server row is invalid")
                return rows[0]

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise CodexCLITransportError("Codex app-server response timed out")
            events = app_selector.select(remaining)
            if not events:
                raise CodexCLITransportError("Codex app-server response timed out")
            for key, _ in events:
                try:
                    chunk = os.read(key.fd, 65536)
                except OSError as error:
                    raise CodexCLITransportError(
                        "Codex app-server output failed"
                    ) from error
                if chunk:
                    if key.data == "stdout":
                        stdout_buffer += chunk
                    else:
                        stderr_chunks.append(chunk)
                else:
                    app_selector.unregister(key.fileobj)
                    if key.data == "stdout":
                        if stdout_buffer.strip():
                            try:
                                rows = strict_jsonl_objects(
                                    stdout_buffer.decode("utf-8")
                                )
                            except UnicodeDecodeError as error:
                                raise CodexCLITransportError(
                                    "Codex output is not strict JSONL"
                                ) from error
                            stdout_buffer = b""
                            if len(rows) != 1:
                                raise CodexCLITransportError(
                                    "Codex app-server row is invalid"
                                )
                            return rows[0]
                        raise CodexCLITransportError(
                            "Codex app-server closed before completing the handshake"
                        )

    def close_app_input_and_collect() -> None:
        nonlocal stdout_buffer
        if process is None or app_selector is None:
            raise CodexCLITransportError("Codex app-server is unavailable")
        if process.stdin is not None:
            process.stdin.close()
            process.stdin = None
        deadline = time.monotonic() + timeout_seconds
        while app_selector.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise CodexCLITransportError("Codex app-server shutdown timed out")
            events = app_selector.select(remaining)
            if not events:
                raise CodexCLITransportError("Codex app-server shutdown timed out")
            for key, _ in events:
                chunk = os.read(key.fd, 65536)
                if chunk:
                    if key.data == "stdout":
                        stdout_buffer += chunk
                    else:
                        stderr_chunks.append(chunk)
                else:
                    app_selector.unregister(key.fileobj)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise CodexCLITransportError("Codex app-server shutdown timed out")
        try:
            returncode = process.wait(timeout=remaining)
        except subprocess.TimeoutExpired as error:
            raise CodexCLITransportError("Codex app-server shutdown timed out") from error
        if returncode != 0:
            raise CodexCLITransportError(
                f"Codex preflight exited with code {returncode}"
            )
        if stdout_buffer.strip():
            try:
                strict_jsonl_objects(stdout_buffer.decode("utf-8"))
            except UnicodeDecodeError as error:
                raise CodexCLITransportError("Codex output is not strict JSONL") from error
            raise CodexCLITransportError("Codex app-server emitted an extra response row")

    try:
        try:
            process = subprocess.Popen(
                [
                    str(resolved),
                    "app-server",
                    "-c",
                    'model_provider="openai"',
                    "--listen",
                    "stdio://",
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=False,
                bufsize=0,
            )
        except (OSError, subprocess.SubprocessError) as error:
            raise CodexCLITransportError(f"Codex preflight failed: {error}") from error
        if process.stdout is None or process.stderr is None:
            raise CodexCLITransportError("Codex app-server pipes are unavailable")
        app_selector = selectors.DefaultSelector()
        app_selector.register(process.stdout, selectors.EVENT_READ, "stdout")
        app_selector.register(process.stderr, selectors.EVENT_READ, "stderr")

        send_app_row(initialize_request)
        initialize_row = read_app_row()
        if (
            set(initialize_row) != {"id", "result"}
            or type(initialize_row.get("id")) is not int
            or initialize_row.get("id") != 1
            or not isinstance(initialize_row.get("result"), dict)
        ):
            raise CodexCLITransportError("Codex initialize response is invalid")
        initialized = initialize_row["result"]

        send_app_row(initialized_notification)
        send_app_row(model_list_request)
        notification_seen = False
        while True:
            response_row = read_app_row()
            if "method" in response_row:
                params = response_row.get("params")
                if (
                    notification_seen
                    or set(response_row) != {"method", "params"}
                    or response_row.get("method")
                    != "remoteControl/status/changed"
                    or not isinstance(params, dict)
                    or set(params)
                    != {
                        "status", "serverName", "installationId", "environmentId"
                    }
                    or params.get("status") != "disabled"
                    or type(params.get("serverName")) is not str
                    or not params["serverName"]
                    or type(params.get("installationId")) is not str
                    or not params["installationId"]
                    or params.get("environmentId") is not None
                ):
                    raise CodexCLITransportError(
                        "Codex app-server notification is invalid"
                    )
                notification_seen = True
                continue
            if (
                set(response_row) != {"id", "result"}
                or type(response_row.get("id")) is not int
                or response_row.get("id") != 2
                or not isinstance(response_row.get("result"), dict)
            ):
                raise CodexCLITransportError("Codex model/list response is invalid")
            catalog_result = response_row["result"]
            break
        close_app_input_and_collect()
    finally:
        if app_selector is not None:
            app_selector.close()
        if process is not None:
            if process.stdin is not None:
                process.stdin.close()
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=timeout_seconds)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=timeout_seconds)
            if process.stdout is not None:
                process.stdout.close()
            if process.stderr is not None:
                process.stderr.close()

    if not isinstance(initialized, dict) or not isinstance(catalog_result, dict):
        raise CodexCLITransportError("Codex app-server handshake is incomplete")
    if (
        set(catalog_result) != {"data", "nextCursor"}
        or not isinstance(catalog_result.get("data"), list)
        or catalog_result.get("nextCursor") is not None
    ):
        raise CodexCLITransportError("Codex model catalog result is invalid")
    catalog = catalog_result["data"]
    if any(not isinstance(row, dict) for row in catalog):
        raise CodexCLITransportError("Codex model catalog contains a non-object row")
    candidates = [
        row
        for row in catalog
        if row.get("id") == model or row.get("model") == model
    ]
    if len(candidates) != 1:
        raise CodexCLITransportError("Codex target model catalog row is not unique")
    target_model = candidates[0]
    if (
        not {"id", "model", "upgrade", "hidden"}.issubset(target_model)
        or type(target_model["id"]) is not str
        or target_model["id"] != model
        or type(target_model["model"]) is not str
        or target_model["model"] != model
        or target_model["upgrade"] is not None
        or type(target_model["hidden"]) is not bool
        or target_model["hidden"] is not False
    ):
        raise CodexCLITransportError("Codex target model catalog row is invalid")

    sentinel = "FEATURE_ATLAS_GPT_PREFLIGHT_OK_71f0c6"
    prompt = "Return exactly this text and use no tools: " + sentinel
    exec_call = run(codex_exec_argv(str(resolved), model), input_text=prompt)
    if parse_codex_exec_jsonl(exec_call.stdout) != sentinel:
        raise CodexCLITransportError("Codex sentinel response is invalid")
    reported_version = initialized.get("userAgent")
    if type(reported_version) is not str or not reported_version:
        raise CodexCLITransportError("Codex app-server version is missing")
    return CodexModelIdentity(
        cli_version=cli_version,
        model=model,
        model_provider="openai",
        reported_version=reported_version,
        resolved_executable=str(resolved),
    )
