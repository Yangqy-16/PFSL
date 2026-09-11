"""tmux-backed task launcher with deterministic GPU selection."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Sequence


RUNS_ROOT = Path(__file__).resolve().parent / "runs"
_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


@dataclass(frozen=True)
class GPUStat:
    index: int
    free_mb: int
    utilization: int


def _query_gpus() -> list[GPUStat]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        raise RuntimeError("Unable to query GPUs with nvidia-smi") from error

    stats: list[GPUStat] = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 3:
            raise RuntimeError(f"Unexpected nvidia-smi row: {line!r}")
        try:
            stats.append(
                GPUStat(
                    index=int(fields[0]),
                    free_mb=int(fields[1]),
                    utilization=int(fields[2]),
                )
            )
        except ValueError as error:
            raise RuntimeError(f"Unexpected nvidia-smi row: {line!r}") from error
    return stats


def select_freest_gpu() -> int:
    """Select max free memory, then min utilization, then min index."""

    stats = _query_gpus()
    if not stats:
        raise RuntimeError("No GPUs were reported by nvidia-smi")
    return max(
        stats,
        key=lambda item: (item.free_mb, -item.utilization, -item.index),
    ).index


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _save_status(path: Path, payload: object) -> None:
    from esmc_sae.artifacts import save_json

    save_json(path, payload)


def _validate_name(name: str) -> str:
    if not _NAME_PATTERN.fullmatch(name):
        raise ValueError(
            "name must contain only letters, digits, '.', '_' and '-'"
        )
    return name


def _tmux_session_is_live(name: str) -> bool:
    result = subprocess.run(
        ["tmux", "has-session", "-t", f"={name}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if result.returncode not in (0, 1):
        raise RuntimeError("tmux could not inspect existing sessions")
    return result.returncode == 0


def _normalize_command(command: Sequence[str]) -> list[str]:
    normalized = list(command)
    if not normalized or normalized[0] != "--":
        raise ValueError("command vector requires an explicit '--' separator")
    normalized.pop(0)
    if not normalized:
        raise ValueError("launch requires a command after '--'")
    return normalized


def _write_command_script(
    path: Path,
    *,
    status_path: Path,
    stdout_path: Path,
    session: str,
    mode: str,
    gpu: int | None,
    eta_minutes: int,
    signal_channel: str | None,
    command: Sequence[str],
) -> None:
    environment = (
        f"export CUDA_VISIBLE_DEVICES={shlex.quote(str(gpu))}"
        if gpu is not None
        else "export CUDA_VISIBLE_DEVICES="
    )
    runner = [
        sys.executable,
        "-m",
        "esmc_sae.runtime",
        "_run",
        "--status-path",
        str(status_path),
        "--stdout-path",
        str(stdout_path),
        "--session",
        session,
        "--mode",
        mode,
        "--eta-minutes",
        str(eta_minutes),
    ]
    if gpu is not None:
        runner.extend(["--gpu", str(gpu)])
    if signal_channel is not None:
        runner.extend(["--signal-channel", signal_channel])
    runner.extend(["--", *command])
    contents = "\n".join(
        [
            "#!/usr/bin/env bash",
            "set -eu",
            environment,
            (
                "export MPLCONFIGDIR="
                "${MPLCONFIGDIR:-/tmp/matplotlib-esmc-sae}"
            ),
            f"exec {shlex.join(runner)}",
            "",
        ]
    )
    path.write_text(contents, encoding="utf-8")
    path.chmod(0o755)


def _initial_status(
    *,
    session: str,
    mode: str,
    gpu: int | None,
    eta_minutes: int,
    command: Sequence[str],
) -> dict[str, object]:
    started = datetime.now(timezone.utc)
    return {
        "session": session,
        "child_pid": None,
        "mode": mode,
        "gpu": gpu,
        "eta_minutes": eta_minutes,
        "started_at": started.isoformat(),
        "estimated_end_at": (
            started + timedelta(minutes=eta_minutes)
        ).isoformat(),
        "ended_at": None,
        "exit_code": None,
        "command": list(command),
    }


def _launch(args: argparse.Namespace) -> int:
    name = _validate_name(args.name)
    if args.eta_minutes <= 0:
        raise ValueError("eta-minutes must be a positive integer")
    command = _normalize_command(args.command)
    if _tmux_session_is_live(name):
        raise RuntimeError(f"tmux session is already live: {name}")

    gpu = select_freest_gpu() if args.mode == "gpu" else None
    run_dir = RUNS_ROOT / name
    if run_dir.exists():
        raise FileExistsError(f"run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    command_path = run_dir / "command.sh"
    stdout_path = run_dir / "stdout.log"
    status_path = run_dir / "status.json"
    stdout_path.touch()
    signal_channel = f"esmc_sae_done_{name}" if args.wait else None
    _save_status(
        status_path,
        _initial_status(
            session=name,
            mode=args.mode,
            gpu=gpu,
            eta_minutes=args.eta_minutes,
            command=command,
        ),
    )
    _write_command_script(
        command_path,
        status_path=status_path,
        stdout_path=stdout_path,
        session=name,
        mode=args.mode,
        gpu=gpu,
        eta_minutes=args.eta_minutes,
        signal_channel=signal_channel,
        command=command,
    )

    try:
        result = subprocess.run(
            [
                "tmux",
                "new-session",
                "-d",
                "-P",
                "-F",
                "#{pane_pid}",
                "-s",
                name,
                str(command_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as error:
        with status_path.open("r", encoding="utf-8") as handle:
            failed_status = json.load(handle)
        failed_status["ended_at"] = _timestamp()
        failed_status["exit_code"] = 255
        _save_status(status_path, failed_status)
        detail = (error.stderr or error.stdout or str(error)).strip()
        raise RuntimeError(f"tmux failed to start session: {detail}") from error
    pane_pid = int(result.stdout.strip())
    gpu_text = str(gpu) if gpu is not None else "cpu"
    print(
        f"session={name} pid={pane_pid} gpu={gpu_text} "
        f"eta_minutes={args.eta_minutes}",
        flush=True,
    )

    if not args.wait:
        return 0
    subprocess.run(["tmux", "wait-for", signal_channel], check=True)
    with status_path.open("r", encoding="utf-8") as handle:
        status = json.load(handle)
    exit_code = status.get("exit_code")
    if type(exit_code) is not int:
        raise RuntimeError("completed task did not record an exit code")
    return exit_code


def _run_child(args: argparse.Namespace) -> int:
    status_path = Path(args.status_path)
    stdout_path = Path(args.stdout_path)
    try:
        command = _normalize_command(args.command)
        started = datetime.now(timezone.utc)
        environment = os.environ.copy()
        if args.mode == "gpu":
            environment["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        else:
            environment["CUDA_VISIBLE_DEVICES"] = ""

        try:
            with stdout_path.open("ab", buffering=0) as output:
                child = subprocess.Popen(
                    command,
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    env=environment,
                    cwd=str(Path(__file__).resolve().parents[1]),
                )
                status = {
                    "session": args.session,
                    "child_pid": child.pid,
                    "mode": args.mode,
                    "gpu": args.gpu,
                    "eta_minutes": args.eta_minutes,
                    "started_at": started.isoformat(),
                    "estimated_end_at": (
                        started + timedelta(minutes=args.eta_minutes)
                    ).isoformat(),
                    "ended_at": None,
                    "exit_code": None,
                    "command": command,
                }
                _save_status(status_path, status)
                exit_code = child.wait()
        except BaseException:
            ended = _timestamp()
            try:
                with status_path.open("r", encoding="utf-8") as handle:
                    failed_status = json.load(handle)
                failed_status["ended_at"] = ended
                failed_status["exit_code"] = 255
                _save_status(status_path, failed_status)
            except BaseException as status_error:
                raise RuntimeError(
                    "Unable to persist failed child status"
                ) from status_error
            raise

        status["ended_at"] = _timestamp()
        status["exit_code"] = exit_code
        try:
            _save_status(status_path, status)
        except BaseException as status_error:
            raise RuntimeError(
                "Unable to persist final status"
            ) from status_error
        return exit_code
    finally:
        if args.signal_channel:
            subprocess.run(
                ["tmux", "wait-for", "-S", args.signal_channel],
                check=False,
            )


def _show_status(args: argparse.Namespace) -> int:
    name = _validate_name(args.name)
    status_path = RUNS_ROOT / name / "status.json"
    if not status_path.is_file():
        raise FileNotFoundError(f"status does not exist: {name}")
    with status_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    launch = subparsers.add_parser("launch")
    launch.add_argument("--name", required=True)
    launch.add_argument("--mode", required=True, choices=("cpu", "gpu"))
    launch.add_argument("--eta-minutes", required=True, type=int)
    launch.add_argument("--wait", action="store_true")
    launch.add_argument("command", nargs=argparse.REMAINDER)
    launch.set_defaults(handler=_launch)

    status = subparsers.add_parser("status")
    status.add_argument("--name", required=True)
    status.set_defaults(handler=_show_status)

    run = subparsers.add_parser("_run", help=argparse.SUPPRESS)
    run.add_argument("--status-path", required=True)
    run.add_argument("--stdout-path", required=True)
    run.add_argument("--session", required=True)
    run.add_argument("--mode", required=True, choices=("cpu", "gpu"))
    run.add_argument("--gpu", type=int)
    run.add_argument("--eta-minutes", required=True, type=int)
    run.add_argument("--signal-channel")
    run.add_argument("command", nargs=argparse.REMAINDER)
    run.set_defaults(handler=_run_child)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return args.handler(args)
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
