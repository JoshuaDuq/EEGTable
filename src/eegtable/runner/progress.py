"""Progress reporting for a run: plain text for people, JSON lines for programs.

The JSON events follow the protocol of the EEG_fMRI_Pipeline terminal UI, one
object per line: ``start``, ``subject_start``, ``progress``, ``log``,
``subject_done``, ``complete`` and ``error``. A recording plays the part of a
subject, so a front end written for that protocol can follow a run unchanged.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Protocol, TextIO

CHECK = "✓"
CROSS = "✗"
_INDENT = "      "


class Reporter(Protocol):
    """Receives a run's progress."""

    def start(self, labels: Sequence[str], output_root: Path) -> None:
        """The run is about to process these recordings."""

    def recording_start(self, label: str, position: int, total: int) -> None:
        """A recording is starting."""

    def step(self, label: str, step: str, current: int, total: int) -> None:
        """A recording has reached a step."""

    def recording_done(self, label: str, success: bool, message: str) -> None:
        """A recording has finished, with a one-line summary or error."""

    def complete(self, success: bool, seconds: float, message: str, outputs: Sequence[str]) -> None:
        """The run has finished."""


class NullReporter:
    """Reports nothing."""

    def start(self, labels: Sequence[str], output_root: Path) -> None:
        del labels, output_root

    def recording_start(self, label: str, position: int, total: int) -> None:
        del label, position, total

    def step(self, label: str, step: str, current: int, total: int) -> None:
        del label, step, current, total

    def recording_done(self, label: str, success: bool, message: str) -> None:
        del label, success, message

    def complete(self, success: bool, seconds: float, message: str, outputs: Sequence[str]) -> None:
        del success, seconds, message, outputs


class _Pace:
    """How far a run has got, and how long the rest should take at that rate."""

    def __init__(self, clock: Callable[[], float]) -> None:
        self.clock = clock
        self.started = clock()
        self.total = 0
        self.done = 0

    def begin(self, total: int) -> None:
        self.started, self.total, self.done = self.clock(), total, 0

    def finish_one(self) -> tuple[float, float | None]:
        """Seconds elapsed, and seconds left, or None when nothing is."""
        self.done += 1
        elapsed = self.clock() - self.started
        left = self.total - self.done
        # Wall time per finished recording already reflects how many run at once.
        return elapsed, (elapsed / self.done * left if left > 0 else None)


class TextReporter:
    """One line per recording, for a person watching a terminal."""

    def __init__(
        self, stream: TextIO | None = None, *, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self.stream = sys.stdout if stream is None else stream
        self._pace = _Pace(clock)
        self._any_started = False
        self._last_started: str | None = None

    def start(self, labels: Sequence[str], output_root: Path) -> None:
        self._pace.begin(len(labels))
        noun = "recording" if len(labels) == 1 else "recordings"
        self._write(f"eegtable · {len(labels)} {noun} → {output_root}")

    def recording_start(self, label: str, position: int, total: int) -> None:
        self._write(f"[{position}/{total}] {label}")
        self._any_started = True
        self._last_started = label

    def step(self, label: str, step: str, current: int, total: int) -> None:
        del label, step, current, total

    def recording_done(self, label: str, success: bool, message: str) -> None:
        # Recordings computed in parallel finish out of turn; name those.
        out_of_turn = self._any_started and label != self._last_started
        named = f"{label} · " if out_of_turn else ""
        self._last_started = None
        elapsed, left = self._pace.finish_one()
        pace = (
            ""
            if left is None
            else f"  ({self._pace.done}/{self._pace.total} done · {human_duration(elapsed)} elapsed"
            f" · about {human_duration(left)} left)"
        )
        self._write(f"{_INDENT}{CHECK if success else CROSS} {named}{message}{pace}")

    def complete(self, success: bool, seconds: float, message: str, outputs: Sequence[str]) -> None:
        del success, seconds, outputs
        self._write(message)

    def _write(self, line: str) -> None:
        print(line, file=self.stream, flush=True)


class JsonReporter:
    """One JSON object per line, for a program following the run.

    ``subject_done`` also carries ``elapsed`` and ``eta`` in seconds (``eta`` is null
    once nothing is left), which a front end written for the protocol can ignore.
    """

    def __init__(
        self, stream: TextIO | None = None, *, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self.stream = sys.stdout if stream is None else stream
        self._pace = _Pace(clock)

    def start(self, labels: Sequence[str], output_root: Path) -> None:
        self._pace.begin(len(labels))
        self.emit(
            event="start",
            operation="eegtable run",
            subjects=list(labels),
            total_subjects=len(labels),
            output_root=str(output_root),
        )

    def recording_start(self, label: str, position: int, total: int) -> None:
        del position, total
        self.emit(event="subject_start", subject=label)

    def step(self, label: str, step: str, current: int, total: int) -> None:
        self.emit(
            event="progress",
            subject=label,
            step=step,
            current=current,
            total=total,
            pct=round(100 * current / total) if total else 0,
        )

    def recording_done(self, label: str, success: bool, message: str) -> None:
        self.emit(event="log", level="info" if success else "error", message=message, subject=label)
        elapsed, left = self._pace.finish_one()
        self.emit(
            event="subject_done",
            subject=label,
            success=success,
            elapsed=round(elapsed, 1),
            eta=None if left is None else round(left, 1),
        )

    def complete(self, success: bool, seconds: float, message: str, outputs: Sequence[str]) -> None:
        self.emit(event="log", level="info" if success else "warning", message=message)
        self.emit(
            event="complete", success=success, duration=round(seconds, 3), outputs=list(outputs)
        )

    def error(self, code: str, message: str) -> None:
        """A problem that stopped the run before or instead of processing."""
        self.emit(event="error", code=code, message=message)

    def emit(self, **event: Any) -> None:
        """Write one event."""
        print(json.dumps(event), file=self.stream, flush=True)


# cmd.exe and PowerShell take double quotes; POSIX quoting would wrap every Windows path,
# backslashes and all, in single quotes that cmd.exe keeps as part of the name.
_WINDOWS = os.name == "nt"


def shell_command(args: Sequence[str]) -> str:
    """``args`` as one command line that can be pasted into this platform's shell."""
    return subprocess.list2cmdline(args) if _WINDOWS else shlex.join(args)


def human_duration(seconds: float) -> str:
    """A duration as a person would say it: 4.3 s, 45 s, 12 min, 2 h 05 min."""
    if seconds < 10:
        return f"{seconds:.1f} s"
    if seconds < 90:
        return f"{seconds:.0f} s"
    minutes = round(seconds / 60)
    if minutes < 90:
        return f"{minutes} min"
    return f"{minutes // 60} h {minutes % 60:02d} min"
