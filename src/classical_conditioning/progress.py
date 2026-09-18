"""User-visible stage and item progress for long-running analysis.

Review note: all output here is sent to stderr, preserving stdout for paths or
structured command results that callers may capture.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TypeVar

# Generic type retained by iter_items() so it yields the same item type it gets.
T = TypeVar("T")

# tqdm is deliberately optional: the package remains usable before optional UI
# dependencies are installed, with numbered text output as the fallback.
_TQDM = None
try:
    from tqdm import tqdm as _TQDM
except ImportError:  # pragma: no cover - optional until environment sync
    _TQDM = None


@dataclass
class PipelineProgress:
    """Print stage banners and per-item progress to stderr."""

    enabled: bool = True
    stream: object | None = None

    def __post_init__(self) -> None:
        # Use stderr by default, but permit tests and embedding applications to
        # provide their own file-like stream.
        if self.stream is None:
            self.stream = sys.stderr

    def _write(self, message: str) -> None:
        # A disabled reporter must be a no-op rather than requiring callers to
        # wrap every reporting call in their own enabled check.
        if not self.enabled:
            return
        stream = self.stream
        assert stream is not None
        print(message, file=stream, flush=True)

    def stage(self, title: str, *, detail: str | None = None) -> None:
        """Announce a major pipeline stage."""
        # One stable banner format makes long terminal logs easy to scan.
        line = f"\n==> {title}"
        if detail:
            line = f"{line} -- {detail}"
        self._write(line)

    def info(self, message: str) -> None:
        # Indentation visually attaches an informational line to its stage.
        self._write(f"    {message}")

    def item_done(self, index: int, total: int, label: str, *, status: str) -> None:
        # Include a numerator/denominator so resumed batch logs remain auditable.
        self._write(f"    [{index}/{total}] {label}: {status}")

    def step(self, label: str, *, status: str) -> None:
        """Flag a sub-step inside the current stage (for example one fish stage)."""
        self._write(f"      > {label}: {status}")

    @contextmanager
    def step_timer(self, label: str):
        """Announce a sub-step, then report elapsed time when it finishes."""
        # Avoid timing/reporting work entirely in quiet mode while retaining the
        # same context-manager control flow for the caller.
        if not self.enabled:
            yield
            return
        self.step(label, status="running")
        started = time.perf_counter()
        # Report both success and failure duration, then preserve the original
        # exception and traceback for normal error handling.
        try:
            yield
        except Exception:
            elapsed = time.perf_counter() - started
            self.step(label, status=f"failed after {elapsed:.1f}s")
            raise
        else:
            elapsed = time.perf_counter() - started
            self.step(label, status=f"done in {elapsed:.1f}s")

    @contextmanager
    def stage_timer(self, title: str, *, detail: str | None = None):
        """Announce a stage and report elapsed seconds when it finishes."""
        # Unlike step_timer, stage timing reports a final duration even if the
        # body raises, which is useful evidence when diagnosing failures.
        self.stage(title, detail=detail)
        started = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - started
            self.info(f"finished in {elapsed:.1f}s")

    def iter_items(
        self,
        items: Iterable[T],
        *,
        description: str,
        total: int | None = None,
    ) -> Iterator[T]:
        """Iterate with a tqdm bar when available, else numbered messages."""
        # Materialising allows a known fallback total even when items was a
        # generator; this helper is intended for finite per-recording batches.
        sequence = list(items)
        count = total if total is not None else len(sequence)
        # Quiet mode preserves data flow but makes no terminal/UI calls.
        if not self.enabled:
            yield from sequence
            return
        # Prefer the interactive progress bar when installed; otherwise emit a
        # deterministic text line before each item.
        if _TQDM is not None:
            yield from _TQDM(
                sequence,
                total=count,
                desc=description,
                file=self.stream,
                dynamic_ncols=True,
                leave=True,
            )
            return
        for index, item in enumerate(sequence, start=1):
            self.info(f"{description} [{index}/{count}] {item}")
            yield item


def default_progress(*, enabled: bool = True) -> PipelineProgress:
    # Central factory keeps future default presentation changes in one place.
    return PipelineProgress(enabled=enabled)
