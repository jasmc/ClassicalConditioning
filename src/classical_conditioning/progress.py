"""User-visible stage and item progress for long-running analysis."""

from __future__ import annotations

import sys
import time
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TypeVar

T = TypeVar("T")

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
        if self.stream is None:
            self.stream = sys.stderr

    def _write(self, message: str) -> None:
        if not self.enabled:
            return
        stream = self.stream
        assert stream is not None
        print(message, file=stream, flush=True)

    def stage(self, title: str, *, detail: str | None = None) -> None:
        """Announce a major pipeline stage."""
        line = f"\n==> {title}"
        if detail:
            line = f"{line} -- {detail}"
        self._write(line)

    def info(self, message: str) -> None:
        self._write(f"    {message}")

    def item_done(self, index: int, total: int, label: str, *, status: str) -> None:
        self._write(f"    [{index}/{total}] {label}: {status}")

    def step(self, label: str, *, status: str) -> None:
        """Flag a sub-step inside the current stage (for example one fish stage)."""
        self._write(f"      > {label}: {status}")

    @contextmanager
    def step_timer(self, label: str):
        """Announce a sub-step, then report elapsed time when it finishes."""
        if not self.enabled:
            yield
            return
        self.step(label, status="running")
        started = time.perf_counter()
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
        sequence = list(items)
        count = total if total is not None else len(sequence)
        if not self.enabled:
            yield from sequence
            return
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
    return PipelineProgress(enabled=enabled)
