"""Where things are on disk. The only module that knows the store's folder names."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Layout:
    root: Path
    """The store folder, e.g. workspace/store. Its parent is the workspace."""

    @property
    def workspace(self) -> Path:
        """Media paths are stored relative to this folder, so store and media move together."""
        return self.root.parent

    @property
    def runs(self) -> Path:
        return self.root / "runs"

    @property
    def tmp(self) -> Path:
        """Runs being written. On the same disk as runs/, so the final rename is one step."""
        return self.root / "tmp"

    def run_dir(self, run_id: str) -> Path:
        return self.runs / run_id

    def create(self) -> None:
        for folder in (self.runs, self.tmp):
            folder.mkdir(parents=True, exist_ok=True)

    def relative_media_path(self, path: Path) -> str:
        """Path of a media file relative to the workspace; refuses files outside it."""
        resolved = path.resolve()
        try:
            return resolved.relative_to(self.workspace).as_posix()
        except ValueError:
            raise ValueError(
                f"{resolved} is outside the workspace {self.workspace}. "
                "Put media under the workspace (e.g. workspace/media/) so the store can find it."
            ) from None
