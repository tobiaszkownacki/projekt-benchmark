"""Safe access to a single run's artifact directory.

This module is the only barrier between a logged-in participant's browser and
the datasets, which §5.3 names as the project's most valuable asset: anyone
holding them can run the benchmark privately and the competition stops meaning
anything. The threat model is therefore not an anonymous scanner but a
legitimate, verified entrant with a valid task_id.

Two rules shape everything below:

1.  Textual checks are not the authority; ``realpath`` is. Rejecting ".." by
    string comparison misses URL encoding, overlapping prefixes and symlinks, so
    the string checks here are only a cheap early exit and the containment
    decision is always made on the resolved path.
2.  The check and the open must not be separable. Between resolving a path and
    reading it, the downloader can write into the same directory -- and ``scp -r``
    happily carries symlinks over from the cluster, so a hostile link can appear
    with no attacker present on this side. Opening with O_NOFOLLOW and stat-ing
    the descriptor rather than the path closes that window.
"""

import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional
from uuid import UUID

from app.settings import settings

MAX_RELATIVE_LENGTH = 1024

# Inline is granted by extension, never by sniffing the bytes. Serving a user's
# file as text/html is cross-site scripting on our own origin, and SVG counts as
# HTML for this purpose because it can carry script.
_INLINE_TYPES = {
    ".png": "image/png",
}
_TEXT_TYPES = {
    ".csv": "text/csv; charset=utf-8",
    ".json": "application/json",
    ".py": "text/plain; charset=utf-8",
    ".out": "text/plain; charset=utf-8",
    ".log": "text/plain; charset=utf-8",
    ".txt": "text/plain; charset=utf-8",
    ".md": "text/plain; charset=utf-8",
    ".err": "text/plain; charset=utf-8",
}
_FALLBACK_TYPE = "application/octet-stream"

# Which preview the browser should build for a file. The server never sends
# markup; it sends data and a hint, and React renders it as text nodes.
_PREVIEW_KIND = {
    ".png": "image",
    ".csv": "table",
    ".json": "json",
    ".py": "code",
    ".out": "log",
    ".log": "log",
    ".err": "log",
    ".txt": "text",
    ".md": "text",
}


class ArtifactError(Exception):
    """Base class for refusals. Messages never echo a filesystem path."""


class ArtifactRejected(ArtifactError):
    """The request is malformed or tries to leave the run directory."""


class ArtifactNotFound(ArtifactError):
    """The run directory or the requested entry does not exist."""


class ArtifactTooLarge(ArtifactError):
    """Above the preview ceiling; downloadable but not rendered."""


@dataclass(frozen=True)
class FileEntry:
    path: str
    name: str
    is_dir: bool
    size: int
    modified: float
    preview: Optional[str]


def artifact_root() -> Path:
    """The resolved root. Resolving it matters: if /downloads were itself a
    symlink, comparing an unresolved prefix against a resolved child would
    always disagree."""
    return settings.artifact_root.resolve()


def run_root(task_id: UUID | str, root: Optional[Path] = None) -> Path:
    """Directory for one run, validated to sit directly under the root.

    task_id is parsed as a UUID before it ever reaches the filesystem. That
    parse is itself a control: a UUID cannot contain "/" or "..", so a whole
    class of traversal never gets a chance to be interpreted as a path.
    """
    base = (root or artifact_root()).resolve()
    if not isinstance(task_id, UUID):
        try:
            task_id = UUID(str(task_id))
        except (ValueError, AttributeError, TypeError):
            raise ArtifactRejected("Malformed run identifier")

    candidate = base / str(task_id)
    try:
        resolved = candidate.resolve(strict=True)
    except (FileNotFoundError, RuntimeError):
        raise ArtifactNotFound("No artifacts for this run")
    if not resolved.is_relative_to(base):
        raise ArtifactRejected("Run directory escapes the artifact root")
    if not resolved.is_dir():
        raise ArtifactNotFound("No artifacts for this run")
    return resolved


def resolve(task_id: UUID | str, relative: str, root: Optional[Path] = None) -> Path:
    """Resolve a path relative to a run directory, or refuse."""
    if relative is None:
        raise ArtifactRejected("Missing path")
    if "\0" in relative:
        raise ArtifactRejected("Path contains a null byte")
    if len(relative) > MAX_RELATIVE_LENGTH:
        raise ArtifactRejected("Path is too long")
    if relative.startswith("/") or relative.startswith("\\"):
        raise ArtifactRejected("Absolute paths are not accepted")
    if os.path.isabs(relative) or (len(relative) > 1 and relative[1] == ":"):
        raise ArtifactRejected("Absolute paths are not accepted")

    base = run_root(task_id, root)
    relative = relative.strip("/")
    if not relative:
        return base

    try:
        target = (base / relative).resolve(strict=True)
    except (FileNotFoundError, RuntimeError):
        raise ArtifactNotFound("No such file in this run")

    # is_relative_to, not startswith: "/downloads/<uuid>-other" starts with
    # "/downloads/<uuid>" as a string but is a different directory.
    if not target.is_relative_to(base):
        raise ArtifactRejected("Path escapes the run directory")
    return target


def open_regular_file(path: Path) -> tuple[int, os.stat_result]:
    """Open without following a final symlink, and confirm on the descriptor.

    Checking the descriptor rather than the path is the point: the answer cannot
    then be invalidated by anything that happens to the name afterwards.
    """
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except OSError as exc:
        # ELOOP is what O_NOFOLLOW raises for a symlink.
        raise ArtifactRejected("Refusing to open a link or special file") from exc
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            raise ArtifactRejected("Not a regular file")
    except Exception:
        os.close(fd)
        raise
    return fd, st


def read_preview(path: Path, limit: Optional[int] = None) -> bytes:
    limit = settings.preview_limit_bytes if limit is None else limit
    fd, st = open_regular_file(path)
    try:
        if st.st_size > limit:
            raise ArtifactTooLarge("File is above the preview limit")
        with os.fdopen(fd, "rb") as handle:
            return handle.read(limit + 1)
    except ArtifactTooLarge:
        os.close(fd)
        raise
    except Exception:
        try:
            os.close(fd)
        except OSError:
            pass
        raise


def walk(base: Path) -> Iterator[FileEntry]:
    """Depth-first listing of a run directory.

    followlinks is off and every entry is checked with lstat, so a symlink is
    reported by nobody -- not the tree, not the archive, not the preview.
    """
    for dirpath, dirnames, filenames in os.walk(base, followlinks=False):
        current = Path(dirpath)
        dirnames[:] = sorted(
            d for d in dirnames if not (current / d).is_symlink()
        )
        for name in dirnames:
            entry = current / name
            try:
                st = entry.lstat()
            except OSError:
                continue
            yield FileEntry(
                path=str(entry.relative_to(base)),
                name=name,
                is_dir=True,
                size=0,
                modified=st.st_mtime,
                preview=None,
            )
        for name in sorted(filenames):
            entry = current / name
            try:
                st = entry.lstat()
            except OSError:
                continue
            if not stat.S_ISREG(st.st_mode):
                continue
            yield FileEntry(
                path=str(entry.relative_to(base)),
                name=name,
                is_dir=False,
                size=st.st_size,
                modified=st.st_mtime,
                preview=preview_kind(entry),
            )


def preview_kind(path: Path) -> Optional[str]:
    return _PREVIEW_KIND.get(path.suffix.lower())


def content_disposition(path: Path) -> tuple[str, bool]:
    """Return (content_type, inline). Anything unrecognised is a download."""
    suffix = path.suffix.lower()
    if suffix in _INLINE_TYPES:
        return _INLINE_TYPES[suffix], True
    if suffix in _TEXT_TYPES:
        return _TEXT_TYPES[suffix], False
    return _FALLBACK_TYPE, False


def hardening_headers(filename: str, inline: bool) -> dict[str, str]:
    """Headers applied to every raw artifact response without exception.

    CSP sandbox is the belt to nosniff's braces: it makes the browser treat the
    response as an opaque origin, so even a mistake in the whitelist above
    cannot reach a session cookie on the main domain.
    """
    safe = filename.replace('"', "").replace("\r", "").replace("\n", "")
    disposition = "inline" if inline else "attachment"
    return {
        "X-Content-Type-Options": "nosniff",
        "Content-Security-Policy": "sandbox",
        "Cache-Control": "private, no-store",
        "Content-Disposition": f'{disposition}; filename="{safe}"',
    }


def directory_summary(base: Path) -> tuple[int, int]:
    files = 0
    total = 0
    for entry in walk(base):
        if not entry.is_dir:
            files += 1
            total += entry.size
    return files, total
