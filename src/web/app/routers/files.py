"""The filesystem view: tree, raw file access and the archive download.

Every refusal in here answers to §12.4, and the tests in tests/test_artifacts.py
exercise the cases it names. The security reasoning lives next to the code that
enforces it, in services/artifacts.py.
"""

import io
import os
import zipfile
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from fastapi.responses import StreamingResponse

from app.security import CurrentUser, optional_user
from app.services import artifacts
from app.services import runs as runs_service
from app.services.authz import can_read_run
from app.settings import settings

router = APIRouter(prefix="/api/runs", tags=["files"])


async def _authorised_run(task_id: UUID, user: Optional[CurrentUser]) -> dict:
    row = await runs_service.get(task_id)
    if row is None or not can_read_run(user, row["submitted_by"]):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Run not found")
    return row


def _translate(exc: Exception) -> HTTPException:
    """Map a refusal to a status code without echoing any path back."""
    if isinstance(exc, artifacts.ArtifactRejected):
        return HTTPException(status.HTTP_400_BAD_REQUEST, "Rejected path")
    if isinstance(exc, artifacts.ArtifactNotFound):
        return HTTPException(status.HTTP_404_NOT_FOUND, "Not found")
    if isinstance(exc, artifacts.ArtifactTooLarge):
        return HTTPException(
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, "File is above the preview limit"
        )
    return HTTPException(status.HTTP_400_BAD_REQUEST, "Rejected path")


@router.get("/{task_id}/files")
async def file_tree(
    task_id: UUID, user: Optional[CurrentUser] = Depends(optional_user)
) -> dict:
    run = await _authorised_run(task_id, user)
    try:
        base = artifacts.run_root(task_id)
    except artifacts.ArtifactNotFound:
        # Not an error: §12.3 asks for a specific empty state per artifact
        # status, and "the run has not produced files yet" is one of them.
        return {
            "task_id": str(task_id),
            "entries": [],
            "file_count": 0,
            "total_bytes": 0,
            "artifact_status": run.get("artifact_status"),
            "available": False,
        }
    except artifacts.ArtifactError as exc:
        raise _translate(exc)

    entries = list(artifacts.walk(base))
    files = [e for e in entries if not e.is_dir]
    return {
        "task_id": str(task_id),
        "entries": [
            {
                "path": e.path,
                "name": e.name,
                "is_dir": e.is_dir,
                "size": e.size,
                "modified": e.modified,
                "preview": e.preview,
            }
            for e in entries
        ],
        "file_count": len(files),
        "total_bytes": sum(e.size for e in files),
        "artifact_status": run.get("artifact_status"),
        "available": True,
        "preview_limit_bytes": settings.preview_limit_bytes,
    }


@router.get("/{task_id}/files/raw")
async def raw_file(
    task_id: UUID,
    path: str = Query(..., min_length=1),
    user: Optional[CurrentUser] = Depends(optional_user),
) -> Response:
    await _authorised_run(task_id, user)
    try:
        target = artifacts.resolve(task_id, path)
        if target.is_dir():
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "Path is a directory")
        payload = artifacts.read_preview(target)
    except artifacts.ArtifactError as exc:
        raise _translate(exc)

    content_type, inline = artifacts.content_disposition(target)
    return Response(
        content=payload,
        media_type=content_type,
        headers=artifacts.hardening_headers(target.name, inline),
    )


@router.get("/{task_id}/files/meta")
async def file_meta(
    task_id: UUID,
    path: str = Query(..., min_length=1),
    user: Optional[CurrentUser] = Depends(optional_user),
) -> dict:
    await _authorised_run(task_id, user)
    try:
        target = artifacts.resolve(task_id, path)
        fd, st = artifacts.open_regular_file(target)
        os.close(fd)
    except artifacts.ArtifactError as exc:
        raise _translate(exc)

    content_type, inline = artifacts.content_disposition(target)
    return {
        "path": path,
        "name": target.name,
        "size": st.st_size,
        "modified": st.st_mtime,
        "content_type": content_type,
        "inline": inline,
        "preview": artifacts.preview_kind(target),
        "too_large": st.st_size > settings.preview_limit_bytes,
        "preview_limit_bytes": settings.preview_limit_bytes,
    }


@router.get("/{task_id}/archive.zip")
async def archive(
    task_id: UUID, user: Optional[CurrentUser] = Depends(optional_user)
) -> StreamingResponse:
    await _authorised_run(task_id, user)
    try:
        base = artifacts.run_root(task_id)
    except artifacts.ArtifactError as exc:
        raise _translate(exc)

    # Every entry is re-checked on the way in. That the directory passed once
    # says nothing about the individual files inside it, and a symlink added
    # between the walk and the read would otherwise be packaged up and shipped.
    buffer = io.BytesIO()
    written = 0
    entries = 0
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as bundle:
        for entry in artifacts.walk(base):
            if entry.is_dir:
                continue
            if entries >= settings.archive_max_entries:
                break
            if written + entry.size > settings.archive_limit_bytes:
                break
            try:
                target = artifacts.resolve(task_id, entry.path)
                fd, st = artifacts.open_regular_file(target)
            except artifacts.ArtifactError:
                continue
            with os.fdopen(fd, "rb") as handle:
                bundle.writestr(entry.path, handle.read())
            written += st.st_size
            entries += 1

    buffer.seek(0)
    return StreamingResponse(
        buffer,
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="run-{task_id}.zip"',
            "X-Content-Type-Options": "nosniff",
            "Cache-Control": "private, no-store",
        },
    )
