"""Control plane and static host for the single-page frontend.

One process serves both the API and the built frontend. That is the reason for
choosing a build-time bundler over a server-rendered framework: Node is needed
to build the image and not to run it, so the deployment stays the four
containers §5.1 describes rather than gaining a fifth runtime to patch. It also
removes cross-origin cookies and CORS from the picture entirely.
"""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from app import db
from app.migrations import apply_migrations
from app.routers import (
    admin,
    auth,
    compare,
    events,
    files,
    leaderboard,
    meta,
    runs,
    submissions,
)
from app.settings import settings

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
)
logger = logging.getLogger("web")


@asynccontextmanager
async def lifespan(_: FastAPI):
    if settings.run_migrations:
        applied = await apply_migrations()
        if applied:
            logger.info("Applied migrations: %s", ", ".join(applied))
    await db.open_pool()
    await db.broker.start()
    logger.info("Control plane ready")
    try:
        yield
    finally:
        await db.broker.stop()
        await db.close_pool()


app = FastAPI(
    title="Benchmark Czarnej Skrzynki",
    description="Control plane for the black-box optimizer benchmark",
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/api/openapi",
    redoc_url=None,
    openapi_url="/api/openapi.json",
)

for module in (
    auth, runs, files, leaderboard, compare, submissions, events, admin, meta,
):
    app.include_router(module.router)


@app.get("/healthz")
async def healthz() -> dict:
    try:
        await db.fetch_one("SELECT 1 AS ok")
        database = "up"
    except Exception as exc:
        database = f"down: {type(exc).__name__}"
    return {"status": "ok", "database": database}


@app.exception_handler(HTTPException)
async def http_exception_handler(_: Request, exc: HTTPException) -> JSONResponse:
    detail = exc.detail
    body = detail if isinstance(detail, dict) else {"message": detail}
    return JSONResponse(status_code=exc.status_code, content=body, headers=exc.headers)


_SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "same-origin",
    "X-Frame-Options": "DENY",
}


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    for header, value in _SECURITY_HEADERS.items():
        response.headers.setdefault(header, value)
    if not request.url.path.startswith("/api/runs"):
        # Artifact responses set their own, stricter policy; everything else
        # gets a policy with no external origins, which is also why no font or
        # script is loaded from a CDN anywhere in the frontend.
        response.headers.setdefault(
            "Content-Security-Policy",
            "default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; "
            "script-src 'self'; connect-src 'self'; font-src 'self'; "
            "object-src 'none'; base-uri 'none'; frame-ancestors 'none'",
        )
    return response


_INDEX = settings.static_root / "index.html"

if settings.static_root.is_dir():
    app.mount(
        "/assets",
        StaticFiles(directory=settings.static_root / "assets", check_dir=False),
        name="assets",
    )


@app.get("/{full_path:path}")
async def spa(full_path: str) -> Response:
    """Serve the frontend, and hand every unmatched path to its router.

    Without this, opening /runs/<id>/files directly -- or reloading it, or
    following it from an email -- returns 404, because only the SPA knows that
    route. §11.2 makes every resource having its own shareable URL a
    requirement, so this is load-bearing rather than convenience, and it is
    covered by a test.
    """
    if full_path.startswith("api/"):
        raise HTTPException(404, "Not found")

    candidate = (settings.static_root / full_path).resolve()
    if (
        full_path
        and settings.static_root.is_dir()
        and candidate.is_file()
        and candidate.is_relative_to(settings.static_root.resolve())
    ):
        return FileResponse(candidate)

    if _INDEX.is_file():
        return FileResponse(_INDEX)
    return JSONResponse(
        status_code=503,
        content={"message": "Frontend nie jest zbudowany. Uruchom `npm run build`."},
    )
