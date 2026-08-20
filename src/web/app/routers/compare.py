"""Overlaid convergence curves with median and interquartile band.

The statistics live in services/series.py; this router selects runs, groups
them, and exports the aggregate. Export uses the same aggregation the chart
does, so a figure and the CSV behind it cannot disagree.
"""

import csv
import io
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import Response

from app import db
from app.security import CurrentUser, optional_user
from app.services import runs as runs_service
from app.services import series as series_service
from app.services.authz import can_read_run

router = APIRouter(prefix="/api/compare", tags=["compare"])

GROUPINGS = {
    "optimizer": lambda r: r["optimizer_name"] or "?",
    "optimizer_dataset": lambda r: f"{r['optimizer_name']} · {r['dataset']}",
    "optimizer_model": lambda r: f"{r['optimizer_name']} · {r['model_name']}",
    "run": lambda r: (r["run_name"] or str(r["task_id"])[:8]),
}


def _parse_ids(raw: str) -> list[UUID]:
    ids: list[UUID] = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            ids.append(UUID(chunk))
        except ValueError:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "Malformed run identifier")
    if not ids:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "No runs selected")
    if len(ids) > 60:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Too many runs selected")
    return ids


async def _collect(
    ids: list[UUID], user: Optional[CurrentUser], x: str, metric: str, group_by: str
) -> tuple[dict, list[dict]]:
    rows = await db.fetch_all(
        f"""
        SELECT {runs_service.RUN_COLUMNS},
               s.epochs, s.loss, s.accuracy,
               s.gradient_count  AS series_gradient_count,
               s.database_reaches AS series_database_reaches
          FROM tasks t
          JOIN users u ON u.id = t.submitted_by
          LEFT JOIN results r ON r.task_id = t.task_id
          LEFT JOIN result_series s ON s.task_id = t.task_id
         WHERE t.task_id = ANY(%s)
        """,
        (ids,),
    )
    visible = [r for r in rows if can_read_run(user, r["submitted_by"])]

    grouped: dict[str, list] = {}
    families: dict[str, Optional[str]] = {}
    key_of = GROUPINGS.get(group_by, GROUPINGS["optimizer"])

    for row in visible:
        series_row = {
            "epochs": row.get("epochs"),
            "loss": row.get("loss"),
            "accuracy": row.get("accuracy"),
            "gradient_count": row.get("series_gradient_count"),
            "database_reaches": row.get("series_database_reaches"),
        }
        pairs = series_service.series_points(series_row, x, metric)
        if not pairs:
            continue
        key = key_of(row)
        grouped.setdefault(key, []).append(pairs)
        families.setdefault(key, row.get("family"))

    return {"grouped": grouped, "families": families}, visible


@router.get("")
async def compare(
    runs: str = Query(...),
    x: str = Query("gradient_count"),
    metric: str = Query("loss"),
    group_by: str = Query("optimizer"),
    points: int = Query(200, ge=20, le=1000),
    logx: bool = Query(False),
    user: Optional[CurrentUser] = Depends(optional_user),
) -> dict:
    if x not in series_service.X_AXES:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Unknown x axis")
    if metric not in series_service.METRICS:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Unknown metric")

    ids = _parse_ids(runs)
    bundle, visible = await _collect(ids, user, x, metric, group_by)

    series = [
        series_service.aggregate(
            label=key,
            runs=group,
            family=bundle["families"].get(key),
            points=points,
            logarithmic=logx,
        ).to_dict()
        for key, group in sorted(bundle["grouped"].items())
    ]

    # Pairwise differences of the final median, with n on both sides.
    #
    # No p-values. Which test and how many repetitions is an open team decision,
    # and a test without correction for multiple comparisons is exactly the error
    # this audience notices first. The schema records a seed per run, so the
    # analysis is possible the moment the decision is made.
    finals = []
    for entry in series:
        values = [v for v in entry["median"] if v is not None]
        finals.append((entry["label"], values[-1] if values else None, entry["n_runs"]))

    differences = []
    for i, (label_a, value_a, n_a) in enumerate(finals):
        for label_b, value_b, n_b in finals[i + 1:]:
            if value_a is None or value_b is None:
                continue
            differences.append({
                "a": label_a, "b": label_b,
                "median_a": value_a, "median_b": value_b,
                "delta": value_a - value_b,
                "relative": ((value_a - value_b) / value_b) if value_b else None,
                "n_a": n_a, "n_b": n_b,
            })

    return {
        "x": x,
        "metric": metric,
        "group_by": group_by,
        "series": series,
        "differences": differences,
        "runs": [runs_service.serialise(r) for r in visible],
        "missing": len(ids) - len(visible),
        "statistical_test": {
            "available": False,
            "note": "Test istotności nie jest liczony: liczba powtórzeń i wybór "
                    "testu (D3) pozostają nierozstrzygnięte, a test bez poprawki "
                    "na wielokrotne porównania byłby błędem. Schemat zapisuje "
                    "ziarno per run, więc analiza jest możliwa po decyzji.",
        },
    }


@router.get("/export.csv")
async def export_csv(
    runs: str = Query(...),
    x: str = Query("gradient_count"),
    metric: str = Query("loss"),
    group_by: str = Query("optimizer"),
    points: int = Query(200, ge=20, le=1000),
    logx: bool = Query(False),
    user: Optional[CurrentUser] = Depends(optional_user),
) -> Response:
    payload = await compare(runs, x, metric, group_by, points, logx, user)

    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow([
        "series", "family", "n_runs", x, f"{metric}_median",
        f"{metric}_q1", f"{metric}_q3", "n_at_x", "within_full_band",
    ])
    for entry in payload["series"]:
        for index, x_value in enumerate(entry["x"]):
            writer.writerow([
                entry["label"], entry["family"] or "", entry["n_runs"], x_value,
                entry["median"][index], entry["q1"][index], entry["q3"][index],
                entry["n_at_x"][index],
                "yes" if index <= entry["full_until_index"] else "no",
            ])

    return Response(
        content=buffer.getvalue(),
        media_type="text/csv; charset=utf-8",
        headers={
            "Content-Disposition": 'attachment; filename="compare.csv"',
            "X-Content-Type-Options": "nosniff",
        },
    )
