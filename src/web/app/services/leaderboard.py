"""Leaderboard aggregation.

The ranking must not collapse into a single "Score" column: the exchange rate
between a gradient evaluation and a sample evaluation is undecided, so the
aggregate is one selectable, self-describing column beside the dimensions it
is derived from, and none of the offered formulas mixes gradients with
samples.

Aggregation is median with quartiles, never the best run: evolutionary methods
are stochastic, and ranking by best result rewards luck.
"""

from uuid import UUID

from app import db
from app.security import CurrentUser
from app.services import series as series_service
from app.services.authz import can_read_run

SCORE_FORMULAS = {
    "loss_v1": {
        "id": "loss_v1",
        "label": "median final loss",
        "direction": "asc",
        "column": "final_loss",
        "note": (
            "Lower is better. No budget normalization: gradient evaluations and processed samples "
            "have no common currency."
        ),
    },
    "accuracy_v1": {
        "id": "accuracy_v1",
        "label": "median final accuracy",
        "direction": "desc",
        "column": "final_accuracy",
        "note": "Higher is better. Also without budget normalization.",
    },
}
DEFAULT_FORMULA = "loss_v1"

_BASE_SQL = """
SELECT
    t.optimizer_name                                              AS optimizer,
    t.family::text                                                AS family,
    t.dataset                                                     AS dataset,
    t.model_name                                                  AS model,
    t.suite::text                                                 AS suite,
    COUNT(*)                                                      AS n_runs,
    COUNT(DISTINCT t.seed)                                        AS n_seeds,
    percentile_cont(0.5)  WITHIN GROUP (ORDER BY r.final_loss)    AS loss_median,
    percentile_cont(0.25) WITHIN GROUP (ORDER BY r.final_loss)    AS loss_q1,
    percentile_cont(0.75) WITHIN GROUP (ORDER BY r.final_loss)    AS loss_q3,
    MIN(r.final_loss)                                             AS loss_min,
    percentile_cont(0.5)  WITHIN GROUP (ORDER BY r.final_accuracy) AS acc_median,
    percentile_cont(0.25) WITHIN GROUP (ORDER BY r.final_accuracy) AS acc_q1,
    percentile_cont(0.75) WITHIN GROUP (ORDER BY r.final_accuracy) AS acc_q3,
    percentile_cont(0.5)  WITHIN GROUP (ORDER BY r.gradient_count)   AS grad_median,
    percentile_cont(0.5)  WITHIN GROUP (ORDER BY r.database_reaches) AS reach_median,
    MODE() WITHIN GROUP (ORDER BY r.stop_reason::text)            AS stop_reason_mode,
    ARRAY_AGG(ARRAY[t.task_id::text, t.submitted_by::text]
              ORDER BY r.final_loss)                              AS task_owners
FROM tasks t
JOIN results r ON r.task_id = t.task_id
WHERE t.task_status = 'COMPLETED'
  AND t.optimizer_name IS NOT NULL
"""


async def query(
    dataset: str | None = None,
    model: str | None = None,
    family: str | None = None,
    suite: str | None = None,
    score: str = DEFAULT_FORMULA,
    limit: int = 200,
    user: CurrentUser | None = None,
) -> dict:
    formula = SCORE_FORMULAS.get(score) or SCORE_FORMULAS[DEFAULT_FORMULA]

    sql = _BASE_SQL
    params: list = []
    if dataset:
        sql += " AND t.dataset = %s"
        params.append(dataset)
    if model:
        sql += " AND t.model_name = %s"
        params.append(model)
    if family:
        sql += " AND t.family::text = %s"
        params.append(family)
    if suite:
        sql += " AND t.suite::text = %s"
        params.append(suite)

    sql += """
    GROUP BY t.optimizer_name, t.family, t.dataset, t.model_name, t.suite
    """
    rows = await db.fetch_all(sql, params)

    def score_of(row: dict) -> float | None:
        return row["loss_median"] if formula["column"] == "final_loss" else row["acc_median"]

    ranked = [r for r in rows if score_of(r) is not None]
    ranked.sort(key=score_of, reverse=(formula["direction"] == "desc"))
    ranked = ranked[:limit]

    def readable_ids(row: dict) -> list[str]:
        # The aggregate counts every run; only the identifiers the requester
        # can open are handed out, so the ranking is not a directory of
        # other people's private runs.
        return [task_id for task_id, owner in row["task_owners"] if can_read_run(user, UUID(owner))]

    readable = [readable_ids(row) for row in ranked]

    return {
        "score_formula": formula,
        "available_formulas": list(SCORE_FORMULAS.values()),
        "rows": [
            {
                "rank": index + 1,
                "optimizer": row["optimizer"],
                "family": row["family"],
                "dataset": row["dataset"],
                "model": row["model"],
                "suite": row["suite"],
                # n_runs travels with every row on purpose: a median over one
                # run is an anecdote, and the interface has to be able to say so.
                "n_runs": row["n_runs"],
                "n_seeds": row["n_seeds"],
                "final_loss": {
                    "median": row["loss_median"],
                    "q1": row["loss_q1"],
                    "q3": row["loss_q3"],
                    "min": row["loss_min"],
                },
                "final_accuracy": {
                    "median": row["acc_median"],
                    "q1": row["acc_q1"],
                    "q3": row["acc_q3"],
                },
                "gradient_count": {"median": row["grad_median"]},
                "database_reaches": {"median": row["reach_median"]},
                "stop_reason_mode": row["stop_reason_mode"],
                "task_ids": readable[index][:32],
                # True when at least one run behind the row opens for this
                # requester; always false for an anonymous visitor.
                "can_view": bool(readable[index]),
                "score": score_of(row),
            }
            for index, row in enumerate(ranked)
        ],
    }


async def filter_options() -> dict:
    rows = await db.fetch_all(
        """
        SELECT DISTINCT t.dataset, t.model_name, t.family::text AS family,
                        t.suite::text AS suite
          FROM tasks t JOIN results r ON r.task_id = t.task_id
        """
    )
    return {
        "datasets": sorted({r["dataset"] for r in rows if r["dataset"]}),
        "models": sorted({r["model_name"] for r in rows if r["model_name"]}),
        "families": sorted({r["family"] for r in rows if r["family"]}),
        "suites": sorted({r["suite"] for r in rows if r["suite"]}),
    }


# A public curve is a median over runs. Over one or two runs that "median" is
# somebody's private run drawn on a public page, so smaller groups are listed
# with their size but get no curve.
MIN_PUBLIC_RUNS = 3

_CURVE_SQL = """
SELECT t.dataset, t.model_name, t.suite::text AS suite,
       t.optimizer_name, t.family::text AS family,
       s.loss, s.accuracy, s.epochs,
       s.gradient_count, s.database_reaches
  FROM tasks t
  JOIN results r ON r.task_id = t.task_id
  JOIN result_series s ON s.task_id = t.task_id
 WHERE t.task_status = 'COMPLETED'
   AND t.optimizer_name IS NOT NULL
   AND t.dataset IS NOT NULL
   AND t.model_name IS NOT NULL
"""


async def public_curves(
    dataset: str | None = None,
    model: str | None = None,
    suite: str | None = None,
    x: str = "database_reaches",
    metric: str = "loss",
    points: int = 160,
) -> dict:
    """Convergence curves per optimizer for one problem, open to everyone.

    Only the aggregate leaves the server: median and interquartile band per
    optimizer, with the run count. No run identifier, owner, seed or single
    run's series is part of the response, so the curves add information to the
    public ranking without exposing the runs behind it.
    """
    rows = await db.fetch_all(_CURVE_SQL)

    cells: dict[tuple[str, str, str], dict[str, list[dict]]] = {}
    for row in rows:
        key = (row["dataset"], row["model_name"], row["suite"])
        cells.setdefault(key, {}).setdefault(row["optimizer_name"], []).append(row)

    def eligible(groups: dict[str, list[dict]]) -> int:
        return sum(1 for runs in groups.values() if len(runs) >= MIN_PUBLIC_RUNS)

    listing = sorted(
        (
            {
                "dataset": d,
                "model": m,
                "suite": s,
                "n_optimizers": eligible(groups),
                "n_runs": sum(len(r) for r in groups.values()),
            }
            for (d, m, s), groups in cells.items()
            if eligible(groups) > 0
        ),
        key=lambda c: (c["dataset"], c["model"], c["suite"]),
    )

    selected = None
    if dataset and model:
        wanted = (dataset, model, suite or "test")
        if wanted in cells and eligible(cells[wanted]) > 0:
            selected = wanted
    if selected is None and listing:
        # Default to the problem with the most methods side by side.
        best = max(listing, key=lambda c: (c["n_optimizers"], c["n_runs"]))
        selected = (best["dataset"], best["model"], best["suite"])

    series: list[dict] = []
    withheld: list[dict] = []
    if selected is not None:
        for name, group in sorted(cells[selected].items()):
            if len(group) < MIN_PUBLIC_RUNS:
                withheld.append({"label": name, "n_runs": len(group)})
                continue
            pairs = [series_service.series_points(r, x, metric) for r in group]
            aggregated = series_service.aggregate(
                label=name,
                runs=pairs,
                family=group[0]["family"],
                points=points,
            )
            if aggregated.n_runs >= MIN_PUBLIC_RUNS:
                series.append(aggregated.to_dict())
            else:
                withheld.append({"label": name, "n_runs": len(group)})

    return {
        "cells": listing,
        "selected": ({"dataset": selected[0], "model": selected[1], "suite": selected[2]} if selected else None),
        "x": x,
        "metric": metric,
        "min_runs": MIN_PUBLIC_RUNS,
        "series": series,
        "withheld": withheld,
    }
