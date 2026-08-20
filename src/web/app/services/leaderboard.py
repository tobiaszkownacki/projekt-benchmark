"""Leaderboard aggregation.

§13.1 is explicit that the ranking must not collapse into a single "Score"
column: D2 -- the exchange rate between a gradient evaluation and a sample
evaluation -- is open, and it decides who wins. Anything hard-coded here would
be a scientific claim the team has expressly delegated to people at the
university.

So the aggregate is one selectable, self-describing column beside the dimensions
it is derived from, and the formulas on offer are only those that do not presume
an answer to D2. None of them mixes gradients with samples.

Aggregation is median with quartiles, never the best run: evolutionary methods
are stochastic, and ranking by best result rewards luck.
"""

from typing import Optional

from app import db

SCORE_FORMULAS = {
    "loss_v1": {
        "id": "loss_v1",
        "label": "mediana straty końcowej",
        "direction": "asc",
        "column": "final_loss",
        "note": "Niższa jest lepsza. Nie normalizuje po budżecie — D2 "
                "(wspólna waluta gradientów i próbek) jest nierozstrzygnięta.",
    },
    "accuracy_v1": {
        "id": "accuracy_v1",
        "label": "mediana dokładności końcowej",
        "direction": "desc",
        "column": "final_accuracy",
        "note": "Wyższa jest lepsza. Również bez normalizacji po budżecie.",
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
    ARRAY_AGG(t.task_id::text ORDER BY r.final_loss)              AS task_ids
FROM tasks t
JOIN results r ON r.task_id = t.task_id
WHERE t.task_status = 'completed'
  AND t.optimizer_name IS NOT NULL
"""


async def query(
    dataset: Optional[str] = None,
    model: Optional[str] = None,
    family: Optional[str] = None,
    suite: Optional[str] = None,
    score: str = DEFAULT_FORMULA,
    limit: int = 200,
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

    def score_of(row: dict) -> Optional[float]:
        return row["loss_median"] if formula["column"] == "final_loss" else row["acc_median"]

    ranked = [r for r in rows if score_of(r) is not None]
    ranked.sort(key=score_of, reverse=(formula["direction"] == "desc"))
    ranked = ranked[:limit]

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
                "task_ids": row["task_ids"][:32],
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
