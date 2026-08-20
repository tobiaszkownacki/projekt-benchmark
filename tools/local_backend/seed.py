"""Populate a development database with genuinely computed runs.

Every completed run in here was produced by driving the project's own
ModelEvaluator with the project's own NumPy optimizers. The convergence curves,
the gradient counts and the sample counts are measurements, not plausible-looking
numbers -- which matters, because a chart drawn from invented data can be made to
look like anything, and the point of a demonstration is that it cannot.

Runs are also created in the states that have no data: waiting in the broker,
waiting in SLURM, running, downloading, failed with a log, failed with no
artifacts at all, and rejected by the validator. §11.3 lists those states
precisely because they are the ones that never get built, and they are exactly
where an interface is usually at its worst.
"""

import argparse
import json
import os
import shutil
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src" / "frontend"))

from tools.local_backend.artifacts import write_run_artifacts  # noqa: E402
from tools.local_backend.runner import (  # noqa: E402
    LOCAL_OPTIMIZERS,
    RUNNER_VERSION,
    LocalBenchmarkRunner,
    StopCondition,
)

# Model size is paired with optimizer family deliberately.
#
# CMA-ES maintains a covariance matrix over the parameter vector: O(n^2) to
# store and O(n^3) to decompose. On a 1603-parameter network one run takes
# minutes, and on a 54k-parameter one it is out of reach entirely. That is not a
# quirk of this backend -- it is the scaling ceiling §18 names, met in practice --
# so the population methods here sit on the smaller networks while the gradient
# methods cover the wider ones.
#
# Differential Evolution keeps no covariance structure, so it scales to the
# wider models and is used there instead.
MATRIX = [
    ("wine", "mlp-1x16", ["adam", "sgd", "cma-es", "des", "de"]),
    ("wine", "mlp-2x32", ["adam", "adamw", "lion", "rmsprop", "sgd_momentum"]),
    ("breast_cancer", "mlp-1x16", ["adam", "lion", "sgd_momentum", "des", "de"]),
    ("digits", "mlp-1x16", ["adam", "adamw", "rmsprop", "sgd_momentum", "de"]),
    ("digits", "mlp-3x64", ["adam", "adamw", "sgd", "rmsprop"]),
]
SEEDS = [11, 23, 42, 57, 71, 89, 101, 113]

SAMPLE_OPTIMIZER = '''import numpy as np

from benchmark.evaluator import ModelEvaluator
from benchmark.optimizer_protocols import NumpyBenchmarkOptimizer


class SignSgdOptimizer(NumpyBenchmarkOptimizer):
    """Sign-SGD: krok o stałej długości w kierunku znaku gradientu."""

    def __init__(self, initial_params, lr: float = 0.01, **config):
        super().__init__(initial_params, **config)
        self.params = np.asarray(initial_params, dtype=np.float64)
        self.lr = lr

    def step(self, evaluator: ModelEvaluator) -> bool:
        _loss, gradient = evaluator.evaluate_with_grad()
        self.params = self.params - self.lr * np.sign(np.asarray(gradient))
        evaluator.set_params(self.params)
        return False
'''


def connect(url: str) -> psycopg.Connection:
    return psycopg.connect(url, row_factory=dict_row, autocommit=False)


def ensure_users(conn) -> dict[str, uuid.UUID]:
    """Development accounts. Passwords come from the environment.

    No password is written into this file. A default baked into a seeding script
    is a default that reaches a server eventually.
    """
    from auth.passwords import hash_password

    people = [
        ("admin@benchmark.pw.edu.pl", "admin", "Administrator systemu",
         "Politechnika Warszawska", "SEED_ADMIN_PASSWORD"),
        ("badacz@benchmark.pw.edu.pl", "verified", "Badacz testowy",
         "Zakład Sztucznej Inteligencji EIT", "SEED_USER_PASSWORD"),
        ("gosc@benchmark.pw.edu.pl", "verified", "Uczestnik konkursu",
         "Uniwersytet Jagielloński", "SEED_USER_PASSWORD"),
        ("nowy@benchmark.pw.edu.pl", "unverified", "Konto oczekujące",
         "Politechnika Wrocławska", "SEED_USER_PASSWORD"),
    ]

    identifiers: dict[str, uuid.UUID] = {}
    with conn.cursor() as cur:
        for email, role, name, organisation, env_var in people:
            password = os.environ.get(env_var)
            if not password:
                raise SystemExit(
                    f"{env_var} is not set. Export it before seeding; this script "
                    "will not invent a password."
                )
            cur.execute(
                """
                INSERT INTO users (email, password_hash, role, auth_provider,
                                   display_name, associated_organisation,
                                   join_reason, last_login_at)
                VALUES (%s, %s, %s, 'email', %s, %s, %s, NOW())
                ON CONFLICT (email) DO UPDATE
                    SET role = EXCLUDED.role,
                        display_name = EXCLUDED.display_name,
                        password_hash = EXCLUDED.password_hash
                RETURNING id
                """,
                (email, hash_password(password), role, name, organisation,
                 "Konto testowe utworzone przez seed."),
            )
            identifiers[email] = cur.fetchone()["id"]
    conn.commit()
    return identifiers


def _submission(conn, user_id, name, kind, builtin, family, source, log, status):
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO submissions (submitted_by, display_name, kind, builtin_name,
                                     source_code, source_sha256, family, status,
                                     validator_log, validator_version, validated_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, 'seed-1', NOW())
            RETURNING submission_id
            """,
            (user_id, name, kind, builtin, source,
             __import__("hashlib").sha256(source.encode()).hexdigest() if source else None,
             family, status, log),
        )
        return cur.fetchone()["submission_id"]


def _insert_task(conn, **kwargs) -> uuid.UUID:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO tasks (
                queue_name, executor_name, submitted_by, dataset, run_name,
                optimizer_params, submission_id, seed, suite, model_name,
                optimizer_name, family, stop_condition, task_status,
                artifact_status, artifact_root, artifact_bytes, artifact_files,
                executor_task_id, error_message, runner_version, gpu_model,
                created_at, queued_at, started_at, completed_at, updated_at
            )
            VALUES (
                %(queue_name)s, %(executor_name)s, %(submitted_by)s, %(dataset)s,
                %(run_name)s, %(optimizer_params)s, %(submission_id)s, %(seed)s,
                %(suite)s, %(model_name)s, %(optimizer_name)s, %(family)s,
                %(stop_condition)s, %(task_status)s, %(artifact_status)s,
                %(artifact_root)s, %(artifact_bytes)s, %(artifact_files)s,
                %(executor_task_id)s, %(error_message)s, %(runner_version)s,
                %(gpu_model)s, %(created_at)s, %(queued_at)s, %(started_at)s,
                %(completed_at)s, %(updated_at)s
            )
            RETURNING task_id
            """,
            kwargs,
        )
        return cur.fetchone()["task_id"]


def _walk_states(conn, task_id, created, started, completed, failed=False) -> None:
    """Move a task through the states a real run passes through.

    The transitions table is filled by a trigger on tasks, so inserting a row
    already marked completed produces a history with exactly one entry. That
    both undersells the feature and leaves the update path of the trigger
    untested. Walking pending -> running -> completed writes the same history
    the worker and poller would, and proves the trigger fires on updates.
    """
    with conn.cursor() as cur:
        cur.execute(
            """UPDATE tasks SET task_status = 'running', started_at = %s,
                                updated_at = %s WHERE task_id = %s""",
            (started, started, task_id),
        )
        if failed:
            cur.execute(
                """UPDATE tasks SET task_status = 'failed', completed_at = %s,
                                    updated_at = %s WHERE task_id = %s""",
                (completed, completed, task_id),
            )
            return
        # The downloader has not finished yet: a real intermediate state.
        cur.execute(
            """UPDATE tasks SET task_status = 'completed', artifact_status = 'downloading',
                                completed_at = %s, updated_at = %s WHERE task_id = %s""",
            (completed, completed, task_id),
        )
        cur.execute(
            """UPDATE tasks SET artifact_status = 'ready', updated_at = %s
                WHERE task_id = %s""",
            (completed, task_id),
        )


def _store_result(conn, task_id, result) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO results (task_id, final_loss, final_accuracy, gradient_count,
                                 database_reaches, total_steps, total_epochs,
                                 wall_time_seconds, stop_reason)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (task_id) DO NOTHING
            """,
            (task_id, result.final_loss, result.final_accuracy, result.gradient_count,
             result.database_reaches, result.total_steps, result.total_epochs,
             result.wall_time_seconds, result.stop_reason),
        )
        cur.execute(
            """
            INSERT INTO result_series (task_id, epochs, loss, accuracy,
                                       gradient_count, database_reaches,
                                       wall_time_seconds)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (task_id) DO NOTHING
            """,
            (task_id, result.epoch_history, result.loss_history,
             result.accuracy_history, result.gradient_history,
             result.database_reaches_history, result.time_history),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database-url", default=os.environ.get("DATABASE_URL", ""))
    parser.add_argument("--downloads", default=os.environ.get("ARTIFACT_ROOT", "./downloads"))
    parser.add_argument("--seeds", type=int, default=8, help="seeds per configuration")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--reset", action="store_true", help="delete existing runs first")
    args = parser.parse_args()

    if not args.database_url:
        raise SystemExit("--database-url or DATABASE_URL is required")

    downloads = Path(args.downloads).resolve()
    downloads.mkdir(parents=True, exist_ok=True)

    conn = connect(args.database_url)
    users = ensure_users(conn)
    researcher = users["badacz@benchmark.pw.edu.pl"]
    guest = users["gosc@benchmark.pw.edu.pl"]

    if args.reset:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM tasks")
            cur.execute("DELETE FROM submissions")
        conn.commit()
        for child in downloads.iterdir():
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)

    seeds = SEEDS[: args.seeds]
    now = datetime.now(timezone.utc)
    completed = 0

    for dataset, model, optimizers in MATRIX:
        runner_cache: dict[int, LocalBenchmarkRunner] = {}
        for optimizer_key in optimizers:
            family = LOCAL_OPTIMIZERS[optimizer_key][2]
            owner = researcher if optimizers.index(optimizer_key) % 2 == 0 else guest
            submission_id = _submission(
                conn, owner, optimizer_key, "builtin", optimizer_key, family,
                None,
                f"Optymalizator wbudowany '{optimizer_key}' — kod pochodzi z "
                f"repozytorium, walidacja protokołu pominięta.",
                "accepted",
            )

            for seed in seeds:
                if seed not in runner_cache:
                    runner_cache[seed] = LocalBenchmarkRunner(
                        dataset_name=dataset, model_name=model,
                        stop_condition=StopCondition(max_epochs=args.epochs),
                        batch_size=32, seed=seed,
                    )
                result = runner_cache[seed].run(optimizer_key)
                created = now - timedelta(hours=len(seeds) * 3, minutes=seed)

                task_id = _insert_task(
                    conn,
                    queue_name="ATHENA_WORKER_QUEUE", executor_name="local-cpu",
                    submitted_by=owner, dataset=dataset,
                    run_name=f"{optimizer_key}-{dataset}-s{seed}",
                    optimizer_params=Jsonb({"optimizer": optimizer_key, "seed": seed}),
                    submission_id=submission_id, seed=seed, suite="test",
                    model_name=model, optimizer_name=optimizer_key, family=family,
                    stop_condition=Jsonb({"max_epochs": args.epochs}),
                    task_status="pending", artifact_status="absent",
                    artifact_root=None, artifact_bytes=0, artifact_files=0,
                    executor_task_id=str(4700000 + completed),
                    error_message=None, runner_version=RUNNER_VERSION,
                    gpu_model="CPU (local backend)",
                    created_at=created, queued_at=created,
                    started_at=created + timedelta(seconds=40),
                    completed_at=created + timedelta(seconds=40 + result.wall_time_seconds),
                    updated_at=created + timedelta(seconds=60),
                )

                root = downloads / str(task_id)
                files, total = write_run_artifacts(
                    root, result,
                    optimizer_source=None,
                    slurm_job_id=str(4700000 + completed),
                    extra_metadata={
                        "task_id": str(task_id), "runner_version": RUNNER_VERSION,
                        "backend": "local-cpu", "suite": "test",
                        "note": "Dane publiczne scikit-learn. To NIE są zbiory "
                                "konkursowe projektu.",
                    },
                )
                with conn.cursor() as cur:
                    cur.execute(
                        """UPDATE tasks SET artifact_root = %s, artifact_files = %s,
                                            artifact_bytes = %s WHERE task_id = %s""",
                        (str(root), files, total, task_id),
                    )
                _walk_states(
                    conn, task_id, created,
                    created + timedelta(seconds=40),
                    created + timedelta(seconds=40 + result.wall_time_seconds),
                )
                _store_result(conn, task_id, result)
                completed += 1
                print(f"  {dataset}/{model} {optimizer_key} seed={seed} "
                      f"loss={result.final_loss:.4f} grads={result.gradient_count} "
                      f"samples={result.database_reaches}")
            conn.commit()

    # A final-suite slice, so the interface can show that the competition run and
    # the practice run are different things (§13.1).
    final_runner = LocalBenchmarkRunner(
        "wine", "mlp-1x16", StopCondition(max_epochs=args.epochs), 32, 7
    )
    final_submission = _submission(
        conn, researcher, "cma-es", "builtin", "cma-es", "gradient_free", None,
        "Przebieg finałowy.", "accepted",
    )
    for optimizer_key in ("adam", "cma-es"):
        result = final_runner.run(optimizer_key)
        task_id = _insert_task(
            conn, queue_name="ATHENA_WORKER_QUEUE", executor_name="local-cpu",
            submitted_by=researcher, dataset="wine",
            run_name=f"final-{optimizer_key}-wine",
            optimizer_params=Jsonb({"optimizer": optimizer_key, "seed": 7}),
            submission_id=final_submission, seed=7, suite="final",
            model_name="mlp-1x16", optimizer_name=optimizer_key,
            family=LOCAL_OPTIMIZERS[optimizer_key][2],
            stop_condition=Jsonb({"max_epochs": args.epochs}),
            task_status="pending", artifact_status="absent", artifact_root=None,
            artifact_bytes=0, artifact_files=0, executor_task_id=str(4800001),
            error_message=None, runner_version=RUNNER_VERSION,
            gpu_model="CPU (local backend)", created_at=now - timedelta(hours=2),
            queued_at=now - timedelta(hours=2), started_at=now - timedelta(hours=2),
            completed_at=now - timedelta(hours=1), updated_at=now - timedelta(hours=1),
        )
        root = downloads / str(task_id)
        files, total = write_run_artifacts(
            root, result, slurm_job_id="4800001",
            extra_metadata={"task_id": str(task_id), "suite": "final"},
        )
        with conn.cursor() as cur:
            cur.execute(
                """UPDATE tasks SET artifact_root=%s, artifact_files=%s,
                                    artifact_bytes=%s WHERE task_id=%s""",
                (str(root), files, total, task_id),
            )
        _walk_states(conn, task_id, now - timedelta(hours=2),
                     now - timedelta(hours=2), now - timedelta(hours=1))
        _store_result(conn, task_id, result)
    conn.commit()

    seed_states(conn, downloads, researcher, guest, now)
    conn.close()
    print(f"\nSeeded {completed + 2} computed runs plus the non-terminal states.")


def seed_states(conn, downloads: Path, researcher, guest, now) -> None:
    """The states with no results: queued, running, failed, rejected."""
    uploaded = _submission(
        conn, guest, "sign-sgd", "uploaded", None, "gradient", SAMPLE_OPTIMIZER,
        "PASSED Module and class loaded successfully\n"
        "   └─ Found custom class: SignSgdOptimizer\n"
        "PASSED Protocol compliance check (Duck Typing)\n"
        "PASSED Method get_output_type() is implemented\n"
        "   └─ Returns DTO: NumpyNdarrayTensorEvaluatorDto\n"
        "PASSED Backend inference\n   └─ Inferred NumPy backend from DTO\n"
        "PASSED __init__ signature accepts 'initial_params'\n"
        "PASSED Hardware Check\n   └─ Using CPU for NumPy optimizer\n"
        "PASSED step() method returns a boolean value (bool)\n"
        "PASSED Integration with ModelEvaluator (metrics tracking)\n"
        "   └─ Database reaches: 4, Gradients calculated: 1\n"
        "PASSED Parameter mutation verification\n",
        "accepted",
    )
    rejected = _submission(
        conn, guest, "broken-optimizer", "uploaded", None, "gradient",
        "class Broken:\n    def step(self, evaluator):\n        return 0\n",
        "PASSED Module and class loaded successfully\n"
        "   └─ Found custom class: Broken\n"
        "ERROR Protocol compliance check (Duck Typing)\n"
        "   └─ Missing required methods: get_output_type\n"
        "ERROR __init__ signature accepts 'initial_params'\n"
        "   └─ The first positional argument after 'self' must be 'initial_params'\n"
        "ERROR step() method returns a boolean value (bool)\n"
        "   └─ Returned unexpected type: <class 'int'>\n",
        "rejected",
    )

    common = dict(
        queue_name="ATHENA_WORKER_QUEUE", executor_name="athena",
        optimizer_params=Jsonb({}), suite="test", model_name="mlp-3x64",
        stop_condition=Jsonb({"max_epochs": 12}), artifact_root=None,
        artifact_bytes=None, artifact_files=None, runner_version=None,
        gpu_model="NVIDIA A100-SXM4-40GB",
    )

    states = [
        dict(run_name="sign-sgd-digits-queued", task_status="pending",
             artifact_status="absent", executor_task_id=None, error_message=None,
             submitted_by=guest, submission_id=uploaded, seed=11,
             optimizer_name="sign-sgd", family="gradient", dataset="digits",
             minutes=3),
        dict(run_name="sign-sgd-digits-slurm", task_status="pending",
             artifact_status="absent", executor_task_id="4812004",
             error_message=None, submitted_by=guest, submission_id=uploaded,
             seed=23, optimizer_name="sign-sgd", family="gradient",
             dataset="digits", minutes=26),
        dict(run_name="sign-sgd-digits-running", task_status="running",
             artifact_status="absent", executor_task_id="4812005",
             error_message=None, submitted_by=guest, submission_id=uploaded,
             seed=42, optimizer_name="sign-sgd", family="gradient",
             dataset="digits", minutes=64),
        dict(run_name="cma-es-wine-downloading", task_status="completed",
             artifact_status="downloading", executor_task_id="4812006",
             error_message=None, submitted_by=researcher, submission_id=None,
             seed=57, optimizer_name="cma-es", family="gradient_free",
             dataset="wine", minutes=90),
        dict(run_name="de-digits-failed", task_status="failed",
             artifact_status="ready", executor_task_id="4812007",
             error_message="RuntimeError: CUDA out of memory. Tried to allocate "
                           "2.41 GiB (GPU 0; 39.39 GiB total capacity)",
             submitted_by=researcher, submission_id=None, seed=71,
             optimizer_name="de", family="gradient_free", dataset="digits",
             minutes=140),
        dict(run_name="des-wine-no-artifacts", task_status="failed",
             artifact_status="empty", executor_task_id="4812008",
             error_message="No files found under /net/people/plgrid/plggolem/"
                           "projekt-benchmark/reports/task_...",
             submitted_by=researcher, submission_id=None, seed=89,
             optimizer_name="des", family="gradient_free", dataset="wine",
             minutes=200),
        dict(run_name="broken-optimizer-rejected", task_status="failed",
             artifact_status="absent", executor_task_id=None,
             error_message="Zgłoszenie odrzucone przez walidator protokołu.",
             submitted_by=guest, submission_id=rejected, seed=11,
             optimizer_name="broken-optimizer", family="gradient",
             dataset="digits", minutes=240),
    ]

    for state in states:
        minutes = state.pop("minutes")
        created = now - timedelta(minutes=minutes)
        task_id = _insert_task(
            conn, **common, **state, created_at=created, queued_at=created,
            started_at=created + timedelta(minutes=8)
            if state["task_status"] != "pending" else None,
            completed_at=created + timedelta(minutes=20)
            if state["task_status"] in ("completed", "failed") else None,
            updated_at=created + timedelta(minutes=10),
        )
        # The failed run that did produce a log gets one: §11.3 is explicit that
        # the tail of .out must be visible rather than hidden behind a click,
        # and there is nothing to show if the file does not exist.
        if state["run_name"] == "de-digits-failed":
            root = downloads / str(task_id)
            (root / "logs").mkdir(parents=True, exist_ok=True)
            (root / "logs" / "4812007.out").write_text(
                "srun: job 4812007 queued and waiting for resources\n"
                "srun: job 4812007 has been allocated resources\n"
                "Loading dataset digits ...\n"
                "Building model mlp-3x64 (parameters=54410)\n"
                "epoch   1  loss=  2.2814  acc= 18.32%  grads=       0  samples=    57504\n"
                "epoch   2  loss=  2.1077  acc= 27.10%  grads=       0  samples=   115008\n"
                "Traceback (most recent call last):\n"
                '  File "run_benchmark.py", line 88, in <module>\n'
                "    result = runner.run(optimizer_class)\n"
                "RuntimeError: CUDA out of memory. Tried to allocate 2.41 GiB "
                "(GPU 0; 39.39 GiB total capacity; 36.02 GiB already allocated)\n"
                "srun: error: g0231: task 0: Exited with exit code 1\n",
                encoding="utf-8",
            )
            (root / "metadata.json").write_text(
                json.dumps({"task_id": str(task_id), "status": "failed",
                            "stop_reason": None}, indent=2),
                encoding="utf-8",
            )
            files = [p for p in root.rglob("*") if p.is_file()]
            with conn.cursor() as cur:
                cur.execute(
                    """UPDATE tasks SET artifact_root=%s, artifact_files=%s,
                                        artifact_bytes=%s WHERE task_id=%s""",
                    (str(root), len(files), sum(p.stat().st_size for p in files), task_id),
                )
    conn.commit()


if __name__ == "__main__":
    main()
