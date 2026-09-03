# Running the project locally

Written for someone who has just cloned the repository and wants to click
through the whole thing. No cluster access is required and no secret has to be
sent to you: everything you need is generated on your own machine.

## 1. Why there is no `.env` in your clone

`.env`, `.seed-credentials` and `src/frontend/.streamlit/secrets.toml` are all
gitignored, so a fresh clone does not contain them. That is deliberate, and it
is also why nobody needs to send you theirs.

It helps to see that the project has three separate kinds of secret, and they
have nothing in common except the word:

| kind | examples | how you get it |
|---|---|---|
| **disposable, local** | `POSTGRES_PASSWORD`, `RABBITMQ_PASSWORD`, `SESSION_SECRET`, seed passwords | generate your own, below |
| **personal, cluster** | `ATHENA_HOST/USER/PASSWORD` | your own PLGrid account; never shared |
| **shared, real** | Google/Microsoft OAuth, reCAPTCHA | **not needed locally** |

The first kind is the one people are tempted to paste into a chat. Don't: a
password for a container listening on your own localhost protects nothing, and
sharing it converts a value with zero blast radius into one with an unknown
blast radius.

## 2. Generate your environment

```bash
./scripts/bootstrap-env.sh
```

Writes `.env` and `.seed-credentials`, both mode 600, both already gitignored.
It refuses to overwrite an existing `.env` unless you pass `--force`.

Optionally install the commit guard, which blocks a commit that would place a
secret-shaped value in a tracked file:

```bash
./scripts/check-secrets.sh --install
```

## 3. Start the stack

```bash
docker compose up -d --build postgres rabbitmq web outbox_publisher
```

Four services, not nine, and the omission is intentional: `athena_worker`,
`athena_poller` and `athena_downloader` need PLGrid credentials, and `frontend`
(the Streamlit app) needs `secrets.toml`. Neither is required for the control
plane. Starting them without credentials produces a restart loop and nothing
else.

Database migrations run automatically at boot (`RUN_MIGRATIONS` defaults to
true), so there is no separate migrate step.

Check it came up:

```bash
docker compose ps                      # web and outbox_publisher: healthy
curl -s localhost:8080/healthz
```

| service | URL |
|---|---|
| web (SPA + API) | http://localhost:8080 |
| RabbitMQ management | http://localhost:15672 |
| PostgreSQL | localhost:5432 |

## 4. Seed data

Without this the site is correct but empty, and most of it cannot be judged.
The seeder drives the project's own `ModelEvaluator` with the project's own
NumPy optimizers on three public scikit-learn datasets, so the convergence
curves and budget counters are measured rather than generated. It also creates
one run in each remaining state -- waiting in the broker, waiting in SLURM,
running, downloading, failed with a log, failed with no artifacts, rejected by
the validator -- because otherwise half the interface never renders.

```bash
uv sync --group frontend               # torch, scikit-learn, psycopg
```

One warning about that step: `cupy-cuda12x` is a hard dependency of the root
project, so `uv sync` pulls roughly a gigabyte of CUDA wheels even on a machine
with no NVIDIA card. It installs fine and nothing here imports it, but budget
the download. Then:

```bash
set -a; . ./.env; . ./.seed-credentials; set +a
export DATABASE_URL="postgresql://$POSTGRES_USER:$POSTGRES_PASSWORD@localhost:$POSTGRES_PORT/$POSTGRES_DB"
uv run python -m tools.local_backend.seed --downloads ./downloads
```

Takes a few minutes on a CPU; `--seeds 3 --epochs 6` makes it quicker, `--reset`
clears previous runs first.

Accounts it creates -- passwords are the two values in your `.seed-credentials`:

| account | role |
|---|---|
| `admin@benchmark.pw.edu.pl` | admin |
| `badacz@benchmark.pw.edu.pl` | verified |
| `gosc@benchmark.pw.edu.pl` | verified |
| `nowy@benchmark.pw.edu.pl` | unverified, awaiting approval |

## 5. What to click through

1. **Register** a new account. You land on "waiting for approval" -- registering
   does not grant access. Log in as the admin, go to `/admin`, approve it.
2. **`/leaderboard`** -- switch between the two scoring formulas and watch the
   ranking change. The note under each explains that it does not normalise by
   budget.
3. **`/runs`** -- filter by dataset, model, optimizer, family. Open one run:
   convergence curves, `gradient_count`, `database_reaches`, stop reason.
4. **`/compare`** -- put several runs side by side, export CSV.
5. **Artifact browser** -- PNGs inline, CSVs and logs as text, whole run as ZIP.
   Try putting `..` in the path; it should be refused.
6. **`/submit`** -- download the optimizer template, submit it, watch the daily
   quota decrease.
7. **`/admin`, `/admin/queue`** -- approval queue, live RabbitMQ depth, outbox
   state, budget.
8. **The edge states** seeded in step 4. Each renders differently, and the failed
   ones say what failed.

## 6. What will not work locally, and why that is correct

**A submitted run never starts.** It reaches `queued_broker` ("w kolejce
systemu") and stops. The full path -- form, validation, database write, outbox
row, publisher, RabbitMQ -- works; what is missing is a consumer, because
`athena_worker` needs cluster credentials. Confirm with:

```bash
curl -s localhost:15672/api/queues -u "$RABBITMQ_USER:$RABBITMQ_PASSWORD" \
  | grep -o '"name":"ATHENA_WORKER_QUEUE"[^}]*'
```

`consumers: 0` is the whole story. The rows already showing "running on Athena"
or "downloading" are seeded fixtures, not live cluster state.

**Submissions are accepted without being checked.** `VALIDATOR_ENABLED=0`,
because the validator needs a Docker socket to start its sandbox. Submissions
carry an explicit note saying nothing was verified, rather than silently
appearing to have passed. Set it to `1` if you have a socket to spare.

**OAuth and reCAPTCHA are absent.** The control plane never reads
`secrets.toml`; it has no Streamlit dependency at all, and the OAuth client ids
default to empty. Email and password login is unaffected. Only the Streamlit
`frontend` service needs those values, and it is not part of this stack.

## 7. Tests

```bash
cd src/web && uv run pytest
```

Currently **44 passed, 30 skipped**. The skips are not failures: `test_api.py`
and `test_spa_routing.py` need a live database and stand down unless you point
them at one.

```bash
TEST_DATABASE_URL="$DATABASE_URL" uv run pytest    # runs the other 30 too
```

Frontend:

```bash
cd src/web/frontend && npm ci
npm run lint
npm run lint:css                         # rejects gradients, glow and heavy shadows
npm run build
```

## 8. If you break it

```bash
docker compose down -v                   # -v also drops the database volume
./scripts/bootstrap-env.sh --force
docker compose up -d --build postgres rabbitmq web outbox_publisher
# then seed again
```

Nothing here is precious. Every value is regenerable and every run in
`downloads/` can be recreated by the seeder.
