# Weryfikacja repozytorium — pass na żywym kodzie

**Projekt:** `projekt-benchmark` · **Data:** 19.08.2026 · **Gałąź:** `main`
**Status:** zastępuje sekcje **15** i **17** dokumentu `WEB_MVP_PLAN.md` (rew. 2)

Poprzednie rewizje planu opierały się wyłącznie na `BRIEF.md` i oznaczały twierdzenia
o repo jako `[wg briefu] NIEZWERYFIKOWANE`. Ten dokument je weryfikuje na kodzie.
**Każda pozycja niżej ma cytat `plik:linia`.**

Wynik zbiorczy: **6 defektów potwierdzonych, 1 obalony, 2 twierdzenia briefu nieaktualne.**

---

## 1. CI jest zepsute — POTWIERDZONE ✅

`.github/workflows/ci.yml:27` i `:43` wołają:

```yaml
run: uv sync --extra ci
```

`pyproject.toml` **nie ma sekcji `[project.optional-dependencies]` w ogóle**. Ma wyłącznie
`[dependency-groups]` (`pyproject.toml:32-47`) z grupami `athena-downloader`, `athena-poller`,
`athena-worker`. Ekstras `ci` nie istnieje → `uv sync --extra ci` kończy się błędem
→ **oba joby (`lint` i `test`) padają na kroku instalacji.** Brief §6 miał rację.

### Defekt wtórny, którego brief nie wychwycił

Nawet po naprawieniu ekstrasu **`uv run flake8` (`ci.yml:30`) i `uv run pytest` (`ci.yml:53`)
nadal padną** — `flake8` ani `pytest` nie występują w żadnej liście zależności w całym
`pyproject.toml`. Naprawa musi je dodać, nie tylko utworzyć pustą sekcję.

### Naprawa

```toml
# pyproject.toml — wariant zgodny z obecnym workflow (bez zmian w ci.yml)
[project.optional-dependencies]
ci = ["flake8>=7", "pytest>=8"]
```

Wariant alternatywny (`[dependency-groups] ci` + `uv sync --group ci` w workflow) jest
semantycznie czystszy — to zależności deweloperskie, nie opcjonalna funkcja pakietu — ale
wymaga zmiany w dwóch miejscach `ci.yml`. **Rekomendacja: wariant z `--group`**, bo repo już
używa `[dependency-groups]` dla serwisów Ateny, więc jest spójny z własną konwencją.

### ⚠ Trzeci problem — koszt instalacji w CI

`pyproject.toml:11-26` deklaruje jako **twarde zależności projektu**: `torch>=2.0`,
`torchvision>=0.15`, `cupy-cuda13x>=14.0.1`, `scikit-learn`, `scipy`, `pandas`, `matplotlib`.
`uv sync` w CI zaciągnie to wszystko — łącznie z CuPy skompilowanym pod CUDA 13 — żeby
uruchomić linter. To minuty czasu i realne ryzyko timeoutów oraz flaków.

Rekomendacja: job lintujący powinien instalować **wyłącznie** grupę `ci`
(`uv sync --only-group ci`), bez zależności runtime projektu. Do lintowania nie jest
potrzebny PyTorch.

---

## 2. Martwa konfiguracja flake8 — NOWE, nie ma tego w briefie ✅

`pyproject.toml:28-30` zawiera:

```toml
[tool.flake8]
max-line-length = 120
exclude = [".venv", ".git", "__pycache__", "reports","data",".github","weights"]
```

**flake8 nie czyta `pyproject.toml`.** Natywnie obsługuje `setup.cfg`, `tox.ini` i `.flake8`
(wsparcie dla TOML wymaga wtyczki, której repo nie ma). Ta sekcja jest **martwa**.

Faktycznie działa `setup.cfg:1-7`, który ustawia ten sam `max-line-length = 120` — więc
**zachowanie jest przypadkowo poprawne**, ale istnieją dwie konfiguracje, z czego jedna jest
ignorowana. Ktoś, kto zmieni limit w `pyproject.toml`, nie zrozumie, czemu nic się nie stało.

Ta sama duplikacja: `src/frontend/pyproject.toml:24-26` (również martwa).

**Dodatkowo** `setup.cfg:9-11` używa klucza `include =`, który **nie jest opcją flake8**
(flake8 zna `filename` i `extend-exclude`, nie `include`). Klucz jest po cichu ignorowany.

**Naprawa:** usunąć `[tool.flake8]` z obu plików `pyproject.toml`, zostawić `setup.cfg` jako
jedyne źródło prawdy, usunąć nieistniejący klucz `include`.

---

## 3. Niespójne importy — POTWIERDZONE i dokładnie ograniczone ✅

Brief §9 podejrzewał problem. Jest realny, ale **precyzyjnie ograniczony do 4 linii** —
i nie jest „ryzykiem", tylko dowiedzionym błędem.

Cztery miejsca używają formy `frontend.core.*`:

| Plik:linia | Import |
|---|---|
| `src/frontend/core/database.py:8` | `from frontend.core.config import get_database_url` |
| `src/frontend/auth/repository.py:9` | `from frontend.core.database import get_connection` |
| `src/frontend/auth/recaptcha_widget.py:3` | `from frontend.core.config import get_recaptcha_site_key` |
| `src/frontend/auth/recaptcha.py:3` | `from frontend.core.config import get_recaptcha_min_score, get_recaptcha_secret_key` |

Cała reszta (~30 importów) używa formy płaskiej: `from auth import repository`,
`from views.leaderboard import ...` itd.

### Dowód, że forma `frontend.*` jest błędna

`src/frontend/pyproject.toml:22`:

```toml
[tool.setuptools]
packages = ["core", "auth", "views", "views.admin", "db"]
```

**Nie istnieje pakiet `frontend`.** Pakiety są zadeklarowane jako płaskie, top-level.
Potwierdza to układ kontenera: `src/frontend/Dockerfile:5` ustawia `WORKDIR /app`, a `:10`
robi `COPY . .` — w `/app` lądują katalogi `auth/`, `core/`, `views/`, bez nadrzędnego
`frontend/`. `import frontend.core` nie ma prawa się rozwiązać.

### Dlaczego to blokuje plan

**`auth/repository.py` to dokładnie ten plik, który §19.2 briefu każe reużyć**
(*„reużywając istniejące `auth/repository.py` — to nie jest przepisywanie od zera"*).
Jego import w linii 9 jest zepsuty. **Naprawa tych 4 linii jest warunkiem wstępnym kroku 2
z planu**, nie sprzątaniem na później.

Naprawa jest trywialna: `frontend.core.X` → `core.X` w czterech miejscach.

---

## 4. Brak frontendu w `docker-compose.yml` — POTWIERDZONE ✅

`docker-compose.yml` definiuje pięć serwisów: `postgres` (:2), `rabbitmq` (:22),
`athena_worker` (:44), `athena_poller` (:57), `athena_downloader` (:70). **Nie ma serwisu
frontendu.**

Przy czym `src/frontend/Dockerfile` **istnieje i wygląda na kompletny** — jest tylko nigdy
nie budowany. Luka jest więc mniejsza, niż sugeruje brief: to brak ~10 linii w compose,
nie brak konteneryzacji.

Uwaga do montowania artefaktów: `docker-compose.yml:83` montuje `./downloads:/downloads`
do downloadera **do zapisu** (poprawnie). Serwis web musi dostać ten sam wolumen
**`:ro`** — zgodnie z sekcją 4 planu.

---

## 5. `frontend/env` NIE jest w `.gitignore` — POTWIERDZONE, wysokie ryzyko ✅

KROK 2 zadania każe utworzyć plik `frontend/env` (bez kropki) z hasłem do bazy i ostrzega,
że sekrety nie mogą trafić do commita. **Obecna konfiguracja go nie chroni.**

Sprawdzone wzorce:

| Plik:linia | Wzorzec | Czy łapie plik `env`? |
|---|---|---|
| `.gitignore:139` | `.env` | ❌ inna nazwa |
| `.gitignore:141` | `env/` | ❌ ukośnik na końcu → **tylko katalogi** |
| `src/frontend/.gitignore:2` | `.env` | ❌ inna nazwa |

Plik o nazwie `env` (zwykły plik, nie katalog) **nie pasuje do żadnego wzorca** i zostanie
zaproponowany do commita przez `git add`.

**Naprawa przed utworzeniem pliku** — dopisać do `src/frontend/.gitignore`:

```
env
```

### Co jest chronione poprawnie ✅

- `secrets.toml` — pokryty dwukrotnie: `.gitignore:146` (wzorzec bez ukośnika łapie na każdym
  poziomie) oraz `src/frontend/.gitignore:1` (`.streamlit` ignoruje cały katalog).
- `downloads/` — `.gitignore:22`. Artefakty runów nie trafią do repo.
- `.env` w korzeniu — `.gitignore:139`.

---

## 6. Rozbieżność: `frontend/env` vs `.env` w korzeniu — NOWE ⚠

Zadanie każe umieścić `POSTGRES_*` w `frontend/env`. **Compose czyta co innego.**

Każdy serwis w `docker-compose.yml` ma `env_file: .env` (linie 6, 26, 55, 68, 81) — czyli
**`.env` w korzeniu repo**. Szablon `.env.template:1-16` potwierdza, że to tam należą
`POSTGRES_DB/USER/PASSWORD/PORT`, `RABBITMQ_*` oraz `ATHENA_*`.

Wniosek: żeby `docker compose up` zadziałał, zmienne Postgresa muszą być w **`/.env`**,
nie w `frontend/env`. Plik `frontend/env` może być dodatkowo potrzebny dla samego procesu
Streamlita — ale nie zastępuje `.env` w korzeniu.

**Rekomendacja:** utworzyć `/.env` na bazie `.env.template` (wypełniając też `RABBITMQ_*`
i `ATHENA_*`, których zadanie nie podało — bez nich `athena_*` nie wstaną).

---

## 7. Brief §7 i D8 są NIEAKTUALNE — kod jest na `main` ✅✅

To jest najważniejsze ustalenie tego passa, bo **zmienia plan pracy**.

Brief §7 twierdzi: *„Kod został usunięty z `main` podczas refaktoru i wymaga przywrócenia —
poniżej stan z `891bf11^`"*, a D8 pyta *„Kiedy przywrócić runner na `main`"*.

**Refaktor już wylądował.** Kod jest na `main`, tylko pod inną ścieżką — `src/benchmark_core/`
zamiast `src/benchmark/`:

| Komponent | Ścieżka na `main` |
|---|---|
| Ewaluator | `src/benchmark_core/optimization_engine/evaluator.py` |
| Runner | `src/benchmark_core/optimization_engine/runner.py` |
| Wejście | `src/benchmark_core/optimization_engine/run_benchmark.py` |
| Protokoły optymalizatora | `.../optimizer_protocols/` (7 plików, w tym oba przykłady) |
| Optymalizatory NumPy | `.../optimizers/numpy/` (adam, adamw, lion, rmsprop, sgd, des, de, cmaes) |
| Optymalizatory CuPy | `.../optimizers/cupy/` (komplet analogiczny) |
| Rejestr | `.../optimizers/registry.py` |
| DTO | `.../evaluator_dtos/` (numpy, cupy, pytorch, konwertery, rejestr) |
| Warunki stopu | `src/benchmark_core/metrics/stop_metrics.py` |
| Wykresy | `src/benchmark_core/plotting/benchmark_analyzer.py` |

Ewaluator ma komplet metod z kontraktu §7 briefu — zweryfikowane w
`evaluator.py`: `batch_size:72`, `param_count:77`, `get_params:81`, `set_params:95`,
`evaluate:108`, `evaluate_with_grad:127`, `grad:152`, `get_predictions:176`.

### Konsekwencja dla planu

- **D8 jest faktycznie zamknięta.** Nie ma czego „przywracać".
- Sekcja `/docs` (13 planu) może opisywać **żywy kod**, a nie stan z historii — i powinna
  importować przykłady bezpośrednio z `optimizer_protocols/example_gradient_optimizer.py`
  i `example_evolutionary_optimizer.py`, zamiast je duplikować. To rozwiązuje problem
  rozjeżdżania się dokumentacji z kodem.
- Seed danych testowych (14.1 planu) może użyć **prawdziwego runnera na CPU**, zamiast
  fabrykować szeregi.

---

## 8. Walidator jest częściowo na `main` — brief §9 nieaktualny ⚠

Brief §9 klasyfikuje „Upload optymalizatora + walidacja" jako *„na branchu —
`origin/optim_validation`, działa, niezmergowane"*.

Tymczasem `src/benchmark_core/optimization_engine/optimizers/validation/verify_optimizer.py`
**jest na `main`**. To jest ten sam plik, który brief §6 rysuje w diagramie architektury jako
rdzeń silnika walidacyjnego.

Czego **nie** znalazłem na `main`: `frontend/core/validator.py` — czyli warstwy uruchamiającej
sandbox Dockera (`--network none`, `--memory 2g`, `--read-only`, timeout 30 s, `sandboxuser`),
którą brief §7 lokalizuje właśnie na gałęzi `optim_validation`.

**Wniosek:** krok 10 planu („zmerguj walidator z gałęzi") wymaga przescopowania — do
przeniesienia jest prawdopodobnie **tylko warstwa sandboxa**, nie sam walidator. Do
potwierdzenia przez `git diff main origin/optim_validation --stat` w sesji z powłoką.

---

## 9. OBALONE: brak `uv.lock` dla frontendu ❌

Rozważałem, czy `src/frontend/Dockerfile:7` (`COPY pyproject.toml uv.lock ./`) nie wywali się
na brakującym locku. **Nie wywali** — `src/frontend/uv.lock` istnieje. Odnotowuję jako
sprawdzone i czyste, żeby nikt nie tracił na to czasu.

---

## 10. Model danych — zgodny z briefem, plan kompatybilny ✅

`src/db/db_schemas/03_tasks_schema.sql:1-17` definiuje `tasks` dokładnie tak, jak opisuje
brief §10 (`task_id`, `queue_name`, `executor_name`, `submitted_by`, `task_status`,
`created_at`, `updated_at`, `dataset`, `run_name`, `optimizer_params`, `completed_at`,
`error_message`, `executor_task_id`).

Wszystkie kolumny dokładane przez sekcję 5.3 planu (`submission_id`, `seed`, `suite`,
`model_name`, `stop_condition`, `artifact_*`, `queued_at`, `started_at`, `runner_version`,
`gpu_model`) są **nowe — brak kolizji nazw**. `ALTER TABLE` z planu zadziała bez zmian.

Potwierdza się też defekt z sekcji 6 planu: `docker-compose.yml:15` montuje schematy do
`/docker-entrypoint-initdb.d`, który **wykonuje się wyłącznie na pustym wolumenie**. Migracje
przyrostowe są konieczne.

---

## Kolejność napraw

Uszeregowana wg tego, co blokuje co:

| # | Naprawa | Blokuje | Koszt |
|---|---|---|---|
| 1 | `env` do `src/frontend/.gitignore` | **Bezpieczeństwo — przed utworzeniem pliku** | 1 linia |
| 2 | 4 importy `frontend.core.*` → `core.*` | Krok 2 planu (reużycie `repository.py`) | 4 linie |
| 3 | `/.env` z `.env.template` (+ `RABBITMQ_*`, `ATHENA_*`) | `docker compose up` | plik |
| 4 | `[project.optional-dependencies].ci` lub `--group ci` + flake8/pytest | Zielone CI | ~5 linii |
| 5 | `uv sync --only-group ci` w jobie lintującym | Czas i stabilność CI | 2 linie |
| 6 | Usunąć martwe `[tool.flake8]` z obu `pyproject.toml`, usunąć klucz `include` | Higiena | ~8 linii |
| 7 | Serwis `web` w compose + `/downloads:ro` | „Cały system z jednego compose" | ~12 linii |
| 8 | Przescopować krok 10 planu po `git diff` z `optim_validation` | Zakres pracy nad walidatorem | analiza |

Pozycje 1–2 są **warunkami wstępnymi** — reszta planu na nich stoi.

---

## Czego ten pass NIE objął

Uczciwie, żeby nikt nie założył większego pokrycia niż jest:

- **Nie uruchamiałem niczego.** Brak powłoki w tej sesji — zero `docker compose up`, zero
  `uv sync`, zero `pytest`. Wszystkie defekty ustalone przez czytanie kodu i konfiguracji.
  Defekt CI (1) jest pewny przez inspekcję, ale nie widziałem czerwonego builda.
- **Nie porównałem `main` z `origin/optim_validation`** — wymaga `git diff`. Stąd punkt 8
  jest hipotezą opartą na obecności/nieobecności plików, nie na diffie.
- **Nie czytałem `views/*.py` w całości** — potwierdziłem tylko importy i istnienie
  `views/mock_data.py` (do usunięcia wg §13.1 briefu).
- **Nie audytowałem `auth/repository.py` merytorycznie** — potwierdziłem, że istnieje
  i że ma zepsuty import. Czy nadaje się do reużycia bez zmian, wymaga przeczytania treści.
- **Nie sprawdzałem historii pod kątem zacommitowanych sekretów.** `git log -p` bez powłoki
  jest niewykonalny.

---

*Weryfikacja przeprowadzona 19.08.2026 na `main` w `/srv/root/projekt-benchmark`.
Każde twierdzenie zakotwiczone w `plik:linia`. Zastępuje sekcje 15 i 17 `WEB_MVP_PLAN.md`.*
