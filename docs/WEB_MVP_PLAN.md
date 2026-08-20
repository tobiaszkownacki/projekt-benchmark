# Warstwa webowa — plan i decyzje architektoniczne

**Projekt:** Benchmark Czarnej Skrzynki (`projekt-benchmark`, Politechnika Warszawska)
**Docelowa lokalizacja:** `docs/WEB_MVP_PLAN.md`
**Data:** 19.08.2026 · rew. 2 (stack frontendu ustalony przez właściciela)
**Podstawa:** `BRIEF.md` (952 linie, 17.08.2026) — przeczytany w całości

---

## 0. Status tego dokumentu i uczciwe zastrzeżenie

Ten dokument powstał **wyłącznie na podstawie `BRIEF.md`**. Repozytorium nie było dostępne
w sesji, w której go pisano. Ma to trzy konsekwencje:

1. **Nie ma tu ani jednego cytatu `plik:linia` z kodu, którego bym nie widział.** Odwołania do
   istniejącego kodu (`auth/repository.py`, `views/run_form.py`, `benchmark_analyzer.py`,
   `athena_downloader.py`) są **relacją z briefu**, nie weryfikacją. Oznaczam je `[wg briefu]`.
2. **Każde twierdzenie o stanie repo wymaga potwierdzenia** przy pierwszym kontakcie z kodem —
   zwłaszcza defekt CI (§15.1) i niespójne importy (§15.2).
3. **Decyzje architektoniczne są ważne mimo to** — opierają się na wymaganiach (§5, §11, §12,
   §14, §15), składzie zespołu (§4) i horyzoncie utrzymania (§2), udokumentowanych niezależnie
   od kodu.

### Zmiana w rewizji 2

Stack frontendu został **ustalony decyzją właściciela: React**. Poprzednia rewizja
rekomendowała renderowanie po stronie serwera (Jinja2 + htmx). Sekcja 2 jest przepisana pod
tę decyzję, wraz z uczciwym opisem ryzyka, które ona wnosi, i sposobu jego ograniczenia.

**Warte odnotowania:** zmiana frontendu dotknęła **wyłącznie sekcji 2, 4, 8.5, 12 i 14**.
Model danych (5), kontrakt API (7), specyfikacja bezpieczeństwa (8) i metodologia statystyczna
(9) pozostały bez zmian. To nie przypadek — to jest dokładnie ten zysk, dla którego rozdziela
się control plane od warstwy prezentacji (§15 briefu). Front dało się wymienić bez ruszania
systemu.

---

## 1. GOAL

> **Zbudować control plane i publiczną warstwę webową, które (a) przetrwają jako fundament
> konkursu GECCO 2027, (b) na demo dla Zakładu SI EIT dowodzą, że system naprawdę policzył,
> i (c) dadzą się utrzymać przez kolejne roczniki studentów przez 5–10 lat.**

Kolejność priorytetów przy konflikcie:

- **(a) wygrywa z (b)** przy wyborze zakresu. Control plane (§15) ma wartość nawet gdyby UI
  nigdy nie powstało — bez niego konkurs nie ma jak przyjmować zgłoszeń programowo.
- **(b) decyduje o jakości wykonania.** Odbiorcą są doktorzy i profesorowie, a wygląd jest —
  dosłownie za briefem (§4) — *„warunkiem wiarygodności przy negocjacjach o dostęp do klastra
  wartego miliony"*.
- **(c) jest ograniczeniem**, nie priorytetem konkurującym. §2: *„przez najbliższe 5–10 lat
  […] utrzymywany przez kolejne pokolenia studentów"*. Wybór Reacta podnosi koszt (c) —
  sekcja 2.4 opisuje, czym to kompensujemy.

### Czego GOAL nie obejmuje

Nie jest celem „przepisanie projektu na nowoczesny stack". Jest celem **usunięcie jednej
granicy**: dziś warstwa prezentacji ma poświadczenia do bazy i RabbitMQ [wg briefu §9]. Przy
konkursie z zewnętrznymi uczestnikami to jest granica, której nie chce się mieć w kodzie UI.

---

## 2. Decyzja D1 — Streamlit odchodzi, front w React

**Decyzja: Streamlit odchodzi z warstwy publicznej. Frontend w React. Backend zostaje
w Pythonie.**

### 2.1 Co przesądza o odejściu od Streamlita

Jedno twarde wymaganie zamyka sprawę bez dyskusji o gustach:

> §11.2: **„Zasada projektowa: każdy zasób ma własny URL."** *„Uczestnik konkursu chce wkleić
> link do swojego wyniku w mailu albo w artykule."*

`st.query_params` pozwala zakodować stan w query stringu, ale
`/runs/<id>/files?path=reports/loss_vs_grads.png` jako adresowalny, dzielony zasób — nie.
To **wymóg produktowy, nie estetyczny**.

Drugi, niezależny argument: §15 wymaga control plane, bo GECCO potrzebuje zgłoszeń z CLI/CI.

### 2.2 Wybrany stack

| Warstwa | Wybór | Dlaczego |
|---|---|---|
| Control plane | **FastAPI** (async), psycopg 3 (async) | §15; psycopg 3 już w projekcie [wg briefu §6] |
| Frontend | **React + TypeScript + Vite** | Decyzja właściciela |
| Routing | **React Router** (data router) | §11.2 — każdy zasób ma URL |
| Serwowanie | **`vite build` → statyki serwowane przez FastAPI** | Jeden kontener w runtime, patrz 2.3 |
| Style | **CSS Modules + plik tokenów** (bez Tailwinda) | Egzekwowalność §14, patrz 12.1 |
| Wykresy | **uPlot** + cienki wrapper React | Tysiące punktów + natywne wstęgi (IQR) |
| Live status | **SSE** (`EventSource`) + `LISTEN/NOTIFY` | §15 |
| Migracje | numerowane `.sql` + `schema_migrations` | Patrz sekcja 6 |
| Auth | **reużycie `auth/repository.py`** + Authlib + bcrypt | §19.2 briefu; Authlib już jest |

**TypeScript, nie czysty JS.** Przy kontrakcie API z sekcji 7 typy generowane z OpenAPI
(FastAPI wystawia schemat za darmo) dają zgodność front–back sprawdzaną przy budowaniu.
W projekcie utrzymywanym przez rotujących studentów to jest najtańsza dostępna dokumentacja.

### 2.3 Vite SPA, nie Next.js — i dlaczego to ma znaczenie

React można tu postawić na dwa sposoby. Wybieram **Vite + statyki serwowane przez FastAPI**,
nie Next.js, z jednego konkretnego powodu:

> §5.1: *„po prostu cztery dokery, które sobie siedzą"* · §9: cel zespołu to **„cały system
> z jednego docker compose"**.

Next.js wymaga **procesu Node w runtime** → piąty kontener, drugi runtime do utrzymania,
druga rzecz do zaktualizowania przy CVE. Vite kompiluje do plików statycznych: **Node jest
potrzebny wyłącznie w etapie budowania obrazu**, a w runtime FastAPI serwuje `dist/` obok
`/api`. Jeden kontener, jeden port, jeden origin — i przy okazji znika cała klasa problemów
z CORS i z ciasteczkami cross-origin.

```dockerfile
FROM node:22-alpine AS frontend
WORKDIR /build
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build            # → /build/dist

FROM python:3.12-slim
...
COPY --from=frontend /build/dist /app/static
# FastAPI: mount statyków + catch-all → index.html (deep linki działają po odświeżeniu)
```

**Co przez to tracimy — uczciwie:** brak SSR, więc link wklejony w mailu pokazuje przez chwilę
pusty ekran zanim się wyrenderuje. Przy ~100 użytkownikach i sieci uczelnianej to ułamek
sekundy, ale to jest realna różnica wobec Papers With Code, które renderuje się serwerowo.
Publiczny leaderboard nie będzie też indeksowany przez wyszukiwarki.

**Jeśli to zacznie przeszkadzać** — Next.js można dołożyć przed tym samym API bez dotykania
backendu. To jest dokładnie ta opcjonalność, dla której rozdziela się control plane. Nie
płacimy za nią teraz.

### 2.4 Czego React kosztuje i czym to kompensujemy

Nie udaję, że ta decyzja jest darmowa. Ryzyko jest konkretne i wynika z §2 i §4 briefu:
horyzont 5–10 lat, rotacja studentów, zespół Pythona, a brief mówi wprost *„nie wiem, czy
ktokolwiek faktycznie ma piękną wizję na backend"* (§9). Łańcuch narzędzi npm wymaga
konserwacji, której szablon serwerowy nie wymaga.

Cztery konkretne środki, nie deklaracje:

1. **Zero zależności runtime poza React, React Router, uPlot.** Każda kolejna biblioteka to
   dług, który spłaci ktoś, kogo dziś nie ma w zespole. Brak Tailwinda, brak biblioteki
   komponentów, brak state managera (React Router data loaders wystarczą).
2. **`package-lock.json` commitowany, `npm ci` w budowaniu.** Build odtwarzalny za 3 lata.
3. **Node tylko w etapie budowania** (2.3). Awaria/CVE ekosystemu Node nie dotyka
   działającego systemu.
4. **Typy generowane z OpenAPI**, nie pisane ręcznie. Zmiana kontraktu backendu psuje build
   frontendu natychmiast, a nie na demo.

### 2.5 Dlaczego nie Go w backendzie

Rozważone i **odrzucone**. Powody, w kolejności wagi:

1. **Brief wprost każe reużyć autoryzację.** §19.2: *„reużywając istniejące
   `auth/repository.py` — **To nie jest przepisywanie od zera**"*. Go oznacza napisanie od zera
   OIDC (Google + Microsoft), bcrypt, reCAPTCHA v3 i semantyki sesji — czyli jedynej części
   obecnej strony, która **już działa** (§9). To jest ruch wprost przeciwny instrukcji briefu,
   przy realnym ryzyku regresji w bezpieczeństwie.
2. **Cała reszta systemu to Python.** Ewaluator, optymalizatory, runner, walidator, worker,
   poller, downloader. Drugi język w projekcie utrzymywanym przez studentów ML to bariera
   wejścia, nie zysk.
3. **Zalety Go nie mają tu zastosowania.** Statyczny binarny plik, tanie gorutyny, niskie
   zużycie pamięci — to rozwiązania problemów, których ten projekt nie ma przy ~100
   użytkownikach (§3). Brief mówi wprost: *„To zdejmuje presję na architekturę skalowalną —
   problemem nie jest przepustowość"*.
4. **Jedyne miejsce, gdzie Go byłby wygodny** — SSE z wieloma otwartymi połączeniami — jest
   w Pythonie rozwiązane przez asyncio bez trudu przy tej skali.

Gdyby kiedyś doszedł komponent o realnych wymaganiach przepustowościowych (np. proxy do
strumieniowania artefaktów o dużym wolumenie), Go jest sensownym wyborem **dla tego jednego
komponentu**. Nie dla control plane.

### 2.6 Los Streamlita

Streamlit znika z warstwy publicznej. `/admin` i `/admin/queue` powstają w Reakcie razem
z resztą — mając system wizualny i komponent tabeli, strona administracyjna jest najtańsza
do zbudowania, a utrzymywanie dwóch frontendów jest kosztem samym w sobie.

**Ścieżka odwrotu:** kolejność pracy (sekcja 14) jest ułożona tak, że gdyby zabrakło czasu
przed 31.08, istniejący panel Streamlita może tymczasowo zostać, ale **wpięty w API zamiast
w bazę** (krok 3 z §19 briefu). Poświadczenia i tak znikają z UI. To zaplanowana ścieżka,
nie improwizacja.

### 2.7 Czego ta decyzja nie przesądza

D1 dotyczy technologii. **Nie zamykam D2–D8** — nie mam do tego mandatu (sekcja 17).
Architektura jest zbudowana tak, żeby D2 (waluta budżetu), D3 (powtórzenia i test
statystyczny) i D4 (publiczność wyników) dało się rozstrzygnąć **bez zmiany schematu bazy
i bez zmiany kontraktu API**.

---

## 3. Zakres — pełna mapa stron

Zakres rozszerzony: **wszystkie strony z §11.2, łącznie z P1 i P2.**

| Ścieżka | Strona | Kluczowa zawartość |
|---|---|---|
| `/` | Przegląd | 3 zdania czym to jest, top-10 leaderboardu, liczniki (zgłoszenia / ukończone runy / GPU-h), link do `/docs` |
| `/leaderboard` | Ranking | Pełna tabela, filtry (dataset / model / rodzina / suite), sortowanie po każdej kolumnie, wykres top-N, **wymienialna kolumna agregatu** |
| `/runs` | Uruchomienia | Lista z żywym statusem, powodem stopu, czasem w kolejce; filtry; `?mine=1` |
| `/runs/:id` | Szczegóły runu | Nagłówek stanu + metryki, wykresy zbieżności, przeglądarka plików, log SLURM |
| `/runs/:id/files` | Przeglądarka plików | §12 briefu w całości — sekcja 8 |
| `/submit` | Zgłoszenie | Upload `.py` lub wybór wbudowanego, dataset/model/budżet, **synchroniczny log walidatora** |
| `/docs` | Protokół | Żywy opis kontraktu `step()`, API ewaluatora, 2 pełne przykłady, zasady budżetu, szablon do pobrania |
| `/compare` | Porównanie | Pełne narzędzie analityczne — sekcja 9.3 |
| `/admin` | Panel | Zatwierdzanie kont, zużycie budżetu per użytkownik, unieważnianie zgłoszeń |
| `/admin/queue` | Kolejka | Stan RabbitMQ + DLQ, stan Ateny (`sinfo`/`sacct`), zadania w SLURM, sieroty |

Plus: `/api/*` (sekcja 7), `/api/events` (SSE), `/healthz`.

**Permalinki.** Każdy poniższy jest samodzielnym, dzielonym adresem — wymóg, nie udogodnienie:

```
/runs/9f2c…4b1e
/runs/9f2c…4b1e/files?path=reports/loss_vs_grads.png
/leaderboard?dataset=wine&family=gradient_free&suite=test&score=v1
/compare?runs=9f2c…,7a1b…,c33d…&x=gradient_count&logy=1
```

**Wymóg implementacyjny SPA:** serwer musi mieć catch-all zwracający `index.html` dla ścieżek
nieobsługiwanych przez `/api` i `/healthz` — inaczej odświeżenie strony na `/runs/<id>/files`
zwróci 404. To jest najczęstszy błąd przy SPA i trzeba go pokryć testem (sekcja 19, punkt 6).

---

## 4. Architektura docelowa

```
PRZEGLĄDARKA (React SPA)            CLI / CI uczestnika GECCO
        │                                    │
        │  HTTP — ten sam origin             │  Bearer token
        ▼                                    ▼
┌──────────────────────────────────────────────────────────┐
│  web  —  FastAPI                                         │
│  ├─ /            statyki z vite build (catch-all → SPA)  │
│  ├─ /api/*       control plane                           │
│  ├─ /api/events  SSE                                     │
│  └─ auth         reużyte auth/repository.py              │
│                                                          │
│  /downloads  zamontowany :ro  ◄── §12                    │
└───────┬──────────────────────┬───────────────────────────┘
        │ psycopg3 (async)     │ NIE publikuje na kolejkę
        ▼                      ▼
┌────────────────┐      ┌──────────────────┐
│ PostgreSQL 16  │      │  queue_outbox    │  (tabela, nie broker)
│ users · tasks  │◄─────┤  transakcyjnie   │
│ submissions    │      └────────┬─────────┘
│ results        │               │
│ result_series  │               ▼
│ transitions    │      ┌──────────────────┐
└───────┬────────┘      │ outbox-publisher │──► RabbitMQ ──► worker/poller/downloader
        │                └──────────────────┘                       │
        │ LISTEN task_changed                                       │ scp
        └──────────────► SSE ──► przeglądarka          /downloads/<task_id>/
```

Cztery zmiany względem stanu dzisiejszego:

1. **UI nie ma poświadczeń do RabbitMQ.** Publikacja przez outbox (sekcja 10).
2. **UI nie trzyma własnej puli psycopg** — jeden dostęp do bazy, przez warstwę repozytoriów.
3. **`/downloads` montowany read-only.** Obrona w głąb: błąd w kodzie nie zapisze niczego
   w katalogu artefaktów.
4. **Front i API na jednym originie** — brak CORS, ciasteczka `SameSite=Strict` działają
   bez wyjątków.

---

## 5. Model danych

Uzupełnia braki z §10 briefu. Wszystko jako **migracje przyrostowe** — `users` i `tasks`
są rozszerzane, nie przepisywane.

### 5.1 Typy

```sql
CREATE TYPE stop_reason_t       AS ENUM ('GRADIENT_LIMIT','DATABASE_LIMIT','EPOCH_LIMIT',
                                         'OPTIMIZER_CONVERGED','MAX_STEPS');
CREATE TYPE optimizer_family_t  AS ENUM ('gradient','gradient_free');
CREATE TYPE benchmark_suite_t   AS ENUM ('test','final');
CREATE TYPE submission_status_t AS ENUM ('validating','rejected','accepted');
CREATE TYPE artifact_status_t   AS ENUM ('absent','downloading','ready','empty','error');
```

`artifact_status_t` istnieje po to, żeby §11.3 („zakończone, artefakty się ściągają",
„nieudane bez artefaktów") miało reprezentację w danych, a nie było zgadywane z obecności
katalogu.

### 5.2 `submissions`

```sql
CREATE TABLE submissions (
  submission_id     UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  submitted_by      UUID NOT NULL REFERENCES users(id),
  display_name      TEXT NOT NULL,
  kind              TEXT NOT NULL CHECK (kind IN ('builtin','uploaded')),
  builtin_name      TEXT,
  source_code       TEXT,
  source_sha256     CHAR(64),
  output_type       TEXT,                    -- Numpy / Cupy / PyTorch DTO
  family            optimizer_family_t,
  status            submission_status_t NOT NULL DEFAULT 'validating',
  validator_log     TEXT,
  validator_version TEXT,
  created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
  validated_at      TIMESTAMPTZ,
  CONSTRAINT builtin_xor_source CHECK (
    (kind = 'builtin' AND builtin_name IS NOT NULL AND source_code IS NULL) OR
    (kind = 'uploaded' AND source_code IS NOT NULL AND source_sha256 IS NOT NULL)
  )
);
CREATE INDEX ON submissions (submitted_by, created_at DESC);
CREATE INDEX ON submissions (source_sha256) WHERE source_sha256 IS NOT NULL;
```

`source_sha256` daje trzy rzeczy naraz: wykrywanie duplikatów (oszczędność grantu),
powiązanie artefaktu z dokładną wersją kodu, reprodukowalność.

`family` jest **kolumną, nie wyliczeniem w locie** — §13.1 nazywa to „osią, wokół której
kręci się cała teza projektu", więc musi być filtrowalna w SQL.

### 5.3 Rozszerzenie `tasks`

```sql
ALTER TABLE tasks
  ADD COLUMN submission_id   UUID REFERENCES submissions(submission_id),
  ADD COLUMN seed            BIGINT,
  ADD COLUMN suite           benchmark_suite_t NOT NULL DEFAULT 'test',
  ADD COLUMN model_name      TEXT,
  ADD COLUMN stop_condition  JSONB,
  ADD COLUMN artifact_root   TEXT,
  ADD COLUMN artifact_status artifact_status_t NOT NULL DEFAULT 'absent',
  ADD COLUMN artifact_bytes  BIGINT,
  ADD COLUMN artifact_files  INTEGER,
  ADD COLUMN queued_at       TIMESTAMPTZ,
  ADD COLUMN started_at      TIMESTAMPTZ,
  ADD COLUMN runner_version  TEXT,
  ADD COLUMN gpu_model       TEXT;
```

`seed`, `runner_version`, `gpu_model` adresują ryzyko reprodukowalności z §18 (cuDNN
niedeterministyczne na A100). `queued_at`/`started_at` są potrzebne, żeby `/runs` pokazało
**czas w kolejce** — wymóg §11.2.

### 5.4 `results`

```sql
CREATE TABLE results (
  task_id           UUID PRIMARY KEY REFERENCES tasks(task_id) ON DELETE CASCADE,
  final_loss        DOUBLE PRECISION,
  final_accuracy    DOUBLE PRECISION,
  gradient_count    BIGINT NOT NULL,
  database_reaches  BIGINT NOT NULL,
  total_steps       BIGINT,
  total_epochs      INTEGER,
  wall_time_seconds DOUBLE PRECISION,
  stop_reason       stop_reason_t NOT NULL,
  recorded_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX ON results (final_loss);
```

### 5.5 `result_series` — szeregi czasowe

**Decyzja: tablice równoległe, jeden wiersz na run.** Nie wąska tabela, nie JSONB.

```sql
CREATE TABLE result_series (
  task_id           UUID PRIMARY KEY REFERENCES tasks(task_id) ON DELETE CASCADE,
  epochs            INTEGER[]          NOT NULL,
  loss              DOUBLE PRECISION[] NOT NULL,
  accuracy          DOUBLE PRECISION[],
  gradient_count    BIGINT[]           NOT NULL,
  database_reaches  BIGINT[]           NOT NULL,
  wall_time_seconds DOUBLE PRECISION[],
  CONSTRAINT series_same_length CHECK (
    cardinality(loss)             = cardinality(epochs) AND
    cardinality(gradient_count)   = cardinality(epochs) AND
    cardinality(database_reaches) = cardinality(epochs)
  )
);
```

**Uzasadnienie — nieoczywiste.** Naturalnym odruchem jest wąska tabela `(task, epoch, metric,
value)`, bo pozwala liczyć kwantyle w SQL przez `percentile_cont`. Ten odruch jest tu
**błędny**, z powodu merytorycznego, nie wydajnościowego:

> Oś X wykresu to **budżet, nie epoka** (§13.2). Różne optymalizatory zużywają różną ilość
> budżetu na epokę — CMA-ES z populacją λ=20 zużywa 20× więcej `database_reaches` na epokę
> niż SGD. Punkty pomiarowe **nie są wyrównane między runami na osi budżetu**. Żeby policzyć
> medianę „przy 10 000 gradientów", trzeba najpierw zinterpolować każdy run na wspólną siatkę
> budżetu — a tego SQL nie zrobi sensownie.

Skoro agregacja i tak musi się odbyć w Pythonie po wczytaniu całych serii, wąska tabela nie
daje nic, a kosztuje 5 × liczba_epok wierszy na run. Tablice są zwarte, TOAST je kompresuje,
odczyt to jeden wiersz. Struktura odwzorowuje 1:1 pięć równoległych szeregów z
`BenchmarkResult` [wg briefu §8]. `CHECK` na równej długości łapie klasę błędów, która inaczej
objawia się jako przesunięty wykres.

### 5.6 `task_state_transitions`

```sql
CREATE TABLE task_state_transitions (
  id          BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  task_id     UUID NOT NULL REFERENCES tasks(task_id) ON DELETE CASCADE,
  from_status TEXT,
  to_status   TEXT NOT NULL,
  actor       TEXT NOT NULL,     -- 'worker' | 'poller' | 'downloader' | 'api' | 'admin:<uuid>'
  detail      JSONB,
  occurred_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX ON task_state_transitions (task_id, occurred_at);
```

Adresuje ryzyko z §18: *„Czemu mój run wisi od dwóch godzin" = archeologia po pięciu
warstwach*. Brief nazywa to „tanie teraz, drogie później".

### 5.7 `queue_outbox`

```sql
CREATE TABLE queue_outbox (
  id           BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  exchange     TEXT NOT NULL,
  routing_key  TEXT NOT NULL,
  payload      JSONB NOT NULL,
  created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
  published_at TIMESTAMPTZ,
  attempts     INTEGER NOT NULL DEFAULT 0,
  last_error   TEXT
);
CREATE INDEX ON queue_outbox (id) WHERE published_at IS NULL;
```

Uzasadnienie w sekcji 10.

### 5.8 Powiadomienia dla SSE

```sql
CREATE FUNCTION notify_task_change() RETURNS trigger AS $$
BEGIN
  PERFORM pg_notify('task_changed', json_build_object(
    'task_id',      NEW.task_id,
    'status',       NEW.task_status,
    'artifact',     NEW.artifact_status,
    'submitted_by', NEW.submitted_by
  )::text);
  RETURN NEW;
END $$ LANGUAGE plpgsql;

CREATE TRIGGER tasks_notify
AFTER INSERT OR UPDATE OF task_status, artifact_status, executor_task_id ON tasks
FOR EACH ROW EXECUTE FUNCTION notify_task_change();
```

**Uwaga:** ładunek `pg_notify` ma limit ~8000 bajtów. Wysyłamy **tylko identyfikatory i stan**,
klient dociąga szczegóły przez `/api/runs/{id}`. Wpychanie metryk w notyfikację działa do dnia,
w którym przestaje.

---

## 6. Migracje

Obecnie schemat ładuje się przez `docker-entrypoint-initdb.d` [wg briefu §10]. Defekt:
**ten katalog wykonuje się wyłącznie na pustym wolumenie**. Każda zmiana schematu na
działającej instancji zostanie po cichu pominięta.

**Rekomendacja:** numerowane pliki SQL + tabela `schema_migrations` + krótki runner
uruchamiany przy starcie kontenera `web`.

```
src/db/migrations/
  0001_initial.sql          -- odtworzenie obecnego stanu (users, tasks)
  0002_submissions.sql
  0003_results_series.sql
  0004_transitions.sql
  0005_outbox_notify.sql
```

Alembic byłby uzasadniony przy SQLAlchemy. Projekt używa surowego psycopg 3, więc dokładanie
SQLAlchemy tylko dla migracji to zależność bez pokrycia.

---

## 7. Kontrakt API

Prefiks `/api`. Uwierzytelnianie: ciasteczko sesyjne (przeglądarka) **albo**
`Authorization: Bearer <token>` (CLI/CI — wymóg GECCO z §15). Ten sam kod autoryzacji dla obu.

**Przy froncie w React ten kontrakt staje się jedynym interfejsem systemu** — nie ma już
ścieżki „szablon czyta bazę po cichu". To realny zysk: wszystko, co widzi przeglądarka,
przechodzi przez tę samą, testowalną powierzchnię, której używa CLI uczestnika GECCO.

**Typy dla frontendu generowane z OpenAPI** (`openapi-typescript`), commitowane do repo.
Rozjazd kontraktu psuje build, a nie demo.

### 7.1 Endpointy

| Metoda | Ścieżka | Zwraca |
|---|---|---|
| `POST` | `/api/submissions` | `submission_id`, `status`, `validator_log`, lista `task_id` |
| `GET` | `/api/submissions/{id}` | stan zgłoszenia + log walidatora |
| `GET` | `/api/runs` | lista; filtry `mine`, `status`, `dataset`, `model`, `family`, `suite`; kursor |
| `GET` | `/api/runs/{id}` | metadane, metryki, `stop_reason`, ID joba SLURM, `artifact_status` |
| `GET` | `/api/runs/{id}/series` | szeregi po downsamplingu; `?x=`, `?metric=`, `?points=` |
| `GET` | `/api/runs/{id}/files` | drzewo: ścieżka, rozmiar, typ, mtime |
| `GET` | `/api/runs/{id}/files/raw?path=` | zawartość pliku — **ochrona z sekcji 8** |
| `GET` | `/api/runs/{id}/archive.zip` | strumieniowy zip katalogu |
| `GET` | `/api/leaderboard` | ranking; filtry + `?score=<wersja>` |
| `GET` | `/api/compare?runs=a,b,c` | wyrównane serie + tabela różnic |
| `GET` | `/api/events` | SSE: strumień zmian statusu |
| `GET` | `/api/admin/queue` | RabbitMQ, DLQ, SLURM, sieroty, budżet per user |

### 7.2 `GET /api/runs/{id}/series`

```
?x=gradient_count|database_reaches|epoch     (domyślnie gradient_count)
?metric=loss|accuracy                         (domyślnie loss)
?points=1000                                  (maks. 5000)
```

```json
{
  "task_id": "9f2c…4b1e",
  "x": "gradient_count",
  "metric": "loss",
  "points": [[0, 2.31], [128, 1.87], [256, 1.42]],
  "truncated": false,
  "downsample": "lttb",
  "original_points": 4096
}
```

**Downsampling: LTTB** (Largest-Triangle-Three-Buckets), nie próbkowanie co n-ty punkt.
Krzywa straty ma szpilki, które naiwne próbkowanie gubi — a szpilka w krzywej zbieżności to
informacja, nie szum. Cel ~1000 punktów. `original_points` jest w odpowiedzi celowo:
użytkownik ma widzieć, że patrzy na przybliżenie.

### 7.3 `GET /api/leaderboard`

```json
{
  "score_formula": {"id": "v1", "label": "mediana final_loss",
                    "note": "formuła tymczasowa — D2 nierozstrzygnięta"},
  "rows": [{
    "optimizer": "cma-es", "family": "gradient_free",
    "dataset": "wine", "model": "mlp-3x64", "suite": "test",
    "n_runs": 12,
    "final_loss":     {"median": 0.412, "q1": 0.388, "q3": 0.447},
    "final_accuracy": {"median": 0.871, "q1": 0.862, "q3": 0.880},
    "gradient_count":   {"median": 0},
    "database_reaches": {"median": 240000},
    "stop_reason_mode": "DATABASE_LIMIT",
    "score": 0.412
  }]
}
```

Trzy rzeczy celowe:

- **`score_formula` jest obiektem, nie liczbą w nagłówku.** D2 nierozstrzygnięta (§8), więc
  formuła jest wersjonowana, wybieralna przez `?score=` i **opisuje samą siebie**. Nagłówek
  kolumny w UI pokazuje, która formuła jest aktywna.
- **`n_runs` zawsze zwracane.** Mediana z `n=1` musi być w UI widocznie oznaczona — inaczej
  leaderboard kłamie. §13.1: *„pojedynczy przebieg to anegdota"*.
- **`q1`/`q3` obok mediany.** Rozrzut jest częścią wyniku.

### 7.4 Kody odpowiedzi

- `404` zamiast `403` dla cudzego runu — nie ujawniamy istnienia zasobu (8.4).
- `409` przy przekroczeniu limitu dziennego, z `Retry-After`.
- `422` przy odrzuceniu przez walidator, z pełnym logiem w ciele.

---

## 8. Filesystem view — specyfikacja bezpieczeństwa

**Jedyna bariera między przeglądarką a tajnymi datasetami** (§5.3, §12.4). Główny wymóg
bezpieczeństwa projektu.

### 8.1 Model zagrożenia

Atakujący to zalogowany, zweryfikowany uczestnik konkursu. Chce dostać datasety, bo ich
posiadanie niszczy sens konkursu (§5.3). Ma legalne konto i legalny `task_id`.

Wektory: path traversal, dowiązania symboliczne, kodowanie URL, TOCTOU, XSS, dostęp do cudzego
runu, wyciek przez nazwy plików w komunikatach błędu.

### 8.2 Kanonizacja ścieżki

Kolejność ma znaczenie. Kontrole tekstowe **nie są** autorytetem; autorytetem jest `realpath`.

```python
ARTIFACT_ROOT = Path("/downloads").resolve(strict=True)   # resolve TAKŻE korzenia

def resolve_artifact_path(task_id: UUID, rel: str) -> Path:
    # 1. odrzuty wstępne (tanie, nie są zabezpieczeniem samym w sobie)
    if "\0" in rel or rel.startswith("/") or len(rel) > 1024:
        raise Reject()

    run_root = (ARTIFACT_ROOT / str(task_id)).resolve(strict=True)
    # 2. run_root MUSI leżeć w ARTIFACT_ROOT
    if not run_root.is_relative_to(ARTIFACT_ROOT):
        raise Reject()

    # 3. kanonizacja celu — tu rozwijają się '..' i dowiązania
    target = (run_root / rel).resolve(strict=True)
    if not target.is_relative_to(run_root):
        raise Reject()
    return target
```

Trzy subtelności łatwe do pominięcia:

1. **`ARTIFACT_ROOT` też trzeba `resolve()`** — gdyby `/downloads` było dowiązaniem,
   porównanie prefiksu dałoby fałszywy wynik.
2. **`is_relative_to`, nie `str.startswith`.** `startswith` przepuszcza
   `/downloads/9f2c-evil` przy korzeniu `/downloads/9f2c`.
3. **Walidacja `task_id` parserem UUID przed dotknięciem FS.** UUID nie może zawierać `..`
   ani `/`, więc parsowanie jest samo w sobie zabezpieczeniem.

### 8.3 Otwarcie pliku — TOCTOU i dowiązania

Kanonizacja nie wystarcza: między sprawdzeniem a otwarciem downloader może dopisać
dowiązanie. To nie jest teoretyczne — **`scp -r` potrafi przenieść dowiązania symboliczne
z Ateny**, więc wrogie dowiązanie może trafić do katalogu artefaktów bez udziału atakującego
po stronie serwera.

```python
fd = os.open(target, os.O_RDONLY | os.O_NOFOLLOW)   # odmawia otwarcia dowiązania
st = os.fstat(fd)
if not stat.S_ISREG(st.st_mode):                    # tylko zwykłe pliki
    raise Reject()
if st.st_size > PREVIEW_LIMIT:                      # 2 MB, §12.4
    raise TooLarge()
```

`O_NOFOLLOW` + `fstat` na **deskryptorze** (nie na ścieżce) zamyka okno TOCTOU. Wersja
paranoiczna — przechodzenie komponentów przez `os.openat` z `O_NOFOLLOW` na każdym kroku —
jest dostępna, gdyby zespół uznał model zagrożenia za ostrzejszy.

Chodzenie po drzewie: `os.walk(..., followlinks=False)`, pomijanie wszystkiego, co nie jest
zwykłym plikiem lub katalogiem.

### 8.4 Autoryzacja

```python
def can_read_run(user, task) -> bool:
    if user and user.role == "admin":          return True
    if user and task.submitted_by == user.id:  return True
    return PUBLIC_RESULTS                       # D4 — jedna flaga, jedno miejsce
```

**D4 jest nierozstrzygnięta.** Dlatego polityka jest **jedną funkcją i jedną flagą**, a nie
warunkiem rozsianym po endpointach. Zespół zamyka D4 zmieniając jedną wartość.

Brak dostępu → **`404`, nie `403`**. `403` potwierdza, że run o takim UUID istnieje.

### 8.5 Serwowanie treści i renderowanie w React

**Nigdy nie renderujemy HTML ani SVG inline.** Whitelist po rozszerzeniu, nie po sniffingu:

| Rozszerzenie | `Content-Type` | Dyspozycja |
|---|---|---|
| `.png` | `image/png` | inline |
| `.csv` | `text/csv` | attachment (podgląd budujemy z danych) |
| `.json` | `application/json` | attachment (podgląd jako drzewo) |
| `.py`, `.out`, `.log`, `.txt` | `text/plain; charset=utf-8` | attachment (podgląd jako tekst) |
| **wszystko inne** | `application/octet-stream` | **attachment** |

Nagłówki na **każdej** odpowiedzi z `/files/raw`:

```
X-Content-Type-Options: nosniff
Content-Security-Policy: sandbox
Content-Disposition: attachment; filename="…"      (poza whitelistą inline)
Cache-Control: private, no-store
```

`Content-Security-Policy: sandbox` każe przeglądarce potraktować zasób jako **unikalny,
nieprzezroczysty origin**. Neutralizuje nawet błąd w whiteliście — jeśli kiedyś ktoś przez
pomyłkę wpuści `text/html`, skrypt i tak nie dostanie się do sesji na domenie głównej. Jeden
nagłówek, duża redukcja skutków pomyłki.

#### ⚠ Zasada dla frontendu w React

Renderowanie podglądu przenosi się z serwera do przeglądarki. To wymaga jednej twardej reguły:

> **Treść pliku nigdy nie trafia do `dangerouslySetInnerHTML`.** Bez wyjątków, bez „ale to
> tylko podświetlanie składni".

React escapuje wszystko wstawiane jako dziecko elementu, więc `{fileContent}` jest bezpieczne
z definicji. Konkretnie:

- **`.py`** → podświetlanie **tokenowe**: serwer (Pygments) zwraca listę `[typ, tekst]`,
  React renderuje `<span className={typ}>{tekst}</span>`. Nigdy gotowy HTML.
  Alternatywa: Shiki/highlight.js po stronie klienta na czystym tekście — też bez `innerHTML`.
- **`.csv`** → parser zwraca tablicę tablic, React buduje `<table>` z `{komórka}`.
- **`.json`** → `JSON.parse` + komponent drzewa. Nigdy `eval`.
- **`.out`** → `<pre>{tekst}</pre>` z wirtualizacją przy dużych plikach.
- **`.png`** → `<img src="/api/runs/:id/files/raw?path=…">`. Przeglądarka dostaje `image/png`
  + `nosniff`, więc nawet spreparowany plik nie wykona się jako skrypt.

**Reguła egzekwowana lintem:** ESLint `react/no-danger` jako `error`. Zakaz, którego nikt nie
egzekwuje, rozmywa się przy trzecim pull requeście.

**Kod optymalizatora jest tekstem. Nigdy nie jest importowany ani wykonywany.**

### 8.6 Pobranie katalogu jako `.zip`

Strumieniowo, **z ponownym zastosowaniem tych samych kontroli per wpis** — nie ufamy temu, że
skoro katalog przeszedł kontrolę, to każdy plik w nim też. Pomijamy dowiązania i pliki
specjalne. Limit sumaryczny + limit liczby wpisów.

### 8.7 Testy — obowiązkowe, nie opcjonalne

| # | Scenariusz | Oczekiwane |
|---|---|---|
| 1 | `path=../../etc/passwd` | 400/404, brak odczytu |
| 2 | `path=..%2f..%2fetc%2fpasswd` (URL-encoded) | 400/404 |
| 3 | `path=....//....//etc/passwd` | 400/404 |
| 4 | `path=/etc/passwd` (bezwzględna) | 400/404 |
| 5 | dowiązanie w katalogu runu → `/etc/passwd` | 400/404, `O_NOFOLLOW` odmawia |
| 6 | dowiązanie → katalog datasetów | 400/404 |
| 7 | `path` z bajtem zerowym | 400 |
| 8 | `task_id` = `../inny-run` | 400, parser UUID odrzuca |
| 9 | cudzy run, użytkownik `verified` | **404** (nie 403) |
| 10 | cudzy run, `admin` | 200 |
| 11 | plik 3 MB | brak podglądu, tylko pobranie |
| 12 | `evil.svg` | `application/octet-stream` + attachment |
| 13 | `evil.html` | `application/octet-stream` + attachment |
| 14 | każda odpowiedź `/files/raw` | ma `nosniff` i `CSP: sandbox` |
| 15 | zip z dowiązaniem w katalogu | dowiązanie pominięte |
| 16 | plik `.py` o treści `<script>alert(1)</script>` | renderowany jako tekst, brak wykonania |

Testy 5, 6 i 15 wymagają fixture'u tworzącego prawdziwe dowiązania — bez tego najgroźniejszy
wektor nie jest pokryty. Test 16 to regresja pod zasadę z 8.5 (dodany w rew. 2).

### 8.8 Funkcje interfejsu (§12.3)

Dwupanelowy układ: drzewo ~280 px + podgląd. Poniżej 900 px — stos z przyciskiem powrotu.

- **Deep link** `?path=` przez `useSearchParams` — odświeżenie wraca do tego samego pliku.
- **Breadcrumb** klikalny nad podglądem.
- **Stan rozwinięcia** drzewa w `localStorage`, klucz per `task_id`.
- **Metadane**: nazwa, rozmiar, typ, mtime, pobranie, kopiowanie ścieżki.
- **Renderery** wg 8.5.
- **Stany puste** z konkretnym komunikatem per `artifact_status`:
  `absent` + run trwa → „artefakty pojawią się po zakończeniu";
  `downloading` → „wyniki są pobierane z klastra";
  `empty` → dosłowny komunikat downloadera `No files found under …`.
- **Skróty klawiszowe** (obowiązkowe w zakresie rozszerzonym): `↑`/`↓` po drzewie, `Enter`
  otwiera, `←`/`→` zwija/rozwija katalog, `/` fokus na wyszukiwanie, `g`/`G` początek i koniec
  pliku w podglądzie `.out`, `y` kopiuje ścieżkę, `?` pokazuje ściągę. Wszystkie z widoczną
  listą pod `?` — skrót, o którym nikt nie wie, nie istnieje.

---

## 9. Wykresy, statystyka i `/compare`

### 9.1 Oś X to budżet

Domyślnie `gradient_count`, przełącznik na `database_reaches` i `epoch`. Czas zegarowy jest
świadomie deprecjonowany (§8) — **nie jest domyślną osią ani metryką rankingu**.

### 9.2 Mediana i wstęga IQR — poprawnie

Miejsce, w którym łatwo narysować wykres, który wygląda dobrze i wprowadza w błąd.

1. **Wspólna siatka budżetu.** Runy mają punkty w różnych miejscach osi X. Zbuduj siatkę
   ~200–1000 wartości (log-spaced przy zakresie wielorzędowym).
2. **Interpolacja schodkowa, nie liniowa.** Wartość straty przy budżecie *b* to **ostatnia
   zaobserwowana wartość przy budżecie ≤ *b***. Interpolacja liniowa **wymyśla** pomiary,
   których nie było — na wykresie do publikacji to błąd merytoryczny.
3. **Kwantyle po runach** w każdym punkcie siatki: p25 / p50 / p75.
4. **Obetnij wstęgę tam, gdzie kończą się dane.** ⚠ Runy kończą się przy różnych budżetach
   (różne `stop_reason`). Liczenie kwantyli po malejącej liczbie runów **sztucznie zwęża
   wstęgę po prawej stronie** i sugeruje zbieżność, której nie ma. Rysuj pełną wstęgę tylko
   dopóki **wszystkie** runy mają dane; dalej linią przerywaną, z `n` w tooltipie.
5. **Downsampling dopiero teraz.** Agreguj, potem downsampluj — nie odwrotnie. Downsampling
   przed agregacją zmienia wartości kwantyli.

Punkt 4 jest najważniejszy — bez niego wykres systematycznie kłamie na korzyść metod, które
kończą wcześnie.

**Gdzie to liczyć:** po stronie serwera, w `/api/compare`. Nie w przeglądarce — bo ta sama
agregacja zasila eksport CSV i musi dać identyczne liczby.

### 9.3 `/compare` — pełne narzędzie analityczne

- Wybór N runów; zestaw zakodowany w URL (`?runs=a,b,c`) → link jest permalinkiem.
- Nałożone krzywe, mediana + wstęga IQR per grupa (optymalizator × dataset × model).
- Przełącznik osi X: gradienty / próbki / epoki.
- Skala logarytmiczna na osi straty (§13.2 — „przy porównywaniu rzędów wielkości konieczność").
- Zoom, wyłączanie serii kliknięciem w legendę, tooltip z dokładną wartością i `n`.
- **Tabela różnic**: dla każdej pary — różnica mediany final_loss, przedział, `n` obu grup.
  ⚠ **Bez p-wartości.** D3 (jaki test, ile powtórzeń) nierozstrzygnięta, a dorzucenie testu
  bez poprawki na wielokrotne porównania byłoby błędem, który akurat ta publiczność wychwyci
  natychmiast. Schemat (`seed` per run) go **umożliwia** — implementacja czeka na decyzję.
- **Eksport**: wykres → PNG (canvas uPlota) i SVG (generowany serwerowo z tych samych danych),
  dane → CSV. CSV zawiera **dane po agregacji wraz z `n`**, żeby wynik dało się odtworzyć.

### 9.4 uPlot w React

uPlot jest biblioteką imperatywną — wrapper to ~40 linii i jest to celowy wybór, nie
niedoróbka:

```tsx
function Plot({data, opts}: {data: uPlot.AlignedData; opts: uPlot.Options}) {
  const ref = useRef<HTMLDivElement>(null);
  const plot = useRef<uPlot | null>(null);
  useEffect(() => {
    plot.current = new uPlot(opts, data, ref.current!);
    return () => plot.current?.destroy();
  }, [opts]);                       // opts memoizowane u rodzica
  useEffect(() => { plot.current?.setData(data); }, [data]);
  return <div ref={ref} />;
}
```

**Dlaczego uPlot, a nie Recharts/visx:** kilkanaście serii × tysiące punktów. Recharts
renderuje każdy punkt jako element SVG w drzewie React — przy tej gęstości zamula.
uPlot rysuje na canvasie i ma **natywne wsparcie dla `bands`**, czyli dokładnie wstęgi IQR
z 9.2. To jest ten rzadki przypadek, gdzie biblioteka pasuje do zadania 1:1.

### 9.5 Paleta wykresów

`_DARK_BG_PALETTE` (20 neonów na czarnym tle) **odpada** [wg briefu §13.2].

**Okabe–Ito**, bezpieczna dla daltonizmu:

```
#000000  #E69F00  #56B4E9  #009E73  #0072B2  #D55E00  #CC79A7
```

Kanoniczny Okabe–Ito ma jeszcze `#F0E442` (żółty) — **pomijam go na jasnym tle**, kontrast
niewystarczający. W trybie ciemnym `#000000` → `#EDF1F5`.

**Serie rozróżniane także stylem linii** (ciągła / kreskowana / kropkowana / kreska-kropka) —
w druku czarno-białym kolor znika, a te wykresy trafią do artykułów.

---

## 10. Kolejka — dlaczego outbox, a nie `aio-pika`

Brief ostrzega (§15): *„`pika` jest blokująca i nie jest async-safe"*. To prawda, ale
**`aio-pika` rozwiązuje tylko połowę problemu**.

Druga połowa: zapis do `tasks` i publikacja na RabbitMQ to dwie operacje bez wspólnej
transakcji. Padnie broker między nimi — zgłoszenie jest w bazie, ale nigdy nie poleci. Padnie
po publikacji, przed commitem — poleci zadanie, którego nie ma w bazie.

**Rozwiązanie: transakcyjny outbox.**

```
BEGIN;
  INSERT INTO submissions …;
  INSERT INTO tasks …;
  INSERT INTO queue_outbox (exchange, routing_key, payload) VALUES …;
COMMIT;
```

Osobny, mały proces drenuje `queue_outbox` i publikuje zwykłą, blokującą `pika` — **poza
event loopem, więc problem async-safety znika sam**.

Zyski:
- Awaria RabbitMQ nie zwraca 500 użytkownikowi i nie gubi zgłoszenia.
- Endpoint nie dotyka brokera → UI nie potrzebuje poświadczeń do kolejki (cel z sekcji 1).
- `at-least-once` z jawnym `attempts` i `last_error` zamiast cichej utraty.

Koszt: opóźnienie publikacji rzędu sekundy. Przy 1–3 zgłoszeniach na użytkownika dziennie
(§5.2) nieistotne.

**Do ryzyka idempotencji z §18** (worker pada po `sbatch`, przed zapisem job ID): `task_id`
jako nazwa joba SLURM + sprawdzenie `squeue --name=<task_id>` przed submitem. To moduł
Bartka — **do uzgodnienia, nie do samodzielnej zmiany** (§4).

---

## 11. Status na żywo

```
poller → UPDATE tasks → trigger → pg_notify('task_changed', {ids})
                                        │
                        psycopg3 async LISTEN (dedykowane połączenie)
                                        │
                            asyncio broadcast → SSE → EventSource
```

Szczegóły decydujące o tym, czy to działa w produkcji:

- **Dedykowane połączenie** do `LISTEN`, poza pulą. Połączenie z puli zostanie oddane i nasłuch
  zniknie.
- **Heartbeat co ~15 s** (komentarz SSE `: ping`). Bez tego pośredniki zamykają bezczynne
  połączenie.
- **`X-Accel-Buffering: no`**, jeśli przed aplikacją stoi nginx — inaczej buforuje strumień.
- **Filtrowanie po stronie serwera:** użytkownik dostaje zdarzenia tylko dla runów, które
  wolno mu widzieć. Ta sama funkcja `can_read_run` co w 8.4 — jedna polityka, nie dwie.
- Przy wielu workerach uvicorna każdy ma własny `LISTEN`; `pg_notify` rozgłasza do wszystkich.
- **Po stronie React:** `EventSource` w jednym hooku na całą aplikację, zdarzenie unieważnia
  cache konkretnego runu i wyzwala refetch. Nie otwieramy połączenia per komponent.

### 11.1 Stany i komunikaty (§11.3)

| Stan | Komunikat | Dodatkowo |
|---|---|---|
| w kolejce RabbitMQ | „w kolejce systemu" | pozycja w kolejce |
| w kolejce SLURM | „w kolejce na Atenie" | ID zadania SLURM + czas oczekiwania |
| liczy się | „liczy się na Atenie" | czas trwania + limit `--time` |
| zakończone, artefakty się ściągają | „pobieranie wyników z klastra" | pasek, bez artefaktów |
| zakończone | „zakończone" | komplet |
| nieudane | „nieudane" | `error_message` + ogon `.out` **rozwinięty, nie za kliknięciem** |
| nieudane bez artefaktów | „nieudane — brak artefaktów" | dosłowny komunikat downloadera |
| odrzucone przez walidator | „odrzucone przy walidacji" | pełny log walidatora |

---

## 12. System wizualny

Rejestr: Papers With Code / OpenReview / Distill.pub / dokumentacja scikit-learn.
**Nie SaaS landing.** Wrażenie ma robić kompletność i precyzja, nie efekty.

### 12.1 Zakazy (§14.2) — egzekwowane maszynowo

Zero gradientów (tła, przyciski, nagłówki, karty). Zero glassmorphism, neonów, świecenia.
Zero ciężkich cieni — separacja **linią 1 px**. Zero emoji jako ikon sekcji. Zero animacji
dekoracyjnych. Zero marketingowego copy.

**Dlatego CSS Modules, a nie Tailwind.** Przy Tailwindzie zakazy są rozsiane po klasach
w JSX i nie da się ich sprawdzić `grep`em. Przy CSS Modules cały styl jest w plikach `.css`,
więc CI może to egzekwować:

```bash
# fail build, jeśli ktoś doda gradient albo ciężki cień
! grep -rIEn 'linear-gradient|radial-gradient|backdrop-filter|box-shadow:[^;]*blur\([2-9][0-9]' \
    frontend/src --include='*.css'
```

Do tego ESLint: `react/no-danger: error` (zasada z 8.5).

Zakaz, którego nikt nie egzekwuje, rozmywa się przy trzecim pull requeście.

### 12.2 Tokeny

Tryb ciemny obowiązkowy (§14.3). **Nigdy nie definiuj koloru wyłącznie wewnątrz media
query** — pełna paleta na `:root`, nadpisania w dwóch miejscach.

```css
:root {
  --bg:        #FCFCFD;
  --surface:   #F3F5F7;
  --border:    #D9DEE4;
  --text:      #11161C;
  --text-2:    #5F6B79;
  --accent:    #1D4E89;
  --success:   #2C6E49;
  --error:     #9C4221;
}

/* domyślne „system" — tylko preferencja, bez atrybutu */
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg:      #0E1116;
    --surface: #161B22;
    --border:  #2A313A;
    --text:    #E6EAF0;
    --text-2:  #9AA6B2;
    --accent:  #6BA3E5;
    --success: #5FB37E;
    --error:   #E08A5F;
  }
}

/* jawny wybór wygrywa w obie strony */
:root[data-theme="dark"] { /* te same wartości co wyżej */ }

body { background: var(--bg); color: var(--text); }
```

Kolory trybu ciemnego to **rozjaśnione warianty** palety z §14.3, dobrane pod kontrast ≥ 7:1
dla tekstu głównego i ≥ 4.5:1 dla wtórnego. Akcent `#1D4E89` na ciemnym tle ma za mały
kontrast — stąd `#6BA3E5`.

**Przełącznik motywu:** ustawia `data-theme` na `<html>` i zapisuje wybór w `localStorage`.
Trzy stany: `system` (brak atrybutu) / `light` / `dark`. Skrypt ustawiający atrybut musi być
**inline w `index.html`, przed załadowaniem bundla** — inaczej użytkownik zobaczy mignięcie
jasnego motywu przed przełączeniem (FOUC). To jest konkretny minus SPA, który trzeba obsłużyć
ręcznie.

### 12.3 Typografia (§14.4)

Fonty **wyłącznie systemowe** — zero pobrań, zero CDN, zgodność z restrykcyjnym CSP.

```css
--font-serif: Charter, "Bitstream Charter", "Sitka Text", Cambria, Georgia, serif;
--font-sans:  system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
--font-mono:  ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas,
              "Liberation Mono", monospace;
```

- **Nagłówki** — szeryfowe, z umiarem (tytuły stron i sekcji).
- **Tekst** — bezszeryfowy systemowy, miara 65–75 znaków.
- **Wszystkie dane** — monospace: ID, ścieżki, liczby w tabelach, nazwy optymalizatorów,
  UUID, ID zadania SLURM.

```css
.num, td.num, .mono { font-family: var(--font-mono); font-variant-numeric: tabular-nums; }
```

`tabular-nums` **obowiązkowo** wszędzie, gdzie cyfry ustawiają się w kolumnie.

### 12.4 Układ (§14.5)

Gęstość danych ponad przestronność — 40 wierszy tabeli bije 8 kart. Podsumowanie przed
szczegółem. Stan kodowany **formą, nie samym kolorem** (pigułka z tekstem). Szerokie treści
przewijają się we własnym kontenerze — **strona nigdy nie przewija się w poziomie**. Liczby
zawsze z jednostką i kontekstem: `12 480 gradientów / limit 100 000`.

### 12.5 Nazewnictwo (§14.6)

| W kodzie | W interfejsie |
|---|---|
| `database_reaches` | **przetworzone próbki** (z dymkiem wyjaśniającym) |
| `executor_task_id` | **ID zadania SLURM** |
| `task_status = running` | **liczy się na Atenie** |
| `gradient_count` | **wyliczone gradienty** |
| `stop_reason = DATABASE_LIMIT` | **wyczerpany limit próbek** |
| `stop_reason = OPTIMIZER_CONVERGED` | **optymalizator zgłosił zbieżność** |

Ostatnie dwa są istotne merytorycznie: „zbiegłem" i „wyczerpałem budżet" to zupełnie różne
wyniki (§8) i interfejs nie może ich mieszać.

---

## 13. `/docs` — strona protokołu

Nie jest stroną pomocniczą. Dla uczestnika GECCO to **jedyna instrukcja**, a dla profesora na
demo — dowód, że system ma przemyślany kontrakt.

- Kontrakt `step(evaluator) -> bool` z wyjaśnieniem, że `True` = zbiegłem.
- Pełna tabela API ewaluatora z **kolumną „efekt na licznikach budżetu"** — informacja,
  której uczestnik potrzebuje najbardziej i której nie ma nigdzie indziej.
- **Dwa kompletne, kopiowalne przykłady**: gradientowy i ewolucyjny. Nie fragmenty — pliki,
  które działają po wklejeniu. Przycisk „kopiuj".
- Zasady budżetu i warunki stopu (`StopCondition`, `StopReason`).
- Szablon `optimizer_template.py` do pobrania.
- Opis warstwy DTO (`get_output_type()`) i kiedy wybrać CuPy zamiast NumPy.
- Opis sandboxa walidatora — co jest sprawdzane i dlaczego zgłoszenie może zostać odrzucone.

**Znane problemy kontraktu do udokumentowania jawnie** (§7 briefu) — uczciwość wobec
uczestników jest wartością, nie ujmą:

1. `step()` nie wie, ile budżetu zostało. Standardowe suity (CEC, COCO) wystawiają
   `remaining_budget`. Brak tej informacji uniemożliwia chłodzenie sigmy.
2. Brak `evaluate_population(list) -> list[float]` — metody populacyjne płacą podatek
   λ sekwencyjnych forward passów zamiast jednego wsadowego.
3. Losowość nie jest kontrolowana (globalny `np.random` bez ziarna).
4. `get_output_type()` bez `self` i bez `@classmethod` — wywołanie na instancji się wysypie.

To **defekty modułu prowadzącego**, nie do naprawy z tej strony (§4). Do zgłoszenia jako issue.

---

## 14. Kolejność pracy

Ułożona tak, żeby na każdym etapie system działał, zgodnie z §19 briefu.

| # | Krok | Efekt |
|---|---|---|
| 0 | Ten dokument jako pierwszy commit | Decyzje zapisane przed kodem |
| 1 | Migracje: `submissions`, `results`, `result_series`, transitions, outbox, notify | Jest z czego zbudować cokolwiek |
| 2 | Szkielet FastAPI + reużycie `auth/repository.py` + sesje + Bearer | Control plane stoi |
| 3 | `/api/runs`, `/api/runs/{id}`, `/api/leaderboard` + seed realistycznych danych | API weryfikowalne bez frontendu |
| 4 | Szkielet Vite + React Router + typy z OpenAPI + Dockerfile wieloetapowy | Deep linki działają, jeden kontener |
| 5 | System wizualny: tokeny, typografia, `<DataTable>`, `<StatusPill>`, lint CSS | Reszta stron jest tania |
| 6 | `/runs`, `/runs/:id` | Pierwsza realna strona |
| 7 | **Filesystem view + 16 testów bezpieczeństwa** | Funkcja, która sprzedaje projekt |
| 8 | `/api/runs/{id}/series` + LTTB + wrapper uPlot | Wykresy |
| 9 | `/leaderboard` na prawdziwych danych, `mock_data.py` usunięte | Koniec atrapy |
| 10 | Walidator z `origin/optim_validation` + `/submit` z synchronicznym logiem | Pełna ścieżka zgłoszenia |
| 11 | Outbox publisher | UI bez poświadczeń do kolejki |
| 12 | SSE + `LISTEN` + stany z §11.3 | Status na żywo |
| 13 | `/compare` — pełne narzędzie | Wartość analityczna |
| 14 | `/admin`, `/admin/queue` | Operacyjność |
| 15 | `/docs`, `/` | Warstwa dla zwiedzającego |
| 16 | Frontend w `docker-compose.yml` + healthchecki | „Cały system z jednego compose" |
| 17 | Naprawa CI (15.1) + flake8 max-line 120 + pytest + build frontu w CI | Zielony pipeline |

Kroki 1–7 mają **samodzielną wartość** nawet gdyby reszta nie powstała — dają model danych,
control plane i jedyną funkcję, która na demo dowodzi, że system naprawdę policzył.

Krok 4 jest celowo wcześnie: postawienie szkieletu SPA z routingiem i wieloetapowym
Dockerfile'em **przed** pisaniem stron weryfikuje najbardziej ryzykowne założenie tej
architektury (catch-all, deep linki, jeden kontener) w momencie, w którym zmiana kursu jest
jeszcze tania.

### 14.1 Seed danych testowych

Bez tego nie da się zweryfikować niczego wizualnie. Potrzebne:

- ≥ 4 optymalizatory × ≥ 2 rodziny × ≥ 8 ziaren — żeby mediana i IQR miały sens.
- Runy w **każdym** stanie z §11.3, łącznie z „nieudany bez artefaktów" i „odrzucony przez
  walidator" — stany błędne są najczęściej nieprzetestowane.
- Katalog `/downloads/<task_id>/` z PNG-ami, CSV, `.out`, `metadata.json`, `optimizer.py`.
- **Fixture z dowiązaniem symbolicznym** do testów bezpieczeństwa (8.7).
- **Plik `.py` z treścią `<script>` i plik `evil.svg`** — pod testy 12, 13 i 16.

---

## 15. Defekty do naprawy i do zgłoszenia

### 15.1 CI — `uv sync --extra ci` [wg briefu §6, NIEZWERYFIKOWANE]

Brief twierdzi, że `.github/workflows/ci.yml` woła `uv sync --extra ci`, a root
`pyproject.toml` nie definiuje `[project.optional-dependencies].ci`, przez co oba joby padają.

**Nie zweryfikowałem tego** — nie miałem repo. Przy pierwszym kontakcie sprawdzić i naprawić:

```toml
# wariant A — jeśli workflow zostaje przy --extra
[project.optional-dependencies]
ci = ["flake8>=7", "pytest>=8", "pytest-cov"]
```
```toml
# wariant B — nowocześniejszy, wymaga zmiany na `uv sync --group ci`
[dependency-groups]
ci = ["flake8>=7", "pytest>=8", "pytest-cov"]
```

Wariant B jest właściwszy semantycznie (to zależności deweloperskie, nie opcjonalna funkcja
pakietu), ale wymaga zmiany workflow.

**Dodatkowo w rew. 2:** CI musi zyskać drugi job — `npm ci && npm run build && npm test`
dla frontendu, plus lint CSS z 12.1 i ESLint. Bez tego zakazy z §14 nie są egzekwowane,
a rozjazd typów z OpenAPI wyjdzie dopiero na demo.

### 15.2 Niespójne importy [wg briefu §9, NIEZWERYFIKOWANE]

Część modułów używa `from auth import ...`, część `from frontend.core import ...`. Przy
`WORKDIR /app` druga forma może się nie rozwiązać. Do ujednolicenia przy migracji — naturalny
moment, bo i tak przenosimy warstwę dostępu do danych.

### 15.3 Brak frontendu w `docker-compose.yml` [wg briefu §9]

Compose dostarcza tylko Postgresa i RabbitMQ. Cel zespołu to „cały system z jednego docker
compose". Dołożyć serwis `web` z `depends_on: {condition: service_healthy}` i `/downloads`
zamontowanym **`:ro`**.

### 15.4 Migracje nie wykonują się na istniejącym wolumenie

`docker-entrypoint-initdb.d` działa tylko przy pustym wolumenie. Patrz sekcja 6.

### 15.5 ⚠ Sekrety w prompcie zadania — do rotacji

Plik promptu tego zadania zawiera **w jawnym tekście**: `client_secret` Google OAuth,
`client_secret` Microsoft OAuth, `secret_key` reCAPTCHA oraz `cookie_secret`. Te wartości są
zapisane na dysku maszyny `kuba` i w transkrypcie sesji.

**Rekomendacja: rotacja wszystkich czterech**, niezależnie od losów tego zadania. Niezależnie
od tego `.gitignore` musi zawierać `frontend/env`, `**/secrets.toml`, `.streamlit/secrets.toml`,
a przed każdym commitem trzeba sprawdzać `git diff --cached --name-only`.

### 15.6 Defekty kontraktu optymalizatora

Cztery pozycje z sekcji 13 — moduł prowadzącego, do zgłoszenia jako issues, nie do
samodzielnej naprawy (§4: „wejście w cudzy moduł wymaga uzgodnienia").

---

## 16. Czego świadomie nie robimy

| Nie robimy | Dlaczego |
|---|---|
| Wgrywanie własnych datasetów i sieci | §16 — „piękna wizja prof. Arabasa", jawnie krok dalej |
| System kredytów | §16 — na start limit dzienny |
| CI/CD z automatycznym deployem | §16 — prowadzący uważa za overkill |
| Kubernetes, wielowęzłowość | §5.1 — jedna VM, cztery kontenery |
| Drugi backend wykonawczy (lokalny CPU) | §16 — wartościowy, ale nie na PoC |
| SSR / Next.js | 2.3 — piąty kontener; do dołożenia później bez zmian w backendzie |
| Backend w Go | 2.5 — koszt natychmiastowy, zysk niemierzalny przy ~100 użytkownikach |
| **Test statystyczny na `/compare`** | D3 nierozstrzygnięta; schemat go umożliwia |
| **Zaszyta formuła rankingu** | D2 nierozstrzygnięta; agregat wymienialny z założenia |
| **Zmiany w module kolejki / walidatora / ewaluatora** | §4 — cudze moduły, tylko integracja |

---

## 17. Otwarte decyzje — status

| # | Decyzja | Status |
|---|---|---|
| D1 | Streamlit zostaje czy odchodzi | **ZAMKNIĘTA** — odchodzi; front React + Vite, backend FastAPI (sekcja 2, stack ustalony przez właściciela) |
| D2 | Wspólna waluta budżetu Adam vs CMA-ES | Otwarta — `score_formula` wersjonowana, architektura nie przesądza |
| D3 | Ile powtórzeń i jaki test statystyczny | Otwarta — `seed` w schemacie umożliwia dowolną decyzję |
| D4 | Czy wyniki publiczne | Otwarta — jedna flaga `PUBLIC_RESULTS` w `can_read_run` |
| D5 | Podział repozytoriów | Otwarta — poza zakresem warstwy webowej |
| D6 | Wyniki w Postgresie czy pliki | **Rozstrzygnięta faktycznie**: skalary i szeregi do Postgresa, pliki zostają jako artefakty. Bez tego leaderboard musiałby parsować CSV w locie |
| D7 | Model limitów | Otwarta — API zwraca `409` + `Retry-After`, UI pokazuje pozostały limit; sam model do ustalenia |
| D8 | Kiedy przywrócić runner na `main` | Otwarta — blokuje reprodukowalny seed danych |

**Mandat obejmuje D1.** D2, D3 i D4 są jawnie delegowane do doktorów z PW (§8: *„jesteśmy
tylko głupimi studentami […] potrzebujemy czegoś, co będzie faktycznie twarde i naukowo
sensowne"*). Architektura jest zbudowana tak, żeby każdą z nich zamknąć **bez migracji
schematu i bez zmiany kontraktu API**.

---

## 18. Ryzyka tego planu

| Ryzyko | Skutek | Mitygacja |
|---|---|---|
| **React w zespole Pythona, horyzont 5–10 lat** (§2, §4) | Moduł-sierota po odejściu autora | Sekcja 2.4: zero zbędnych zależności, Node tylko w buildzie, `package-lock` w repo, typy z OpenAPI |
| Zakres rozszerzony (9 stron + pełne `/compare` + `/admin/queue`) vs PoC 31.08 | Nie wszystko zdąży | Kolejność z sekcji 14 — kroki 1–7 mają wartość samodzielną |
| Brak SSR — pusty ekran przy pierwszym wejściu z linku | Gorsze pierwsze wrażenie na demo | Mały bundle, inline skrypt motywu przeciw FOUC (12.2); Next.js dołożalny później bez zmian w backendzie |
| `/admin/queue` wymaga danych z RabbitMQ i SLURM | Wejście w moduł Bartka i Adama (§4) | Tylko odczyt (management API RabbitMQ, `sinfo`/`sacct` przez istniejący poller); żadnych zmian w ich kodzie bez uzgodnienia |
| Wyciek datasetów przez przeglądarkę plików | Zniszczenie sensu konkursu | Sekcja 8 w całości + 16 testów, w tym dowiązania i XSS |
| XSS przez treść pliku w React | Skrypt na własnej domenie | Zasada z 8.5 + `react/no-danger: error` + `CSP: sandbox` na `/files/raw` |
| Wykres median/IQR narysowany naiwnie | Wprowadza w błąd publiczność naukową | Sekcja 9.2, zwłaszcza punkt 4 (obcinanie wstęgi) |
| Reprodukowalność na A100 (cuDNN) | „Dlaczego mój wynik się nie powtórzył" na leaderboardzie | `seed` + `runner_version` + `gpu_model` w `tasks`; rozrzut z N ziaren |

---

## 19. Definicja ukończenia

Warstwa webowa jest gotowa, gdy:

1. `docker compose up` podnosi **cały** system łącznie z frontendem, z **jednego** obrazu
   `web` (build wieloetapowy).
2. Wszystkie 9 stron z sekcji 3 działa na zasianych, realistycznych danych.
3. 16 testów bezpieczeństwa z 8.7 przechodzi, **w tym testy dowiązań i test XSS**.
4. Testy kontraktu API i downsamplingu przechodzą.
5. `flake8 --max-line-length 120`, `pytest`, `npm run build` i ESLint przechodzą **w CI**,
   nie tylko lokalnie.
6. **Każdy permalink z sekcji 3 działa po odświeżeniu w nowej karcie** — to jest test
   catch-all routingu SPA i najczęstsza usterka tej architektury.
7. Tryb ciemny działa w obu kierunkach (preferencja systemowa i jawny przełącznik), bez
   mignięcia jasnego motywu przy ładowaniu.
8. Lint CSS z 12.1 nie znajduje gradientów ani ciężkich cieni.
9. Screenshoty `/`, `/leaderboard`, `/runs/:id`, filesystem view, `/submit`, `/compare`
   załączone do raportu.

Punkty 5 i 6 są wymienione osobno celowo: „przechodzi lokalnie" nie jest dowodem, dopóki CI
jest zepsute (15.1), a deep link, który działa tylko przy nawigacji wewnątrz SPA, nie spełnia
§11.2.

---

*Dokument opracowany na podstawie `BRIEF.md` z 17.08.2026. Rew. 2: stack frontendu ustalony
przez właściciela (React). Twierdzenia o stanie repozytorium oznaczone `[wg briefu]` wymagają
weryfikacji przy pierwszym kontakcie z kodem.*
