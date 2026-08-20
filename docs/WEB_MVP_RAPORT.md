# Warstwa webowa — raport z wykonania

**Projekt:** Benchmark Czarnej Skrzynki (`projekt-benchmark`, Politechnika Warszawska)
**Gałąź:** `feat/web-mvp` · **Data:** 20.08.2026
**Podstawa:** `BRIEF.md`, `docs/WEB_MVP_PLAN.md`, `docs/WERYFIKACJA_REPO.md`

Ten dokument opisuje, co powstało, czym to zostało udowodnione, czego świadomie
nie zrobiono i co zostaje do rozstrzygnięcia przez zespół. Sekcja 5 prostuje dwa
ustalenia poprzednich dokumentów, które okazały się nieaktualne po zetknięciu
z uruchomionym kodem.

---

## 1. GOAL i decyzja D1

### GOAL

> Zbudować control plane i warstwę webową, które (a) przetrwają jako fundament
> konkursu GECCO 2027, (b) na demo dla Zakładu SI EIT dowodzą, że system
> **naprawdę policzył**, i (c) dadzą się utrzymać przez kolejne roczniki
> studentów.

Przejęty z planu bez zmian. Kolejność priorytetów też: control plane ma wartość
nawet gdyby interfejs nigdy nie powstał, bo bez niego konkurs nie ma jak
przyjmować zgłoszeń programowo.

### D1 — Streamlit odchodzi, front w React

**Decyzja podtrzymana**, ale nie dlatego, że tak było w planie. Oceniłem ją
krytycznie i utrzymuję z jednego twardego powodu:

> §11.2: *„Zasada projektowa: każdy zasób ma własny URL. Uczestnik konkursu chce
> wkleić link do swojego wyniku w mailu albo w artykule."*

`st.query_params` koduje stan w query stringu, ale
`/runs/<id>/files?path=reports/loss_vs_grads.png` jako adresowalny, dzielony
zasób — nie. To jest wymóg produktowy, nie estetyczny, i sam zamyka sprawę.
Drugi, niezależny argument: §15 wymaga control plane, bo GECCO potrzebuje
zgłoszeń z CLI i CI, a nie tylko z formularza.

**Czego nie podtrzymałem z planu.** Plan rekomendował uPlot do wykresów.
Zrezygnowałem. Wstęga IQR, interpolacja schodkowa i reguła obcinania wstęgi tam,
gdzie kończą się przebiegi, są na tyle specyficzne, że bibliotekę i tak trzeba by
do nich naginać — a eksport SVG, którego wymaga §9.3, jest darmowy, kiedy to, co
widać na ekranie, **już jest** SVG. Wykresy są napisane ręcznie, w ~470 liniach.
Konsekwencja jest zgodna z celem 2.4 planu („zero zbędnych zależności"):

**Zależności runtime frontendu to React, React DOM i router. Nic więcej.**
Bundle: 320 kB, 97 kB po gzipie.

### Stack

| Warstwa | Wybór | Uzasadnienie |
|---|---|---|
| Control plane | FastAPI + psycopg 3 (async) | §15; psycopg 3 już w projekcie |
| Frontend | React 19 + TypeScript + Vite | §11.2 — każdy zasób ma URL |
| Serwowanie | `vite build` → statyki z FastAPI | Node tylko w budowaniu obrazu |
| Wykresy | własny komponent SVG | zero zależności, darmowy eksport SVG |
| Status na żywo | `pg_notify` → `LISTEN` → SSE | §15 |
| Kolejka | transakcyjny outbox + osobny publisher | patrz niżej |
| Migracje | numerowane `.sql` + `schema_migrations` | `initdb.d` działa tylko na pustym wolumenie |
| Auth | **reużyte `src/frontend/auth`** | §19.2 |

**Vite zamiast frameworka renderującego serwerowo.** Node jest potrzebny do
zbudowania obrazu i nie jest potrzebny do jego uruchomienia. Deployment zostaje
przy czterech kontenerach z §5.1 zamiast zyskiwać piąty runtime do łatania, a
CORS i ciasteczka cross-origin w ogóle nie wchodzą w grę. Koszt jest realny i
nie ukrywam go: **brak renderowania serwerowego**, więc link otwarty na zimno
przez moment pokazuje pustą stronę, a publiczny ranking nie będzie indeksowany.
Gdyby to zaczęło przeszkadzać, warstwę renderującą da się dołożyć przed tym
samym API bez ruszania backendu.

**Outbox zamiast `aio-pika`.** Brief ostrzega, że `pika` jest blokująca i nie
jest async-safe. To prawda, ale rozwiązuje połowę problemu. Druga połowa: zapis
zadania i publikacja na kolejkę to dwie operacje bez wspólnej transakcji. Padnie
broker między nimi — zgłoszenie jest w bazie i nigdy nie poleci. Padnie po
publikacji, przed commitem — poleci zadanie, którego nie ma w bazie. Wiadomość
wstawiana w tej samej transakcji co zadanie usuwa oba przypadki, a osobny proces
publikuje zwykłą, blokującą `pika` **poza event loopem**, więc pytanie o
async-safety w ogóle nie powstaje. Przy okazji API nie ma żadnych poświadczeń do
brokera, co było postulatem §9.

---

## 2. Co działa i czym to udowodniono

System stoi z jednego `docker compose up`. Poniżej to, co zostało faktycznie
uruchomione i sprawdzone, a nie zadeklarowane.

### 2.1 Dane są mierzone, nie zmyślone

To jest najważniejsze zdanie tego raportu. Wykres narysowany z fabrykowanych
liczb można doprowadzić do dowolnego kształtu; sens demonstracji polega na tym,
że nie można.

`tools/local_backend/` uruchamia **prawdziwy `ModelEvaluator`** i **prawdziwe
optymalizatory NumPy z repozytorium** na publicznych zbiorach ze scikit-learn.
Liczniki gradientów i próbek inkrementuje ewaluator projektu, na jego własnych
zasadach — dokładnie tak, jak robiłby to na klastrze.

Zasiane: **194 policzone przebiegi**, 11 optymalizatorów, obie rodziny metod,
3 zbiory danych, 3 architektury, 8 ziaren na konfigurację, plus **wszystkie
8 stanów z §11.3**, łącznie z tymi, które nie mają wyników: w kolejce brokera,
w kolejce SLURM, liczy się, artefakty się ściągają, nieudane z logiem, nieudane
bez artefaktów, odrzucone przez walidator.

Zbiory danych są celowo tymi zabawkowymi ze scikit-learn i **nigdy** zbiorami
projektu — te są trzymane poza repozytorium dokładnie po to, żeby nikt nie mógł
uruchomić benchmarku prywatnie (§5.3).

### 2.2 Teza projektu, w liczbach

Na `wine` / `mlp-1x16`, 12 epok, mediana z 8 ziaren:

| Optymalizator | Rodzina | Strata | Gradienty | Próbki |
|---|---|---|---|---|
| `adam` | gradientowa | 0.0199 | 72 | 2 136 |
| `cma-es` | bezgradientowa | 0.0297 | **0** | **42 720** |

To jest **D2 postawiona wprost**: CMA-ES osiąga porównywalny wynik nie zużywając
ani jednego gradientu, ale kosztem dwudziestokrotnie większej liczby próbek. Bez
kursu wymiany między tymi walutami nie da się powiedzieć, który wygrał — i
właśnie dlatego żadna z formuł rankingu w tym systemie nie miesza tych dwóch
kolumn.

### 2.3 Bezpieczeństwo przeglądarki plików

To jedyna bariera między przeglądarką zalogowanego uczestnika a zbiorami danymi.
Model zagrożenia nie jest anonimowym skanerem, tylko legalnym uczestnikiem z
własnym `task_id`.

**69 testów przechodzi**, w tym:

- path traversal w 7 wariantach (`..`, kodowanie URL, `....//`, ścieżka
  bezwzględna, po poprawnym prefiksie, separatory Windows),
- **dowiązanie symboliczne wyprowadzające poza katalog runu**,
- **dowiązanie do katalogu datasetów**,
- **dowiązanie podstawione *po* kontroli, a przed odczytem** — to jest test
  TOCTOU i to on uzasadnia `O_NOFOLLOW` z `fstat` na deskryptorze,
- nakładający się prefiks katalogu (`<uuid>` kontra `<uuid>-other`) — przypadek,
  na którym `startswith` daje zły wynik,
- bajt zerowy, ścieżka ponad limit, `task_id` niebędący UUID,
- `evil.svg` i `evil.html` serwowane jako `application/octet-stream` z
  `attachment`,
- plik `.py` o treści `<script>alert(1)</script>` renderowany jako tekst,
- wstrzyknięcie nagłówka przez nazwę pliku,
- limit podglądu 2 MB.

Do tego `react/no-danger: error` w ESLint, żeby zasada „treść pliku nigdy nie
trafia do `dangerouslySetInnerHTML`" nie rozmyła się przy trzecim pull requeście.

### 2.4 Deep linki

Każdy adres z §11.2 działa po odświeżeniu w nowej karcie — jest to pokryte
testem parametryzowanym po 12 ścieżkach, bo brak catch-all po stronie serwera to
najczęstsza usterka tej architektury i objawia się dopiero na demo.

### 2.5 Wykresy, które nie kłamią

Wykres mediany ze wstęgą IQR ma jeden sposób na to, żeby wyglądać przekonująco i
wprowadzać w błąd: przebiegi kończą się przy różnych budżetach, więc dalej po
prawej wstęga liczona jest z coraz mniejszej liczby runów i **zwęża się z
powodu, który nie ma nic wspólnego ze zgodnością między nimi**. Systematycznie
schlebia to metodzie, która skończyła najwcześniej.

Rozwiązanie: `full_until_index` wyznacza ostatni punkt, w którym wszystkie
przebiegi jeszcze mają dane. Za nim wstęga jest kreskowana, linia przechodzi w
przerywaną, a tooltip pokazuje `n` w tym miejscu osi. Jest na to test.

Dodatkowo: interpolacja **schodkowa**, nie liniowa — wartość przy budżecie *b* to
ostatni faktyczny pomiar przy budżecie ≤ *b*. Interpolacja liniowa wymyślałaby
pomiary, których nie było. Też jest na to test.

Downsampling to LTTB, nie „co n-ty punkt": szpilka w krzywej straty jest
informacją, a nie szumem. Test sprawdza dokładnie ten przypadek.

### 2.6 Reużycie autoryzacji

§19.2 każe reużyć `auth/repository.py` i to była słuszna instrukcja: hashowanie
haseł, reguły upsertu OAuth i przepływ zatwierdzania to jedyna część obecnej
strony, która **działa**, a przepisanie ich to ryzyko regresji bezpieczeństwa za
darmo.

Reużycie było niemożliwe z dwóch powodów, oba naprawione u źródła: cztery moduły
importowały nieistniejący pakiet `frontend`, a warstwa danych była wpięta w
`st.cache_resource`. Po naprawie te moduły importują się bez Streamlita, a
Streamlit działa dalej bez zmian.

### 2.7 Zrzuty ekranu

W `docs/screenshots/`: przegląd, ranking (jasny i ciemny), lista uruchomień,
strona runu, przeglądarka plików (podgląd PNG i tabela CSV), run nieudany,
porównanie (jasne i ciemne), protokół, zgłoszenie, panel, kolejka.

---

## 3. Czego świadomie nie zrobiono

| Nie zrobiono | Dlaczego |
|---|---|
| **Naprawy ~40 zepsutych importów w `benchmark_core`** | Cudzy moduł (§4). Zmiana nazw w 40 plikach z gałęzi webowej zderzyłaby się z pracą w toku. Zamiast tego most kompatybilności w `src/compat/`, oznaczony jako tymczasowy. Defekt zgłoszony — patrz §5.1. |
| **Naprawy 159 istniejących naruszeń flake8** | Cudze moduły. Zapisane w `.flake8-baseline`; CI wywala się tylko na nowych. Dług jest widoczny i policzalny, nie schowany. |
| **Testu istotności statystycznej na `/compare`** | D3 nierozstrzygnięta. Test bez poprawki na wielokrotne porównania to błąd, który akurat ta publiczność wychwyci natychmiast. Schemat zapisuje ziarno per run, więc analiza jest możliwa od razu po decyzji. |
| **Zaszytej formuły rankingu** | D2 nierozstrzygnięta. Agregat jest wymienialny i opisuje sam siebie; żadna z dostępnych formuł nie miesza gradientów z próbkami. |
| **Logowania Google/Microsoft w nowym froncie** | Wymaga publicznego adresu zwrotnego, którego to środowisko nie ma. Kod autoryzacji jest przeniesiony, ale niewłączony — nie chcę oddawać niesprawdzonej ścieżki logowania. Streamlit ma to działające. |
| **Odczytu `sinfo`/`sacct` w `/admin/queue`** | Poświadczenia SSH ma wyłącznie kontener pollera. Wystawienie tego wymaga uzgodnienia interfejsu z modułem Bartka, a nie drugiego połączenia z klastrem z warstwy webowej. Panel mówi to wprost, zamiast pokazywać puste pole. |
| **Zmian w module kolejki, walidatora i ewaluatora** | §4 — tylko integracja. |
| **Wgrywania własnych datasetów, systemu kredytów, Kubernetesa** | §16 — poza zakresem MVP. |

---

## 4. Znalezione defekty

Numeracja ciągła; pozycje 1–2 to **nowe ustalenia**, których nie ma ani w
briefie, ani w weryfikacji repo.

### 4.1 `benchmark_core` w ogóle się nie importuje — NOWE, poważne

`WERYFIKACJA_REPO.md` §7 ustala, że ewaluator, runner i optymalizatory są na
`main` pod `src/benchmark_core/`, i uznaje D8 za zamkniętą. **Pliki są, ale
pakiet jest niesprawny.**

Refaktor przeniósł kod z `src/benchmark/` do
`src/benchmark_core/optimization_engine/`, ale zostawił **wszystkie importy
bezwzględne wskazujące na stary układ**. Dokładnie:

```
src/benchmark_core/optimization_engine/runner.py:22    from benchmark.evaluator import ModelEvaluator
src/benchmark_core/optimization_engine/runner.py:25    from src.logging import Log
src/benchmark_core/optimization_engine/optimizers/registry.py:1-21   from benchmark.optimizers...
src/benchmark_core/optimization_engine/optimizer_protocols/__init__.py:1-6
src/benchmark_core/optimization_engine/evaluator_dtos/__init__.py:1-8
… łącznie ~40 modułów
```

Do tego dwa braki, które dobijają sprawę:

- `runner.py:124` importuje `from src.dataset import DATA_SETS, MODELS`, a
  **`src/dataset` nie istnieje** na `main`. To samo `src/config` w
  `run_benchmark.py:20`.
- `evaluator_dtos/__init__.py:1` importuje DTO CuPy bezwarunkowo, a ten robi
  `import cupy` w module. Skutek: **na maszynie bez pasującej wersji CuPy nie da
  się załadować nawet czysto-NumPy'owych optymalizatorów.**

**Konsekwencja dla planu:** rekomendacja z `WERYFIKACJA_REPO.md`, żeby seed
danych „użył prawdziwego runnera na CPU", jest niewykonalna — `BenchmarkRunner`
nie ma jak wstać. Stąd lokalny backend CPU (§2.1), który reużywa te części, które
da się uruchomić.

**Obejście, nie naprawa:** `src/compat/benchmark_aliases.py` odtwarza stare nazwy
pakietów jako widoki na katalogi, w których kod leży teraz, więc istniejące
moduły importują się bez zmian. Nazwy są związane jako pakiety syntetyczne z
`__path__`, a nie jako aliasy zaimportowanych modułów — i to jest istotne,
bo `optimization_engine/__init__.py` sam robi `from benchmark.runner import ...`,
więc zaimportowanie go po to, żeby go zaaliasować, wymagałoby, żeby alias już
istniał. **Ten most ma wygasnąć.** Właściwa naprawa to przepisanie importów.

### 4.2 Dwie sprzeczne definicje `StopReason` — NOWE

`optimization_engine/runner.py:28-33` i `metrics/stop_metrics.py:7-13`
deklarują enum o tej samej nazwie i **różnych wartościach**:

| `runner.py` | `stop_metrics.py` |
|---|---|
| `GRADIENT_LIMIT` | `GRADIENT_LIMIT` |
| `DATABASE_LIMIT` | `DATABASE_REACH_LIMIT` |
| `EPOCH_LIMIT` | `EPOCH_LIMIT` |
| `OPTIMIZER_CONVERGED` | `CONVERGENCE` / `OPTIMIZER_SIGNAL` |
| `MAX_STEPS` | — |
| — | `NONE` |

Schemat bazy podąża za wersją z `runner.py`, bo to ona trafia do
`BenchmarkResult`. Do ujednolicenia w module prowadzącego.

### 4.3 `docker compose up` nie wstaje na świeżym klonie — NOWE

`docker-compose.yml:38` montuje
`src/task_queue/config/rabbitmq/definitions.json`, ale ten plik jest w
`.gitignore:147` (słusznie — zawiera hasło brokera), a w repozytorium jest
wyłącznie `definitions.template.json`. Docker tworzy wtedy **katalog** o tej
nazwie, a RabbitMQ wywala się z komunikatem, który nie wspomina o niczym z
powyższych.

Naprawione: `scripts/render_rabbitmq_definitions.sh` generuje plik z szablonu
i `.env`, walidując JSON przed zapisem.

### 4.4 CI — potwierdzone i szersze, niż opisywała weryfikacja

`ci.yml:27,43` wołało `uv sync --extra ci`, a `pyproject.toml` nie ma sekcji
`[project.optional-dependencies]`. Naprawa samego ekstrasu przesunęłaby awarię o
jeden krok: **ani `flake8`, ani `pytest` nie występowały w żadnej liście
zależności**. Oba dodane jako `[dependency-groups].ci`.

Job lintujący instaluje teraz **wyłącznie** tę grupę. Zależności runtime
projektu obejmują `torch`, `torchvision` i `cupy-cuda13x`; ściąganie CUDA 13 po
to, żeby odpalić linter, to minuty i realne ryzyko timeoutów.

**Skutek uboczny wart odnotowania:** naprawa kroku instalacji **odsłoniła 159
istniejących naruszeń flake8**, których nikt nigdy nie widział, bo oba joby
padały wcześniej. Leżą prawie w całości w cudzych modułach. Rozwiązanie w §3.

### 4.5 Martwa konfiguracja flake8 — potwierdzone

`[tool.flake8]` w obu plikach `pyproject.toml` było **ignorowane** (flake8 nie
czyta pyproject bez wtyczki, której repo nie ma) i przypadkiem zgadzało się z
`setup.cfg`, co jest gorsze niż niezgodność — zmiana limitu w martwej sekcji
sprawia wrażenie, że nic się nie stało. Usunięte. `setup.cfg:9` używał też klucza
`include`, który nie jest opcją flake8 i był po cichu pomijany; zastąpiony przez
`extend-exclude`.

### 4.6 Cztery niesprawne importy — potwierdzone i naprawione

Zgodnie z `WERYFIKACJA_REPO.md` §3, dokładnie cztery linie. Naprawione, bo
`auth/repository.py` to plik, który §19.2 każe reużyć.

### 4.7 Wielowątkowość PyTorcha przy małych modelach — obserwacja

Nie jest to defekt repozytorium, ale kosztowało dość czasu, żeby zapisać.
Domyślna liczba wątków BLAS przy sieciach rzędu setek parametrów i wsadzie 32
daje **katastrofalny** efekt: przebieg CMA-ES na 275 parametrach zajmował minuty
przy ~350% CPU, a przypięty do jednego wątku — **2,2 sekundy**. Zmienne trzeba
ustawić **przed** importem NumPy, bo backendy BLAS czytają je przy ładowaniu.
Przy okazji przebieg staje się odtwarzalny, bo kolejność redukcji przestaje
zależeć od tego, ile rdzeni akurat było wolnych.

### 4.8 Defekty kontraktu optymalizatora — bez zmian

Cztery pozycje z §7 briefu (brak `remaining_budget`, brak
`evaluate_population()`, niekontrolowana losowość, `get_output_type()` bez
`@classmethod`) są **udokumentowane wprost na `/docs`**, razem z 4.2. Uczciwość
wobec uczestników jest wartością: znalezienie tych rzeczy metodą prób i błędów w
trakcie konkursu byłoby gorsze dla wszystkich.

---

## 5. Sprostowania do dokumentów wejściowych

Zgodnie z instrukcją, żeby korygować plan, kiedy zderzy się z rzeczywistością
kodu.

1. **`WERYFIKACJA_REPO.md` §7 jest zbyt optymistyczna.** „Kod jest na `main`" —
   tak, ale się nie importuje (§4.1). Wniosek „seed może użyć prawdziwego runnera
   na CPU" jest niewykonalny. D8 **nie jest** zamknięta w sensie użytkowym: kod
   wrócił, sprawność nie.
2. **`WEB_MVP_PLAN.md` §9.4 (uPlot) — odrzucone.** Uzasadnienie w §1.
3. **`WEB_MVP_PLAN.md` §14 krok 10 (walidator z `optim_validation`).** Nie
   pobierałem tej gałęzi. `verify_optimizer.py` jest na `main` i jest sensowny;
   warstwa sandboxa jest napisana od nowa w `app/services/validator.py`, bo i tak
   musiała wołać walidatora przez most kompatybilności z §4.1. Sandbox ma pełen
   zestaw ograniczeń z §7 plus `--cap-drop ALL`, `no-new-privileges` i
   `--pids-limit`.

---

## 6. Otwarte decyzje — status po tej pracy

| # | Decyzja | Status | Jak zamknąć bez migracji |
|---|---|---|---|
| **D1** | Streamlit zostaje czy odchodzi | **ZAMKNIĘTA** — odchodzi z warstwy publicznej | — |
| **D2** | Wspólna waluta budżetu | Otwarta — **i teraz widać ją w liczbach** (§2.2) | Dodać formułę do `SCORE_FORMULAS`; kolumna agregatu jest wymienialna, UI pokazuje aktywną |
| **D3** | Ile powtórzeń, jaki test | Otwarta | `seed` jest w schemacie; `/compare` ma miejsce na test — brakuje wyłącznie decyzji |
| **D4** | Czy wyniki publiczne | Otwarta | **Jedna flaga** `PUBLIC_RESULTS` w `can_read_run` — jedna funkcja, jedno miejsce |
| **D5** | Podział repozytoriów | Otwarta — poza warstwą webową | — |
| **D6** | Wyniki w Postgresie czy pliki | **Rozstrzygnięta w praktyce** — skalary i szeregi w bazie, pliki jako artefakty | — |
| **D7** | Model limitów | Otwarta — API zwraca `409` + `Retry-After` i raportuje pozostały limit | Zmienia się liczba, nie kształt odpowiedzi |
| **D8** | Kiedy przywrócić runner na `main` | **Otwarta inaczej, niż sądzono** — kod wrócił, ale nie działa (§4.1) | Przepisać ~40 importów, usunąć `src/compat/` |

### Co zespół musi rozstrzygnąć, żeby ruszyć dalej

1. **D2 jest teraz pilna, nie teoretyczna.** Ranking działa na prawdziwych
   danych i widać na nim, że CMA-ES i Adam nie są porównywalne bez kursu wymiany.
   To pytanie do doktorów z PW, tak jak zakładano — ale mają już na czym je
   postawić.
2. **Kto naprawia importy `benchmark_core`?** Do tego czasu wszystko, co
   dotyka silnika, przechodzi przez most, który ma wygasnąć.
3. **`src/dataset` i `src/config`.** Bez nich `BenchmarkRunner` nie wstanie
   nigdzie, także na Atenie. To wygląda na zgubione przy refaktorze, nie na
   celowe usunięcie razem z datasetami.
4. **Interfejs do pollera pod `/admin/queue`.** Panel jest gotowy na stan
   Ateny; brakuje uzgodnionego endpointu po stronie modułu, który ma
   poświadczenia.
5. **Rotacja sekretów.** Poświadczenia OAuth i reCAPTCHA były przekazane w
   jawnym tekście w treści zadania i są zapisane na dysku maszyny roboczej.
   Rekomendacja rotacji stoi, niezależnie od losów tej gałęzi.

---

## 7. Jak to uruchomić

```bash
cp .env.template .env          # wypełnić POSTGRES_*, RABBITMQ_*, ATHENA_*, SESSION_SECRET
./scripts/render_rabbitmq_definitions.sh
docker compose up -d --build
```

Zasianie realistycznych danych (uruchamia prawdziwe optymalizatory):

```bash
export SEED_ADMIN_PASSWORD=... SEED_USER_PASSWORD=...
python -m tools.local_backend.seed --downloads ./downloads --seeds 8 --epochs 12
```

Testy:

```bash
cd src/web && uv sync --group dev && uv run pytest tests -q
cd src/web/frontend && npm ci && npm run lint && npm run lint:css && npm run build
python scripts/lint_baseline.py
```

---

*Sporządzono 20.08.2026 na gałęzi `feat/web-mvp`.*
