import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
FRONTEND_ROOT = REPO_ROOT / "src" / "web" / "frontend"
POLISH_TEXT = re.compile(r"[ąćęłńóśźżĄĆĘŁŃÓŚŹŻ]")


USER_FACING_BACKEND = (
    REPO_ROOT / "src" / "web" / "app" / "services" / "naming.py",
    REPO_ROOT / "src" / "web" / "app" / "services" / "leaderboard.py",
    REPO_ROOT / "src" / "web" / "app" / "services" / "validator.py",
    REPO_ROOT / "src" / "web" / "app" / "routers" / "admin.py",
    REPO_ROOT / "src" / "web" / "app" / "routers" / "auth.py",
    REPO_ROOT / "src" / "web" / "app" / "routers" / "compare.py",
    REPO_ROOT / "src" / "web" / "app" / "routers" / "meta.py",
    REPO_ROOT / "src" / "web" / "app" / "routers" / "submissions.py",
    REPO_ROOT / "src" / "web" / "app" / "main.py",
)


def test_document_declares_english_metadata():
    markup = (FRONTEND_ROOT / "index.html").read_text(encoding="utf-8")

    assert '<html lang="en">' in markup
    assert "<title>Opti-Mark (The Golem Benchmark)</title>" in markup
    assert '<meta name="application-name" content="Opti-Mark (The Golem Benchmark)" />' in markup
    assert (
        'content="A platform for testing, analyzing, and comparing machine-learning optimizers."'
        in markup
    )


def test_user_facing_sources_contain_no_polish_text():
    frontend_files = sorted(
        (*FRONTEND_ROOT.glob("src/**/*.tsx"), *FRONTEND_ROOT.glob("src/**/*.ts"))
    )
    for path in (*frontend_files, *USER_FACING_BACKEND):
        assert not POLISH_TEXT.search(path.read_text(encoding="utf-8")), path
