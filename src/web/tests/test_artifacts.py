"""Containment tests for the artifact browser.

§12.4 makes this the one non-negotiable piece of the web layer: it is the only
barrier between a logged-in participant's browser and the datasets, and holding
the datasets privately is what would make the competition meaningless.

The cases below are the attack list from the plan. Three of them (5, 6, 15)
need real symlinks on disk -- without those fixtures the most dangerous vector is
not covered at all, and scp is documented to carry symlinks over from the
cluster, so the link can arrive with no attacker on this side.
"""

import os
import uuid
from pathlib import Path

import pytest

from app.services import artifacts


@pytest.fixture
def run_id() -> uuid.UUID:
    return uuid.uuid4()


@pytest.fixture
def root(tmp_path: Path, run_id: uuid.UUID) -> Path:
    """An artifact root with one run, plus secrets outside it to aim at."""
    downloads = tmp_path / "downloads"
    run = downloads / str(run_id)
    (run / "reports").mkdir(parents=True)
    (run / "logs").mkdir(parents=True)

    (run / "reports" / "loss_vs_grads.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 64)
    (run / "logs" / "4718392.out").write_text("srun: job 4718392 completed\n")
    (run / "metadata.json").write_text('{"seed": 42}')
    (run / "optimizer.py").write_text("class MyOptimizer:\n    pass\n")

    # The things this module exists to keep unreachable.
    (tmp_path / "secret.txt").write_text("dataset weights")
    datasets = tmp_path / "datasets"
    datasets.mkdir()
    (datasets / "cifar.bin").write_text("secret dataset")

    # A neighbouring run whose directory name shares a prefix with ours: this is
    # what makes a startswith() containment check wrong.
    (downloads / f"{run_id}-other").mkdir()
    (downloads / f"{run_id}-other" / "leak.txt").write_text("another run")

    return downloads


def test_reads_a_normal_file(root, run_id):
    target = artifacts.resolve(run_id, "logs/4718392.out", root)
    assert target.name == "4718392.out"
    assert b"4718392" in artifacts.read_preview(target)


@pytest.mark.parametrize(
    "attack",
    [
        "../../secret.txt",                 # 1  plain traversal
        "..%2f..%2fsecret.txt",             # 2  encoded, already decoded by the framework
        "....//....//secret.txt",           # 3  doubled-up traversal
        "/etc/passwd",                      # 4  absolute
        "../datasets/cifar.bin",            # traversal into the dataset directory
        "reports/../../../secret.txt",      # traversal after a valid prefix
        "\\..\\..\\secret.txt",             # backslash separators
    ],
)
def test_rejects_traversal(root, run_id, attack):
    with pytest.raises(artifacts.ArtifactError):
        artifacts.resolve(run_id, attack, root)


def test_rejects_null_byte(root, run_id):
    with pytest.raises(artifacts.ArtifactRejected):
        artifacts.resolve(run_id, "reports/loss\x00.png", root)


def test_rejects_overlong_path(root, run_id):
    with pytest.raises(artifacts.ArtifactRejected):
        artifacts.resolve(run_id, "a/" * 700, root)


def test_rejects_malformed_task_id(root):
    """A UUID cannot contain a separator, so parsing it is itself a control."""
    with pytest.raises(artifacts.ArtifactRejected):
        artifacts.run_root("../another-run", root)


def test_prefix_overlap_is_not_containment(root, run_id):
    """The neighbouring directory starts with the same string and is not ours."""
    with pytest.raises(artifacts.ArtifactError):
        artifacts.resolve(run_id, "../" + f"{run_id}-other" + "/leak.txt", root)


def test_symlink_to_file_outside_is_refused(root, run_id):
    link = root / str(run_id) / "reports" / "escape.png"
    link.symlink_to(root.parent / "secret.txt")
    with pytest.raises(artifacts.ArtifactError):
        target = artifacts.resolve(run_id, "reports/escape.png", root)
        artifacts.read_preview(target)


def test_symlink_to_dataset_directory_is_refused(root, run_id):
    link = root / str(run_id) / "datasets"
    link.symlink_to(root.parent / "datasets", target_is_directory=True)
    with pytest.raises(artifacts.ArtifactError):
        artifacts.resolve(run_id, "datasets/cifar.bin", root)


def test_symlink_staying_inside_the_run_resolves_to_its_target(root, run_id):
    """An in-bounds link is canonicalised, then read from the canonical path.

    This is the correct outcome rather than a gap. Containment is decided on the
    resolved path, so a link that leaves the run is already refused by the tests
    above; one that does not leave it points at a file the caller may read
    anyway. Reading the canonical path, rather than the link, is also what makes
    the open safe: the name can be repointed afterwards without changing which
    file was opened.
    """
    run = root / str(run_id)
    (run / "alias.out").symlink_to(run / "logs" / "4718392.out")

    resolved = artifacts.resolve(run_id, "alias.out", root)
    assert resolved == (run / "logs" / "4718392.out").resolve()

    fd, st = artifacts.open_regular_file(resolved)
    os.close(fd)
    assert st.st_size > 0

    # It is still absent from the tree, so nothing advertises it.
    assert "alias.out" not in {entry.path for entry in artifacts.walk(run)}


def test_open_refuses_a_symlink_swapped_in_after_the_check(root, run_id):
    """O_NOFOLLOW is what closes the window between resolving and reading.

    The canonical path is checked and then, before it is opened, replaced by a
    link pointing outside. This is not hypothetical here: the downloader writes
    into these directories while they are being browsed, and scp carries
    symlinks over from the cluster.
    """
    run = root / str(run_id)
    victim = run / "logs" / "4718392.out"
    resolved = artifacts.resolve(run_id, "logs/4718392.out", root)

    victim.unlink()
    victim.symlink_to(root.parent / "secret.txt")

    with pytest.raises(artifacts.ArtifactRejected):
        artifacts.open_regular_file(resolved)


def test_walk_omits_symlinks(root, run_id):
    run = root / str(run_id)
    (run / "reports" / "escape.png").symlink_to(root.parent / "secret.txt")
    (run / "datasets").symlink_to(root.parent / "datasets", target_is_directory=True)

    names = {entry.path for entry in artifacts.walk(run)}
    assert "reports/escape.png" not in names
    assert "datasets" not in names
    assert "reports/loss_vs_grads.png" in names


def test_walk_omits_special_files(root, run_id):
    run = root / str(run_id)
    os.mkfifo(run / "pipe")
    assert "pipe" not in {entry.path for entry in artifacts.walk(run)}


def test_preview_limit(root, run_id):
    big = root / str(run_id) / "big.out"
    big.write_bytes(b"x" * 4096)
    with pytest.raises(artifacts.ArtifactTooLarge):
        artifacts.read_preview(big, limit=1024)
    assert len(artifacts.read_preview(big, limit=8192)) == 4096


def test_only_png_is_served_inline():
    """SVG and HTML are downloads, not renderings.

    An SVG carries script and is HTML for this purpose; serving a participant's
    file as markup on our own origin hands them the session of anybody who
    opens it.
    """
    assert artifacts.content_disposition(Path("a/loss.png")) == ("image/png", True)

    for hostile in ("evil.svg", "evil.html", "evil.htm", "evil.xhtml", "evil.js"):
        content_type, inline = artifacts.content_disposition(Path(hostile))
        assert inline is False
        assert content_type == "application/octet-stream"


def test_text_types_are_attachments_with_a_safe_type():
    for name in ("optimizer.py", "job.out", "run.log", "notes.txt"):
        content_type, inline = artifacts.content_disposition(Path(name))
        assert inline is False
        assert content_type.startswith("text/plain")


def test_unknown_extension_falls_back_to_a_download():
    content_type, inline = artifacts.content_disposition(Path("weights.pt"))
    assert (content_type, inline) == ("application/octet-stream", False)


def test_every_raw_response_is_hardened():
    headers = artifacts.hardening_headers("loss.png", inline=True)
    assert headers["X-Content-Type-Options"] == "nosniff"
    # Makes the response an opaque origin, so even a mistake in the whitelist
    # above cannot reach a cookie on the main domain.
    assert headers["Content-Security-Policy"] == "sandbox"
    assert headers["Cache-Control"] == "private, no-store"
    assert headers["Content-Disposition"].startswith("inline;")

    assert artifacts.hardening_headers("x.bin", inline=False)[
        "Content-Disposition"
    ].startswith("attachment;")


def test_filename_cannot_break_out_of_the_header():
    headers = artifacts.hardening_headers('a"; drop\r\nX-Injected: 1', inline=False)
    value = headers["Content-Disposition"]
    assert "\r" not in value and "\n" not in value
    assert value.count('"') == 2


def test_python_source_is_returned_as_bytes_not_executed(root, run_id):
    """A .py file containing markup is data, and stays data."""
    hostile = root / str(run_id) / "xss.py"
    hostile.write_text("<script>alert(1)</script>\n")
    content_type, inline = artifacts.content_disposition(hostile)
    assert content_type.startswith("text/plain")
    assert inline is False
    assert b"<script>" in artifacts.read_preview(hostile)


def test_directory_summary_counts_only_regular_files(root, run_id):
    run = root / str(run_id)
    (run / "reports" / "escape.png").symlink_to(root.parent / "secret.txt")
    files, total = artifacts.directory_summary(run)
    assert files == 4
    assert total > 0


def test_missing_run_directory_is_not_found(root):
    with pytest.raises(artifacts.ArtifactNotFound):
        artifacts.run_root(uuid.uuid4(), root)
