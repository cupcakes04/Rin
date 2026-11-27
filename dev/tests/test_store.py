"""rin.store and Master: runs are written whole or not at all; tables are queryable views."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from rin import Master
from rin.store import vendor_commit, write_durably

WRITER_MODULE = sys.modules["rin.store.writer"]


def _frame_set(n: int = 3) -> pa.Table:
    return pa.table({"media_id": ["m1"] * n, "frame_idx": list(range(n)), "rank": [None] * n})


def test_a_new_store_answers_queries_with_empty_tables(m: Master) -> None:
    for table in ("runs", "media", "frames", "frame_sets"):
        assert m.sql(f"SELECT count(*) FROM {table}").fetchall()[0] == (0,)


def test_a_finished_run_appears_whole(m: Master) -> None:
    with m.writer(step="test.step", config={"a": 1}, inputs=[], tags={"method": "kcenter"}) as w:
        w.write("frame_sets", _frame_set())
    run_dir = m.layout.run_dir(w.run_id)
    # src.diff is there too while src/ has uncommitted changes (test_uncommitted_code_is_kept...)
    assert sorted(p.name for p in run_dir.iterdir() if p.name != "src.diff") == ["frame_sets.parquet", "run.json"]
    assert not any(m.layout.tmp.iterdir())

    info = json.loads((run_dir / "run.json").read_text())
    assert info["step"] == "test.step" and info["config"] == {"a": 1}
    assert info["finished_at"] is not None and info["who"] and info["code"]["rin"]

    run = m.run(w.run_id)
    assert len(run) == 3
    assert m.sql("SELECT tags->>'method' FROM runs").fetchall()[0] == ("kcenter",)


def test_delete_removes_a_run_only_when_nothing_needs_it(m: Master) -> None:
    with m.writer(step="test.parent", config={}, inputs=[]) as w:
        w.write("frame_sets", _frame_set())
    parent = m.run(w.run_id)
    with m.writer(step="test.child", config={}, inputs=[parent.id]) as w:
        w.write("frame_sets", _frame_set(2))
    child = m.run(w.run_id)
    with pytest.raises(ValueError, match="input of 1 runs"):
        m.delete(parent)
    m.delete(child)
    m.delete(parent)
    assert m.sql("SELECT count(*) FROM runs").fetchall() == [(0,)]
    assert not m.layout.run_dir(parent.id).exists() and not any(m.layout.tmp.iterdir())


def test_delete_refuses_a_persons_work(m: Master) -> None:
    with m.writer(step="human.annotate", config={}, inputs=[]) as w:
        w.write("frame_sets", _frame_set())
    with pytest.raises(ValueError, match="person's work"):
        m.delete(m.run(w.run_id))
    assert m.layout.run_dir(w.run_id).exists()


def test_a_failed_run_leaves_nothing(m: Master) -> None:
    with pytest.raises(RuntimeError, match="boom"):
        with m.writer(step="test.step", config={}, inputs=[]) as w:
            w.write("frame_sets", _frame_set())
            raise RuntimeError("boom")
    assert not any(m.layout.runs.iterdir())
    assert not any(m.layout.tmp.iterdir())


def test_who_is_the_login_user_unless_passed(m: Master) -> None:
    with m.writer(step="test.step", config={}, inputs=[]) as default:
        default.write("frame_sets", _frame_set())
    with m.writer(step="test.step", config={}, inputs=[], who="annotator_a") as passed:
        passed.write("frame_sets", _frame_set())
    who = dict(m.sql("SELECT run_id, who FROM runs").fetchall())
    assert who[passed.run_id] == "annotator_a" and who[default.run_id] not in ("", "annotator_a")
    with pytest.raises(ValueError, match="who"):
        m.writer(step="test.step", config={}, inputs=[], who=" ")


def test_views_see_runs_written_after_the_store_was_opened(m: Master) -> None:
    for n in (2, 3, 4):
        with m.writer(step="test.step", config={}, inputs=[]) as w:
            w.write("frame_sets", _frame_set(n))
        assert m.sql("SELECT count(*) FROM frame_sets").fetchall()[0] == (sum(range(2, n + 1)),)


def test_opening_a_store_warns_about_leftover_run_folders(m: Master) -> None:
    """A run killed hard never reaches __exit__, so its temporary folder stays behind."""
    (m.layout.tmp / "20260101-000000-core.ingest-dead").mkdir()
    with pytest.warns(UserWarning, match="1 unfinished run folder"):
        Master(m.layout.root)


def test_a_run_that_writes_nothing_is_refused(m: Master) -> None:
    with pytest.raises(RuntimeError, match="wrote no tables"):
        with m.writer(step="test.step", config={}, inputs=[]):
            pass
    assert not any(m.layout.runs.iterdir())


def test_wrong_columns_are_refused(m: Master) -> None:
    with pytest.raises(ValueError, match="missing columns"):
        with m.writer(step="test.step", config={}, inputs=[]) as w:
            w.write("frame_sets", pa.table({"media_id": ["m1"], "frame_idx": [0]}))
    assert not any(m.layout.runs.iterdir())


def test_nulls_where_the_schema_forbids_them_are_refused(m: Master) -> None:
    table = pa.table({"media_id": ["m1", None], "frame_idx": [0, 1], "rank": [None, None]})
    with pytest.raises(ValueError, match="has nulls"):
        with m.writer(step="test.step", config={}, inputs=[]) as w:
            w.write("frame_sets", table)


def test_config_that_isnt_json_is_refused_before_anything_runs(m: Master) -> None:
    with pytest.raises(TypeError):
        m.writer(step="test.step", config={"path": object()}, inputs=[])


def test_unknown_run_is_an_error(m: Master) -> None:
    with pytest.raises(KeyError, match="No run"):
        m.run("20260101-000000-nope-0000")


def test_uncommitted_code_is_kept_with_the_run(m: Master, monkeypatch: pytest.MonkeyPatch) -> None:
    patch = "--- a/src/x.py\n+++ b/src/x.py\n@@ -1 +1 @@\n-a = 1\n+a = 2\n"
    dirty = {"rin": "abc", "src_dirty": True, "src_diff_sha256": hashlib.sha256(patch.encode()).hexdigest()}
    monkeypatch.setattr(WRITER_MODULE, "code_version", lambda: (dirty, patch))
    with m.writer(step="test.step", config={}, inputs=[]) as w:
        w.write("frame_sets", _frame_set())
    run_dir = m.layout.run_dir(w.run_id)
    assert (run_dir / "src.diff").read_text() == patch
    assert json.loads((run_dir / "run.json").read_text())["code"] == dirty

    monkeypatch.setattr(WRITER_MODULE, "code_version", lambda: ({"rin": "abc", "src_dirty": False}, ""))
    with m.writer(step="test.step", config={}, inputs=[]) as clean:
        clean.write("frame_sets", _frame_set())
    assert not (m.layout.run_dir(clean.run_id) / "src.diff").exists()


def test_code_version_matches_its_own_patch() -> None:
    info, patch = WRITER_MODULE.code_version()
    assert info["src_dirty"] == bool(patch)
    if patch:
        assert info["src_diff_sha256"] == hashlib.sha256(patch.encode()).hexdigest()


def test_a_column_added_later_reads_as_null_in_older_runs(m: Master) -> None:
    """A table that gains a column must not break queries on files written before it."""
    with m.writer(step="test.step", config={}, inputs=[]) as w:
        w.write("frame_sets", _frame_set(2))
    old = m.layout.runs / "20200101-000000-test.old-0000"
    old.mkdir()
    pq.write_table(pa.table({"run_id": ["20200101-000000-test.old-0000"], "media_id": ["m9"], "frame_idx": [7]}),
                   old / "frame_sets.parquet")
    rows = m.sql("SELECT media_id, frame_idx, rank FROM frame_sets ORDER BY media_id").fetchall()
    assert rows == [("m1", 0, None), ("m1", 1, None), ("m9", 7, None)]


def test_a_durable_write_is_whole_or_absent(tmp_path: Path) -> None:
    target = tmp_path / "a.txt"
    write_durably(target, lambda p: p.write_text("hello"))
    assert target.read_text() == "hello" and [p.name for p in tmp_path.iterdir()] == ["a.txt"]

    def crash(p: Path) -> None:
        p.write_text("half")
        raise OSError("disk gone")

    with pytest.raises(OSError, match="disk gone"):
        write_durably(tmp_path / "b.txt", crash)
    assert [p.name for p in tmp_path.iterdir()] == ["a.txt"]  # no b.txt, no .partial left behind


def _instances(boxes: list[tuple[float, float, float, float]]) -> pa.Table:
    n = len(boxes)
    return pa.table({
        "instance_idx": list(range(n)), "media_id": ["m1"] * n, "frame_idx": [0] * n, "class": ["person"] * n,
        "score": [0.9] * n, "x0": [b[0] for b in boxes], "y0": [b[1] for b in boxes],
        "x1": [b[2] for b in boxes], "y1": [b[3] for b in boxes],
        "mask_rle": pa.nulls(n, pa.string()), "keypoints": pa.nulls(n), "track_id": pa.nulls(n, pa.int64()),
    })


def test_boxes_inside_the_unit_square_are_stored(m: Master) -> None:
    with m.writer(step="test.step", config={}, inputs=[]) as w:
        w.write("instances", _instances([(0.0, 0.0, 1.0, 1.0), (0.2, 0.3, 0.4, 0.9)]))
    assert m.run(w.run_id).instances().df()[["x0", "y1"]].values.tolist() == [[0.0, 1.0], [0.2, 0.9]]


@pytest.mark.parametrize("box", [(-0.01, 0, 0.5, 0.5), (0, 0, 1.01, 0.5), (0.5, 0, 0.5, 0.5), (0.6, 0, 0.5, 0.5), (0, 0.7, 0.5, 0.2)])
def test_boxes_outside_the_unit_square_or_without_area_are_refused(m: Master, box: tuple) -> None:
    with pytest.raises(ValueError, match="outside"):
        with m.writer(step="test.step", config={}, inputs=[]) as w:
            w.write("instances", _instances([(0.1, 0.1, 0.2, 0.2), box]))
    assert not any(m.layout.runs.iterdir())


def test_vendor_commit_is_the_checkout_head(tmp_path: Path) -> None:
    import subprocess

    def git(*args: str) -> str:
        return subprocess.run(["git", "-C", str(tmp_path), *args], check=True, capture_output=True, text=True).stdout.strip()

    git("init", "-q")
    (tmp_path / "a.txt").write_text("1")
    git("add", "a.txt")
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "x")
    head = git("rev-parse", "HEAD")
    assert vendor_commit(tmp_path) == head
    (tmp_path / "a.txt").write_text("2")
    assert vendor_commit(tmp_path) == head + "+dirty"



def test_a_nan_score_is_refused(m: Master) -> None:
    table = _instances([(0.1, 0.1, 0.2, 0.2)]).set_column(4, "score", pa.array([float("nan")]))
    with pytest.raises(ValueError, match="NaN"):
        with m.writer(step="test.step", config={}, inputs=[]) as w:
            w.write("instances", table)


def test_a_nan_frame_score_is_refused(m: Master) -> None:
    table = pa.table({"media_id": ["m1"], "frame_idx": [0], "name": ["motion"], "value": [float("nan")]})
    with pytest.raises(ValueError, match="NaN"):
        with m.writer(step="test.step", config={}, inputs=[]) as w:
            w.write("frame_scores", table)


def test_files_in_a_runs_subfolders_are_synced(m: Master, monkeypatch: pytest.MonkeyPatch) -> None:
    synced: list[Path] = []
    real = WRITER_MODULE._fsync
    monkeypatch.setattr(WRITER_MODULE, "_fsync", lambda path: (synced.append(path), real(path)))
    with m.writer(step="test.step", config={}, inputs=[]) as w:
        w.write("frame_sets", _frame_set())
        (w.tmp_dir / "files" / "labels").mkdir(parents=True)
        (w.tmp_dir / "files" / "labels" / "a.txt").write_text("0 0.5 0.5 0.1 0.1")
    names = [p.name for p in synced]
    assert names.index("a.txt") < names.index("labels") < names.index("files")
