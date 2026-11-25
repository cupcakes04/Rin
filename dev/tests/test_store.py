"""rin.store and Master: runs are written whole or not at all; tables are queryable views."""

from __future__ import annotations

import json

import pyarrow as pa
import pytest

from rin import Master


def _frame_set(n: int = 3) -> pa.Table:
    return pa.table({"media_id": ["m1"] * n, "frame_idx": list(range(n)), "rank": [None] * n})


def test_a_new_store_answers_queries_with_empty_tables(m: Master) -> None:
    for table in ("runs", "media", "frames", "frame_sets"):
        assert m.sql(f"SELECT count(*) FROM {table}").fetchall()[0] == (0,)


def test_a_finished_run_appears_whole(m: Master) -> None:
    with m.writer(step="test.step", config={"a": 1}, inputs=[], tags={"method": "kcenter"}) as w:
        w.write("frame_sets", _frame_set())
    run_dir = m.layout.run_dir(w.run_id)
    assert sorted(p.name for p in run_dir.iterdir()) == ["frame_sets.parquet", "run.json"]
    assert not any(m.layout.tmp.iterdir())

    info = json.loads((run_dir / "run.json").read_text())
    assert info["step"] == "test.step" and info["config"] == {"a": 1}
    assert info["finished_at"] is not None and info["who"] and info["code"]["rin"]

    run = m.run(w.run_id)
    assert len(run) == 3
    assert m.sql("SELECT tags->>'method' FROM runs").fetchall()[0] == ("kcenter",)


def test_a_failed_run_leaves_nothing(m: Master) -> None:
    with pytest.raises(RuntimeError, match="boom"):
        with m.writer(step="test.step", config={}, inputs=[]) as w:
            w.write("frame_sets", _frame_set())
            raise RuntimeError("boom")
    assert not any(m.layout.runs.iterdir())
    assert not any(m.layout.tmp.iterdir())


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
