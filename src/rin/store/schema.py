"""The tables' columns: dev/notes/data_model.md, in code.

Each table here is what one run writes into its own folder. The writer adds `run_id` as the
first column, so a step never sets it. A table joins this file in the milestone that first
writes it.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

TABLES: dict[str, pa.Schema] = {
    "media": pa.schema([
        pa.field("media_id", pa.string(), nullable=False),
        pa.field("path", pa.string(), nullable=False),          # relative to the workspace
        pa.field("kind", pa.string(), nullable=False),          # video | image
        pa.field("width", pa.int32(), nullable=False),
        pa.field("height", pa.int32(), nullable=False),
        pa.field("n_frames", pa.int64(), nullable=False),
        pa.field("site", pa.string()),
        pa.field("camera", pa.string()),
        pa.field("started_at", pa.timestamp("us")),             # wall clock at frame 0
        pa.field("split", pa.string(), nullable=False),
    ]),
    "frames": pa.schema([
        pa.field("media_id", pa.string(), nullable=False),
        pa.field("frame_idx", pa.int64(), nullable=False),      # the join key (D-09)
        pa.field("pts", pa.float64(), nullable=False),          # seconds since frame 0
    ]),
    "frame_sets": pa.schema([
        pa.field("media_id", pa.string(), nullable=False),
        pa.field("frame_idx", pa.int64(), nullable=False),
        pa.field("rank", pa.int64()),                           # order of preference, or null
    ]),
    "frame_scores": pa.schema([                                 # long: one row per score
        pa.field("media_id", pa.string(), nullable=False),
        pa.field("frame_idx", pa.int64(), nullable=False),
        pa.field("name", pa.string(), nullable=False),          # motion, blur, dark, bright
        pa.field("value", pa.float64(), nullable=False),
    ]),
    "embeddings": pa.schema([                                   # index into the run's embeddings.npy
        pa.field("row", pa.int64(), nullable=False),            # row in embeddings.npy
        pa.field("media_id", pa.string(), nullable=False),
        pa.field("frame_idx", pa.int64(), nullable=False),
        pa.field("instance_idx", pa.int64()),                   # null: a whole-frame vector
        pa.field("kind", pa.string(), nullable=False),          # cls | mean_patch | box_pooled
    ]),
    "instances": pa.schema([                                    # one thing found on one frame
        pa.field("instance_idx", pa.int64(), nullable=False),   # counts within the run
        pa.field("media_id", pa.string(), nullable=False),
        pa.field("frame_idx", pa.int64(), nullable=False),
        pa.field("class", pa.string(), nullable=False),         # plain text, e.g. person (D-25)
        pa.field("score", pa.float64()),                        # null for a person's box
        pa.field("x0", pa.float64(), nullable=False),           # box corners in [0, 1] (D-10)
        pa.field("y0", pa.float64(), nullable=False),
        pa.field("x1", pa.float64(), nullable=False),
        pa.field("y1", pa.float64(), nullable=False),
        pa.field("mask_rle", pa.string()),                      # COCO RLE counts, media's full size
        pa.field("keypoints", pa.list_(pa.struct([              # matched by name (D-11)
            pa.field("name", pa.string()), pa.field("x", pa.float64()),
            pa.field("y", pa.float64()), pa.field("score", pa.float64()),
        ]))),
        pa.field("track_id", pa.int64()),                       # same object across frames
    ]),
}

EMBEDDING_KINDS = ("cls", "mean_patch", "box_pooled")

RUN_ID_FIELD = pa.field("run_id", pa.string(), nullable=False)

RUNS_COLUMNS: dict[str, str] = {
    "run_id": "VARCHAR",
    "step": "VARCHAR",
    "model": "VARCHAR",
    "checkpoint": "VARCHAR",
    "checkpoint_sha256": "VARCHAR",
    "config": "JSON",
    "inputs": "VARCHAR[]",
    "tags": "JSON",
    "code": "JSON",
    "who": "VARCHAR",
    "started_at": "TIMESTAMP",
    "finished_at": "TIMESTAMP",
}
"""The runs table, read from every run's run.json. DuckDB types."""


def stored_schema(name: str) -> pa.Schema:
    """A table's schema as stored on disk: run_id first, then the step's columns."""
    if name not in TABLES:
        raise KeyError(f"No table {name!r}. Known tables: {sorted(TABLES)}")
    return pa.schema([RUN_ID_FIELD, *TABLES[name]])


def conform(name: str, table: pa.Table) -> pa.Table:
    """Check a step's table against the schema: same columns, castable types, no nulls where
    the schema forbids them. Returns it reordered and cast. Raises on any mismatch."""
    schema = TABLES[name]
    expected, got = set(schema.names), set(table.column_names)
    if expected != got:
        raise ValueError(
            f"Table {name!r}: missing columns {sorted(expected - got)}, "
            f"unexpected columns {sorted(got - expected)}"
        )
    for field in schema:
        if not field.nullable and table.column(field.name).null_count:
            raise ValueError(f"Table {name!r}: column {field.name!r} has nulls")
    table = table.select(schema.names).cast(schema)
    allowed = {"media": ("video", "image"), "embeddings": EMBEDDING_KINDS}.get(name)
    if allowed is not None and len(table):
        bad = pc.invert(pc.is_in(table.column("kind"), pa.array(allowed)))
        if pc.any(bad).as_py():
            raise ValueError(f"Table {name!r}: kind must be one of {allowed}")
    for column in FINITE.get(name, ()):
        values = table.column(column).drop_null().to_numpy()
        if not np.isfinite(values).all():
            raise ValueError(f"Table {name!r}: column {column!r} has NaN or infinite values")
    if name == "instances":
        _check_boxes(table)
    return table


# Float columns that must hold real numbers: NaN passes every < and > comparison, so range
# checks alone let it through.
FINITE = {"frames": ("pts",), "frame_scores": ("value",), "instances": ("score",)}


def _check_boxes(table: pa.Table) -> None:
    """Every box inside [0, 1] with positive width and height (D-10). Nothing is clamped here:
    a step converting pixels must hand over valid boxes, so a geometry bug fails loudly."""
    x0, y0, x1, y1 = (table.column(c).to_numpy() for c in ("x0", "y0", "x1", "y1"))
    bad = ~((x0 >= 0) & (x0 < x1) & (x1 <= 1) & (y0 >= 0) & (y0 < y1) & (y1 <= 1))
    if bad.any():
        i = int(bad.argmax())
        raise ValueError(
            f"Table 'instances': {int(bad.sum())} boxes outside [0, 1] or with no area, "
            f"e.g. row {i}: ({x0[i]}, {y0[i]}, {x1[i]}, {y1[i]})"
        )
    scores = table.column("score").drop_null().to_numpy()
    if (~((scores >= 0) & (scores <= 1))).any():
        raise ValueError("Table 'instances': score outside [0, 1]")
