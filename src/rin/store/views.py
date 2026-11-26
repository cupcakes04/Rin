"""DuckDB views: each table is the union of that table's files across all runs.

Views are rebuilt before every query, so a run written a moment ago is always visible. Each view
reads a file pattern (`runs/*/media.parquet`), so the query text stays the same size however
many runs exist. A table no run has written yet is an empty view with the right columns, so
queries on a new store work instead of failing (DuckDB refuses a pattern that matches nothing).
Files are matched by column name, not position: a column added to a table later reads as null
in older runs' files instead of breaking every query on them.
"""

from __future__ import annotations

import duckdb

from rin.store.layout import Layout
from rin.store.schema import RUNS_COLUMNS, TABLES, stored_schema


def _quoted(text: str) -> str:
    return "'" + text.replace("'", "''") + "'"


def refresh(con: duckdb.DuckDBPyConnection, layout: Layout) -> None:
    for name in TABLES:
        pattern = f"*/{name}.parquet"
        if any(layout.runs.glob(pattern)):
            con.execute(
                f"CREATE OR REPLACE VIEW {name} AS SELECT * FROM "
                f"read_parquet({_quoted(str(layout.runs / pattern))}, union_by_name = true)"
            )
        else:
            con.register(f"_empty_{name}", stored_schema(name).empty_table())
            con.execute(f"CREATE OR REPLACE VIEW {name} AS SELECT * FROM _empty_{name}")

    columns = ", ".join(f"{k!r}: {v!r}" for k, v in RUNS_COLUMNS.items())
    if any(layout.runs.glob("*/run.json")):
        con.execute(
            f"CREATE OR REPLACE VIEW runs AS SELECT * FROM "
            f"read_json({_quoted(str(layout.runs / '*/run.json'))}, columns = {{{columns}}}, format = 'auto')"
        )
    else:
        select = ", ".join(f"NULL::{v} AS {k}" for k, v in RUNS_COLUMNS.items())
        con.execute(f"CREATE OR REPLACE VIEW runs AS SELECT {select} WHERE false")
