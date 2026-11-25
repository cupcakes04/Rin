"""DuckDB views: each table is the union of that table's files across all runs.

Views are rebuilt before every query (a folder listing, no data read), so a run written a
moment ago is always visible. A table no run has written yet is an empty view with the right
columns, so queries on a new store work instead of failing.
"""

from __future__ import annotations

import duckdb

from rin.store.layout import Layout
from rin.store.schema import RUNS_COLUMNS, TABLES, stored_schema


def _sql_list(paths: list[str]) -> str:
    return "[" + ", ".join("'" + p.replace("'", "''") + "'" for p in paths) + "]"


def refresh(con: duckdb.DuckDBPyConnection, layout: Layout) -> None:
    for name in TABLES:
        files = sorted(str(p) for p in layout.runs.glob(f"*/{name}.parquet"))
        if files:
            con.execute(f"CREATE OR REPLACE VIEW {name} AS SELECT * FROM read_parquet({_sql_list(files)})")
        else:
            con.register(f"_empty_{name}", stored_schema(name).empty_table())
            con.execute(f"CREATE OR REPLACE VIEW {name} AS SELECT * FROM _empty_{name}")

    run_files = sorted(str(p) for p in layout.runs.glob("*/run.json"))
    columns = ", ".join(f"{k!r}: {v!r}" for k, v in RUNS_COLUMNS.items())
    if run_files:
        con.execute(
            f"CREATE OR REPLACE VIEW runs AS SELECT * FROM "
            f"read_json({_sql_list(run_files)}, columns = {{{columns}}}, format = 'auto')"
        )
    else:
        select = ", ".join(f"NULL::{v} AS {k}" for k, v in RUNS_COLUMNS.items())
        con.execute(f"CREATE OR REPLACE VIEW runs AS SELECT {select} WHERE false")

