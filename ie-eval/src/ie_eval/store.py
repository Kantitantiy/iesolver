"""
ie_eval.store — SQLite results store.

Şema:
    runs(run_id, config_id, problem_id, run_idx, success, elapsed_s,
         execution_rate, numerical_match, feasibility_checked, feasibility_ok,
         retry_count, error_class, error_message, tokens_in, tokens_out,
         cost_usd, llm_calls, node_count, created_at)

    node_metrics(run_id, node_name, latency_ms, tokens_in, tokens_out,
                 cost_usd, llm_calls, invocations, error_class)

Yalnızca birincil metric sütunları indekslenir (config_id, problem_id).
State snapshot'ları saklanmaz — LangGraph SqliteSaver zaten checkpoint
tutuyor; burada replay için değil, agregasyon için depolama.

Kullanım:
    store = ResultStore(Path("results.sqlite"))
    store.persist(rec)     # runner callback olarak geçirilebilir
    df = store.as_dataframe()   # pandas ile analiz
"""

from __future__ import annotations

import json
import sqlite3
import time
from contextlib import closing
from pathlib import Path
from typing import Any

from ie_eval.runner import RunRecord


_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id              INTEGER PRIMARY KEY AUTOINCREMENT,
    config_id           TEXT    NOT NULL,
    problem_id          TEXT    NOT NULL,
    run_idx             INTEGER NOT NULL,
    success             INTEGER NOT NULL,
    elapsed_s           REAL    NOT NULL,
    execution_rate      INTEGER NOT NULL,
    numerical_match     INTEGER NOT NULL,
    feasibility_checked INTEGER NOT NULL,
    feasibility_ok      INTEGER NOT NULL,
    retry_count         INTEGER NOT NULL DEFAULT 0,
    error_class         TEXT,
    error_message       TEXT,
    tokens_in           INTEGER NOT NULL DEFAULT 0,
    tokens_out          INTEGER NOT NULL DEFAULT 0,
    cost_usd            REAL    NOT NULL DEFAULT 0.0,
    llm_calls           INTEGER NOT NULL DEFAULT 0,
    node_count          INTEGER NOT NULL DEFAULT 0,
    violations_json     TEXT,
    metadata_json       TEXT,
    created_at          REAL    NOT NULL
);

CREATE INDEX IF NOT EXISTS ix_runs_config    ON runs(config_id);
CREATE INDEX IF NOT EXISTS ix_runs_problem   ON runs(problem_id);
CREATE INDEX IF NOT EXISTS ix_runs_config_problem ON runs(config_id, problem_id);

CREATE TABLE IF NOT EXISTS node_metrics (
    run_id       INTEGER NOT NULL,
    node_name    TEXT    NOT NULL,
    latency_ms   REAL    NOT NULL,
    tokens_in    INTEGER NOT NULL DEFAULT 0,
    tokens_out   INTEGER NOT NULL DEFAULT 0,
    cost_usd     REAL    NOT NULL DEFAULT 0.0,
    llm_calls    INTEGER NOT NULL DEFAULT 0,
    invocations  INTEGER NOT NULL DEFAULT 1,
    error_class  TEXT,
    PRIMARY KEY (run_id, node_name),
    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
);
"""


class ResultStore:
    """Thin SQLite wrapper for evaluation runs.

    Concurrency: WAL mode; single writer expected (Faz 4.5 runner sırayla
    koşar). Paralel runner Faz 5 sonrası → connection-per-thread pattern
    eklenecek.
    """

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    # ---- schema ------------------------------------------------------------
    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _init_schema(self) -> None:
        with closing(self._connect()) as conn, conn:
            conn.executescript(_SCHEMA)
            # Backward-compat migration: eski DB'lerde metadata_json yoksa ekle.
            # SQLite ALTER TABLE ADD COLUMN idempotent değil → try/except.
            try:
                conn.execute("ALTER TABLE runs ADD COLUMN metadata_json TEXT")
            except sqlite3.OperationalError:
                pass  # sütun zaten var

    # ---- write -------------------------------------------------------------
    def persist(self, rec: RunRecord) -> int:
        """Insert one RunRecord (+ its per-node metrics). Return run_id."""
        m = rec.metrics
        if m is None:
            # Şansızlık: metrics extractor çalışmadı; minimum yaz
            feas_checked = 0
            feas_ok = 0
            violations_json: str | None = None
            tokens_in = tokens_out = llm_calls = node_count = 0
            cost_usd = 0.0
            execution_rate = 0
            numerical_match = 0
            error_class = None
            per_node: dict[str, dict[str, Any]] = {}
            retry_count = 0
        else:
            feas_checked = int(m.feasibility.checked)
            feas_ok = int(m.feasibility.feasible)
            violations_json = (
                json.dumps(m.feasibility.violations) if m.feasibility.violations else None
            )
            tokens_in = m.total_tokens_in
            tokens_out = m.total_tokens_out
            cost_usd = m.total_cost_usd
            llm_calls = m.total_llm_calls
            node_count = m.node_count
            execution_rate = int(m.execution_rate)
            numerical_match = int(m.numerical_match)
            error_class = m.error_class
            per_node = m.per_node
            retry_count = m.retry_count

        metadata_json = (
            json.dumps(rec.metadata) if getattr(rec, "metadata", None) else None
        )

        with closing(self._connect()) as conn, conn:
            cur = conn.execute(
                """
                INSERT INTO runs (
                    config_id, problem_id, run_idx, success, elapsed_s,
                    execution_rate, numerical_match,
                    feasibility_checked, feasibility_ok,
                    retry_count, error_class, error_message,
                    tokens_in, tokens_out, cost_usd, llm_calls, node_count,
                    violations_json, metadata_json, created_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    rec.config_id, rec.problem_id, rec.run_idx,
                    int(rec.success), rec.elapsed_s,
                    execution_rate, numerical_match,
                    feas_checked, feas_ok,
                    retry_count, error_class, rec.error,
                    tokens_in, tokens_out, cost_usd, llm_calls, node_count,
                    violations_json, metadata_json, time.time(),
                ),
            )
            run_id = int(cur.lastrowid or 0)

            for node_name, slice_ in per_node.items():
                conn.execute(
                    """
                    INSERT INTO node_metrics (
                        run_id, node_name, latency_ms,
                        tokens_in, tokens_out, cost_usd,
                        llm_calls, invocations, error_class
                    ) VALUES (?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        run_id, node_name,
                        float(slice_.get("latency_ms", 0.0) or 0.0),
                        int(slice_.get("tokens_in", 0) or 0),
                        int(slice_.get("tokens_out", 0) or 0),
                        float(slice_.get("cost_usd", 0.0) or 0.0),
                        int(slice_.get("llm_calls", 0) or 0),
                        int(slice_.get("invocations", 1) or 1),
                        slice_.get("error_class"),
                    ),
                )

        return run_id

    # ---- read --------------------------------------------------------------
    def list_runs(self, config_id: str | None = None) -> list[dict[str, Any]]:
        """Return all rows in ``runs`` as list of dicts.

        ``metadata_json`` sütunu otomatik parse edilir → ``row["metadata"]``
        (dict veya {} yoksa). Ham ``metadata_json`` string'i de korunur.
        """
        query = "SELECT * FROM runs"
        params: tuple = ()
        if config_id is not None:
            query += " WHERE config_id = ?"
            params = (config_id,)
        query += " ORDER BY run_id"
        with closing(self._connect()) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(query, params).fetchall()

        out: list[dict[str, Any]] = []
        for r in rows:
            d = dict(r)
            raw = d.get("metadata_json")
            if raw:
                try:
                    d["metadata"] = json.loads(raw)
                except json.JSONDecodeError:
                    d["metadata"] = {}
            else:
                d["metadata"] = {}
            out.append(d)
        return out

    def count(self) -> int:
        with closing(self._connect()) as conn:
            (n,) = conn.execute("SELECT COUNT(*) FROM runs").fetchone()
        return int(n)