"""
iesolver.io.data_loader — single-file → DataBundle.

Plan §2.3'teki "tek dosyalı veri girişi" sözleşmesinin somut karşılığı.
Üç format desteklenir; üçü de ``DataBundle`` soyutlamasına indirgenir:

=============  ====================================================
Extension      Resulting DataBundle.tables keys
=============  ====================================================
``.csv``       ``{"data": <DataFrame>}``
``.xlsx``      sheet adlarına göre çoklu tablo
``.sqlite``    kullanıcı tablo adlarına göre çoklu tablo
=============  ====================================================

Mimari karar — neden suffix-based dispatch?
    LLM'in dosya formatını "tahmin etmesini" istemiyoruz; format
    deterministik bir kapı (data_loader) ile sabitlenir, downstream
    node'lar yalnızca DataBundle'ı tüketir. Bu, makalede
    "format-agnostic reasoning units" argümanının dayanağıdır.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from iesolver.state import DataBundle


class DataLoadError(RuntimeError):
    """Raised when a data file cannot be loaded into a DataBundle."""


# =============================================================================
# Public API
# =============================================================================
def load_data(path: Path | str | None) -> DataBundle:
    """Load a single data file into a :class:`DataBundle`.

    Parameters
    ----------
    path :
        Filesystem path. ``None`` returns an empty bundle (prompt-only run).

    Returns
    -------
    DataBundle
        Normalized in-memory representation, ready for downstream nodes.

    Raises
    ------
    DataLoadError
        When the file is missing, not a regular file, has an unsupported
        suffix, or fails to parse.
    """
    if path is None:
        return DataBundle(source_type="none")

    p = path if isinstance(path, Path) else Path(path)

    if not p.exists():
        raise DataLoadError(f"Data file not found: {p}")
    if not p.is_file():
        raise DataLoadError(f"Not a regular file: {p}")

    suffix = p.suffix.lower()

    try:
        if suffix == ".csv":
            return _load_csv(p)
        if suffix in {".xlsx", ".xls"}:
            return _load_xlsx(p)
        if suffix in {".sqlite", ".sqlite3", ".db"}:
            return _load_sqlite(p)
    except DataLoadError:
        raise
    except Exception as exc:  # noqa: BLE001  — explicit wrap
        raise DataLoadError(f"Failed to parse {p}: {exc}") from exc

    raise DataLoadError(
        f"Unsupported file extension {suffix!r} for {p}. "
        "Supported: .csv, .xlsx, .xls, .sqlite, .sqlite3, .db"
    )


# =============================================================================
# Format-specific loaders
# =============================================================================
def _load_csv(path: Path) -> DataBundle:
    """Load a single-table CSV. Key: ``"data"``."""
    df = pd.read_csv(path)
    return DataBundle(
        tables={"data": df},
        source_path=path,
        source_type="csv",
    )


def _load_xlsx(path: Path) -> DataBundle:
    """Load every sheet of an XLSX workbook. Sheet name = table key."""
    # sheet_name=None → dict[str, DataFrame] for all sheets at once
    sheets: dict[str, pd.DataFrame] = pd.read_excel(path, sheet_name=None)
    if not sheets:
        raise DataLoadError(f"Workbook contains no sheets: {path}")
    return DataBundle(
        tables=sheets,
        source_path=path,
        source_type="xlsx",
    )


def _load_sqlite(path: Path) -> DataBundle:
    """Load every user table of a SQLite file. Table name = key.

    sqlite_*  iç sistem tabloları atlanır.
    """
    with sqlite3.connect(str(path)) as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type = 'table' AND name NOT LIKE 'sqlite_%' "
            "ORDER BY name"
        )
        table_names = [row[0] for row in cur.fetchall()]

        if not table_names:
            raise DataLoadError(f"SQLite file contains no user tables: {path}")

        tables: dict[str, pd.DataFrame] = {}
        for name in table_names:
            # Identifier quoting: SQLite double-quotes; içerideki " kaçırılır.
            quoted = '"' + name.replace('"', '""') + '"'
            tables[name] = pd.read_sql_query(f"SELECT * FROM {quoted}", conn)

    return DataBundle(
        tables=tables,
        source_path=path,
        source_type="sqlite",
    )


__all__ = ["DataLoadError", "load_data"]
