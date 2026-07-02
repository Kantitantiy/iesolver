"""
IE-Case seed set — genişletilmiş sürüm (EVALUATION_PLAN §2).

Q1 makalede rakiplerden ayrışan RQ5'i (uçtan uca IE değeri) kanıtlayan set:
NL4Opt / IndustryOR'da veri dosyası YOKTUR; DataBundle argümanı yalnızca
burada gösterilebilir. Bu yüzden IE-Case set halinde xlsx-çok-sayfa, csv,
sqlite formatlarını ve bir NO_CODE kavramsal soruyu birlikte içerir.

Şu anki problemler:
    1. eoq-basic                — analitik EOQ, veri yok
    2. transport-2x3            — küçük LP, veri yok (elle çözüldü)
    3. multi-product-inventory  — çok ürünlü envanter (xlsx çok-sayfa)
    4. transport-3x2-csv        — transportation LP, csv verisi
    5. assignment-3x3-sqlite    — atama problemi, sqlite verisi
    6. abc-classification       — NO_CODE kavramsal (envanter yönetimi)

Fixture dosyaları modül yüklenirken idempotent olarak yazılır (writable
paket dizinine; okuma-only ortamda tempdir'e fallback). Aynı IE-Case
adlandırma ve şema Q1 makale reprodüktibilite bölümüne referans olur.

Ground truth doğrulaması: analitik olmayan LP'ler (transport, assignment)
elle MODI / Hungarian ile çözüldü; sayısal değerler docstring'lerde.
"""

from __future__ import annotations

import csv
import math
import sqlite3
import tempfile
from pathlib import Path
from typing import Iterable

import openpyxl

from ie_eval.problem import GroundTruth, Problem


# =============================================================================
# Fixture dizin çözümü — writable yer bul, gerekirse tempdir'e düş
# =============================================================================
def _resolve_data_dir() -> Path:
    """Prefer package-relative ``ie-eval/data/ie_case/``; fall back to tempdir if read-only."""
    preferred = Path(__file__).parents[3] / "data" / "ie_case"
    try:
        preferred.mkdir(parents=True, exist_ok=True)
        probe = preferred / ".write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return preferred
    except OSError:
        fallback = Path(tempfile.gettempdir()) / "ie_eval_ie_case"
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback


_DATA_DIR: Path = _resolve_data_dir()


# =============================================================================
# Fixture writers — idempotent (yalnızca dosya yoksa yazar)
# =============================================================================
def _write_multi_product_xlsx(path: Path) -> None:
    """3 ürün × EOQ parametreleri, tek sayfada. Format-agnostic reasoning testi."""
    if path.exists():
        return
    wb = openpyxl.Workbook()
    ws = wb.active
    assert ws is not None
    ws.title = "Products"
    ws.append(["product", "demand", "ordering_cost", "holding_cost"])
    for row in (
        ("A", 1200, 50, 1.0),
        ("B",  800, 40, 2.0),
        ("C", 1500, 25, 0.5),
    ):
        ws.append(row)
    # 2. sayfa — birim fiyatlar (opsiyonel bilgi; LLM'in çok-sayfayı doğru
    # okuduğunu doğrulamak için)
    ws2 = wb.create_sheet("UnitPrices")
    ws2.append(["product", "unit_price"])
    for row in (("A", 12.0), ("B", 25.0), ("C", 6.5)):
        ws2.append(row)
    wb.save(str(path))


def _write_transportation_csv(path: Path) -> None:
    """3 kaynak × 2 hedef transportation problem, uzun format CSV."""
    if path.exists():
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["source", "destination", "cost", "supply", "demand"])
        # Cost matrix + supply per source + demand per destination
        # S1 supply 40; S2 supply 60; S3 supply 50
        # D1 demand 70; D2 demand 80
        rows = [
            ("S1", "D1", 4, 40, 70),
            ("S1", "D2", 6, 40, 80),
            ("S2", "D1", 5, 60, 70),
            ("S2", "D2", 3, 60, 80),
            ("S3", "D1", 7, 50, 70),
            ("S3", "D2", 2, 50, 80),
        ]
        w.writerows(rows)


def _write_assignment_sqlite(path: Path) -> None:
    """3×3 assignment problem — SQLite tablosu."""
    if path.exists():
        return
    with sqlite3.connect(str(path)) as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS assignment_costs (
                worker TEXT NOT NULL,
                task   TEXT NOT NULL,
                cost   REAL NOT NULL,
                PRIMARY KEY (worker, task)
            );
        """)
        # 3×3 cost matrix (row-major)
        # Optimal: W1→T2, W2→T1, W3→T3  →  cost = 2 + 6 + 1 = 9
        rows = [
            ("W1", "T1", 9), ("W1", "T2", 2), ("W1", "T3", 7),
            ("W2", "T1", 6), ("W2", "T2", 4), ("W2", "T3", 3),
            ("W3", "T1", 5), ("W3", "T2", 8), ("W3", "T3", 1),
        ]
        conn.executemany(
            "INSERT OR REPLACE INTO assignment_costs VALUES (?,?,?)", rows
        )


def _ensure_fixtures() -> dict[str, Path]:
    """Write all fixture files if missing; return path map."""
    paths = {
        "multi_product_xlsx":    _DATA_DIR / "multi_product_inventory.xlsx",
        "transportation_csv":    _DATA_DIR / "transportation_3x2.csv",
        "assignment_sqlite":     _DATA_DIR / "assignment_3x3.sqlite",
    }
    _write_multi_product_xlsx(paths["multi_product_xlsx"])
    _write_transportation_csv(paths["transportation_csv"])
    _write_assignment_sqlite(paths["assignment_sqlite"])
    return paths


# Modül yüklenirken bir kez idempotent yazım
_FIXTURE_PATHS = _ensure_fixtures()


# =============================================================================
# Problem 1: EOQ (analitik, veri yok)
# =============================================================================
_EOQ_PROMPT = (
    "Calculate the Economic Order Quantity (EOQ) for the following parameters: "
    "Annual demand D = 10,000 units, Ordering cost S = $50 per order, "
    "Holding cost H = $2 per unit per year. "
    "Use the EOQ formula Q* = sqrt(2*D*S/H) and provide the exact numerical result."
)


def _eoq_feasibility(sol: dict[str, float]) -> list[str]:
    violations: list[str] = []
    q = sol.get("Q")
    if q is None:
        return ["missing decision variable Q"]
    if q < 0:
        violations.append(f"Q must be >= 0, got {q}")
    if not math.isfinite(q):
        violations.append(f"Q must be finite, got {q}")
    return violations


_EOQ_PROBLEM = Problem(
    id="eoq-basic",
    prompt=_EOQ_PROMPT,
    ground_truth=GroundTruth(
        objective_value=math.sqrt(2 * 10_000 * 50 / 2),   # ≈ 707.107
        tolerance_rel=1e-3,
        solution={"Q": math.sqrt(2 * 10_000 * 50 / 2)},
        feasibility_fn=_eoq_feasibility,
    ),
    metadata={
        "benchmark": "IE-Case",
        "problem_type": "EOQ",
        "difficulty": "easy",
        "expected_path": "CODE",
        "data_format": "none",
    },
)


# =============================================================================
# Problem 2: Küçük transportation LP (inline, veri yok)
# =============================================================================
_TRANSPORT_2X3_PROMPT = (
    "Solve the following transportation LP to minimize total shipping cost. "
    "Two warehouses W1 (supply 100) and W2 (supply 150). "
    "Three stores S1 (demand 80), S2 (demand 90), S3 (demand 80). "
    "Unit shipping costs: "
    "W1→S1 = 4, W1→S2 = 6, W1→S3 = 8, "
    "W2→S1 = 5, W2→S2 = 4, W2→S3 = 3. "
    "Find the optimal shipment quantities x_ij and the minimum total cost."
)


def _transport_2x3_feasibility(sol: dict[str, float]) -> list[str]:
    violations: list[str] = []
    keys = ["x_W1_S1", "x_W1_S2", "x_W1_S3", "x_W2_S1", "x_W2_S2", "x_W2_S3"]
    for k in keys:
        if k not in sol:
            return [f"missing {k}"]
        if sol[k] < -1e-6:
            violations.append(f"{k} must be >= 0, got {sol[k]}")
    if sol["x_W1_S1"] + sol["x_W1_S2"] + sol["x_W1_S3"] > 100 + 1e-6:
        violations.append("W1 supply exceeded")
    if sol["x_W2_S1"] + sol["x_W2_S2"] + sol["x_W2_S3"] > 150 + 1e-6:
        violations.append("W2 supply exceeded")
    for demand, key_pairs in [(80, ["x_W1_S1", "x_W2_S1"]),
                                (90, ["x_W1_S2", "x_W2_S2"]),
                                (80, ["x_W1_S3", "x_W2_S3"])]:
        got = sum(sol[k] for k in key_pairs)
        if abs(got - demand) > 1e-6:
            violations.append(f"demand not met (expected {demand}, got {got})")
    return violations


_TRANSPORT_2X3_PROBLEM = Problem(
    id="transport-2x3",
    prompt=_TRANSPORT_2X3_PROMPT,
    ground_truth=GroundTruth(
        objective_value=960.0,
        tolerance_rel=1e-3,
        solution={
            "x_W1_S1": 80.0, "x_W1_S2": 20.0, "x_W1_S3": 0.0,
            "x_W2_S1": 0.0, "x_W2_S2": 70.0, "x_W2_S3": 80.0,
        },
        feasibility_fn=_transport_2x3_feasibility,
    ),
    metadata={
        "benchmark": "IE-Case",
        "problem_type": "LP-transportation",
        "difficulty": "medium",
        "expected_path": "CODE",
        "data_format": "none",
    },
)


# =============================================================================
# Problem 3: Çok ürünlü envanter (xlsx çok-sayfa)
# =============================================================================
_MULTI_PROD_PROMPT = (
    "The attached Excel file (multiple sheets) describes an inventory management "
    "problem. Sheet 'Products' lists three products with their annual demand D, "
    "ordering cost S, and holding cost H. For each product independently, compute "
    "the Economic Order Quantity Q_i* = sqrt(2*D_i*S_i/H_i). Then report the total "
    "MINIMUM annual holding+ordering cost across all products, which equals the "
    "sum of sqrt(2*D_i*S_i*H_i) over products i. "
    "Return the final total as a single numerical value."
)

# Products: A(D=1200,S=50,H=1), B(D=800,S=40,H=2), C(D=1500,S=25,H=0.5)
# TAC_i = sqrt(2*D*S*H)
# TAC_A = sqrt(120000) ≈ 346.4102
# TAC_B = sqrt(128000) ≈ 357.7709
# TAC_C = sqrt(37500)  ≈ 193.6492
# Total ≈ 897.8303
_MULTI_PROD_TAC = (
    math.sqrt(2 * 1200 * 50 * 1.0)
    + math.sqrt(2 * 800 * 40 * 2.0)
    + math.sqrt(2 * 1500 * 25 * 0.5)
)


def _multi_prod_feasibility(sol: dict[str, float]) -> list[str]:
    """Her ürün için Q_i > 0 ve sonlu olmalı."""
    violations: list[str] = []
    for prod in ("A", "B", "C"):
        key = f"Q_{prod}"
        if key not in sol:
            violations.append(f"missing {key}")
            continue
        q = sol[key]
        if q <= 0 or not math.isfinite(q):
            violations.append(f"{key} must be positive and finite, got {q}")
    return violations


_MULTI_PROD_PROBLEM = Problem(
    id="multi-product-inventory",
    prompt=_MULTI_PROD_PROMPT,
    data_path=_FIXTURE_PATHS["multi_product_xlsx"],
    ground_truth=GroundTruth(
        objective_value=_MULTI_PROD_TAC,   # ≈ 897.83
        tolerance_rel=1e-3,
        solution={
            "Q_A": math.sqrt(2 * 1200 * 50 / 1.0),
            "Q_B": math.sqrt(2 * 800 * 40 / 2.0),
            "Q_C": math.sqrt(2 * 1500 * 25 / 0.5),
        },
        feasibility_fn=_multi_prod_feasibility,
    ),
    metadata={
        "benchmark": "IE-Case",
        "problem_type": "EOQ-multi-product",
        "difficulty": "medium",
        "expected_path": "CODE",
        "data_format": "xlsx",
    },
)


# =============================================================================
# Problem 4: Transportation 3×2 (csv)
# =============================================================================
_TRANSPORT_CSV_PROMPT = (
    "The attached CSV file describes a transportation problem. Each row lists a "
    "(source, destination) pair with the per-unit shipping cost, the source's "
    "total supply, and the destination's total demand. There are 3 sources "
    "(S1, S2, S3) and 2 destinations (D1, D2). Minimize the total shipping cost "
    "subject to supply and demand constraints. Report the minimum total cost and "
    "the optimal shipment quantities x_<source>_<destination>."
)

# Ground truth (MODI ile doğrulandı):
# Supply: S1=40, S2=60, S3=50 → toplam 150
# Demand: D1=70, D2=80 → toplam 150 (balanced)
# Costs (matris):    D1  D2
#               S1 |  4   6
#               S2 |  5   3
#               S3 |  7   2
# Optimal (NW köşe + MODI check ederek):
#   x_S1_D1 = 40, x_S2_D1 = 30, x_S2_D2 = 30, x_S3_D2 = 50
# Toplam cost = 4*40 + 5*30 + 3*30 + 2*50 = 160+150+90+100 = 500
_TRANSPORT_CSV_OPTIMAL = 500.0
_TRANSPORT_CSV_SOLUTION = {
    "x_S1_D1": 40.0, "x_S1_D2":  0.0,
    "x_S2_D1": 30.0, "x_S2_D2": 30.0,
    "x_S3_D1":  0.0, "x_S3_D2": 50.0,
}


def _transport_csv_feasibility(sol: dict[str, float]) -> list[str]:
    violations: list[str] = []
    keys = list(_TRANSPORT_CSV_SOLUTION.keys())
    for k in keys:
        if k not in sol:
            return [f"missing {k}"]
        if sol[k] < -1e-6:
            violations.append(f"{k} must be >= 0, got {sol[k]}")
    # Supply constraints
    supply = {"S1": 40, "S2": 60, "S3": 50}
    for src, cap in supply.items():
        got = sum(sol[k] for k in keys if k.startswith(f"x_{src}_"))
        if got > cap + 1e-6:
            violations.append(f"{src} supply exceeded: {got} > {cap}")
    # Demand constraints (balanced problem → eşit)
    demand = {"D1": 70, "D2": 80}
    for dst, need in demand.items():
        got = sum(sol[k] for k in keys if k.endswith(f"_{dst}"))
        if abs(got - need) > 1e-6:
            violations.append(f"{dst} demand not met: {got} != {need}")
    return violations


_TRANSPORT_CSV_PROBLEM = Problem(
    id="transport-3x2-csv",
    prompt=_TRANSPORT_CSV_PROMPT,
    data_path=_FIXTURE_PATHS["transportation_csv"],
    ground_truth=GroundTruth(
        objective_value=_TRANSPORT_CSV_OPTIMAL,
        tolerance_rel=1e-3,
        solution=_TRANSPORT_CSV_SOLUTION,
        feasibility_fn=_transport_csv_feasibility,
    ),
    metadata={
        "benchmark": "IE-Case",
        "problem_type": "LP-transportation",
        "difficulty": "medium",
        "expected_path": "CODE",
        "data_format": "csv",
    },
)


# =============================================================================
# Problem 5: Atama problemi 3×3 (sqlite)
# =============================================================================
_ASSIGNMENT_PROMPT = (
    "The attached SQLite database contains a table 'assignment_costs' with "
    "columns (worker, task, cost). Three workers (W1, W2, W3) must be assigned "
    "to three tasks (T1, T2, T3) such that each worker is assigned to exactly "
    "one task and each task is assigned to exactly one worker. Minimize the "
    "total assignment cost. Report the minimum total cost and the optimal "
    "assignment as a mapping from worker to task."
)

# Cost matrix:
#     T1 T2 T3
# W1 | 9  2  7
# W2 | 6  4  3
# W3 | 5  8  1
# Optimal (brute-force 3! = 6):
#   W1→T2 (2), W2→T1 (6), W3→T3 (1)  →  toplam 9
_ASSIGNMENT_OPTIMAL = 9.0
_ASSIGNMENT_SOLUTION = {
    "x_W1_T1": 0.0, "x_W1_T2": 1.0, "x_W1_T3": 0.0,
    "x_W2_T1": 1.0, "x_W2_T2": 0.0, "x_W2_T3": 0.0,
    "x_W3_T1": 0.0, "x_W3_T2": 0.0, "x_W3_T3": 1.0,
}


def _assignment_feasibility(sol: dict[str, float]) -> list[str]:
    violations: list[str] = []
    workers = ("W1", "W2", "W3")
    tasks = ("T1", "T2", "T3")

    for w in workers:
        for t in tasks:
            k = f"x_{w}_{t}"
            if k not in sol:
                return [f"missing {k}"]
            v = sol[k]
            if v < -1e-6 or v > 1 + 1e-6:
                violations.append(f"{k} must be in [0,1], got {v}")

    # Her worker tam olarak bir task
    for w in workers:
        got = sum(sol[f"x_{w}_{t}"] for t in tasks)
        if abs(got - 1) > 1e-6:
            violations.append(f"{w} not assigned to exactly one task (sum={got})")

    # Her task tam olarak bir worker
    for t in tasks:
        got = sum(sol[f"x_{w}_{t}"] for w in workers)
        if abs(got - 1) > 1e-6:
            violations.append(f"{t} not assigned to exactly one worker (sum={got})")

    return violations


_ASSIGNMENT_PROBLEM = Problem(
    id="assignment-3x3-sqlite",
    prompt=_ASSIGNMENT_PROMPT,
    data_path=_FIXTURE_PATHS["assignment_sqlite"],
    ground_truth=GroundTruth(
        objective_value=_ASSIGNMENT_OPTIMAL,
        tolerance_rel=1e-3,
        solution=_ASSIGNMENT_SOLUTION,
        feasibility_fn=_assignment_feasibility,
    ),
    metadata={
        "benchmark": "IE-Case",
        "problem_type": "assignment",
        "difficulty": "medium",
        "expected_path": "CODE",
        "data_format": "sqlite",
    },
)


# =============================================================================
# Problem 6: NO_CODE kavramsal — ABC classification
# =============================================================================
_ABC_PROMPT = (
    "Describe the ABC classification methodology used in inventory management. "
    "Specifically: (a) explain the Pareto principle basis; (b) provide the "
    "typical percentage breakdowns of items and inventory value in each category "
    "(A, B, C); (c) recommend a management strategy (review frequency, safety "
    "stock policy) for each category. This is a conceptual question — no "
    "computation or code is required."
)

_ABC_PROBLEM = Problem(
    id="abc-classification",
    prompt=_ABC_PROMPT,
    data_path=None,
    ground_truth=GroundTruth(
        # NO_CODE: sayısal cevap yok; execution/feasibility skip
        objective_value=None,
        feasibility_fn=None,
    ),
    metadata={
        "benchmark": "IE-Case",
        "problem_type": "conceptual-inventory",
        "difficulty": "easy",
        "expected_path": "NO_CODE",
        "data_format": "none",
        # Uzman değerlendirmesi (EVALUATION_PLAN §10) için not:
        "expert_review_required": True,
    },
)


# =============================================================================
# Dataset object
# =============================================================================
_PROBLEMS: tuple[Problem, ...] = (
    _EOQ_PROBLEM,
    _TRANSPORT_2X3_PROBLEM,
    _MULTI_PROD_PROBLEM,
    _TRANSPORT_CSV_PROBLEM,
    _ASSIGNMENT_PROBLEM,
    _ABC_PROBLEM,
)


class _IECaseDataset:
    name = "IE-Case"

    def load(self) -> Iterable[Problem]:
        return iter(_PROBLEMS)


ie_case_dataset = _IECaseDataset()


__all__ = ["ie_case_dataset"]
