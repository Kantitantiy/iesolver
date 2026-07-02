"""
ie-eval — Evaluation harness for iesolver.

Public API for Faz 4.5 MVP::

    from ie_eval import Problem, run_one, run_dataset, ResultStore
    from ie_eval.datasets import ie_case
    from ie_eval.validator import numerical_match, check_feasibility

Ablation solve_fn'leri (EVALUATION_PLAN §5)::

    from ie_eval import (
        make_a1_solve, make_a2_solve, make_a3_correctness_fn,
        make_a4_solve, make_a5_solve,
    )

See ``EVALUATION_PLAN.MD`` for the research protocol.
"""

from ie_eval.ablations import (
    make_a1_solve,
    make_a2_solve,
    make_a3_correctness_fn,
    make_a4_solve,
    make_a5_solve,
)
from ie_eval.baselines import single_shot_cot_solve, single_shot_solve
from ie_eval.problem import GroundTruth, Problem
from ie_eval.runner import RunRecord, run_dataset, run_one
from ie_eval.store import ResultStore

__version__ = "0.1.0"

__all__ = [
    "GroundTruth",
    "Problem",
    "ResultStore",
    "RunRecord",
    "__version__",
    "make_a1_solve",
    "make_a2_solve",
    "make_a3_correctness_fn",
    "make_a4_solve",
    "make_a5_solve",
    "run_dataset",
    "run_one",
    "single_shot_cot_solve",
    "single_shot_solve",
]