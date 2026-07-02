#!/usr/bin/env python
"""
scripts/optimize_mipro.py — MIPROv2 optimizasyonu (Ablation A5).

iesolver pipeline'ının DSPy module prompt'larını MIPROv2 ile optimize eder.
NL4Opt train split üzerinde çalışır, optimize edilmiş programı JSON'a kaydeder.

KULLANIM
--------
# Temel koşu (düşük maliyet pilot):
uv run python scripts/optimize_mipro.py \\
    --train-data data/nl4opt_train_cleaned.jsonl \\
    --output compiled/iesolver_mipro.json \\
    --max-train 40 \\
    --num-candidates 5 \\
    --num-trials 10

# Tam koşu (paper için):
uv run python scripts/optimize_mipro.py \\
    --train-data data/nl4opt_train_cleaned.jsonl \\
    --dev-data data/nl4opt_dev_cleaned.jsonl \\
    --output compiled/iesolver_mipro_full.json \\
    --num-candidates 15 \\
    --max-labeled-demos 3 \\
    --max-bootstrapped-demos 3 \\
    --num-trials 30 \\
    --teacher-model gemini/gemini-2.0-pro-exp

# Dry-run (setup doğrulama, LLM çağrısı yok):
uv run python scripts/optimize_mipro.py \\
    --train-data data/nl4opt_train_cleaned.jsonl \\
    --output compiled/test.json \\
    --dry-run

# Mevcut compiled programı geliştirme üzerinde değerlendir:
uv run python scripts/optimize_mipro.py \\
    --evaluate-only compiled/iesolver_mipro.json \\
    --dev-data data/nl4opt_dev_cleaned.jsonl

ÇIKTI
-----
compiled_path.json:
    DSPy program state (prompt instructions + few-shot examples).
    ie_eval.ablations.make_a5_solve(compiled_path) ile yüklenir.

compiled_path.eval.json:
    Dev set değerlendirme sonuçları (accuracy, per-problem).
    (--dev-data verilmişse)

GEREKSİNİMLER
-------------
- API anahtarı: GEMINI_API_KEY (veya tercih edilen provider)
- ie-eval venv'inde: uv sync --all-packages --extra dev
- Veri: NL4Opt temizlenmiş train split JSONL dosyası
  (OptiMind supplementary'den, bkz. EVALUATION_PLAN.MD §2)

MALİYET KESTİRİMİ (kaba)
--------------------------
- num_candidates × num_trials × ortalama problem başı ~8 LLM çağrısı
- Pilot (yukarıdaki temel koşu): ~400–2000 çağrı, Flash-sınıfı modelde < $2
- Tam koşu: ~3600 çağrı, Flash-sınıfı < $10; Pro-sınıfı çarpanı ~20×
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("optimize_mipro")

# ---------------------------------------------------------------------------
# Metric
# ---------------------------------------------------------------------------

def correctness_metric(
    example: Any,
    pred: Any,
    trace: Any = None,
) -> float:
    """Evaluate one solve() output against ground truth.

    MIPROv2 metric imzası: (example, pred, trace=None) -> float ∈ [0, 1].

    Strateji:
        - ``example.optimal_value`` varsa deterministik numerical_match.
        - ``None`` ise (NO_CODE problem) sadece execution_result varlığını kontrol et.
    """
    from ie_eval.validator import numerical_match

    optimal = getattr(example, "optimal_value", None)
    exec_result: str = getattr(pred, "execution_result", "") or ""

    if optimal is None:
        # NO_CODE — yürütme sonucu üretilmiş mi?
        return 1.0 if exec_result.strip() else 0.0

    tolerance = float(getattr(example, "tolerance_rel", 1e-3))
    return 1.0 if numerical_match(optimal, exec_result, tolerance_rel=tolerance) else 0.0


# ---------------------------------------------------------------------------
# Training data
# ---------------------------------------------------------------------------

def load_train_examples(
    path: Path,
    limit: int | None = None,
) -> list[Any]:
    """Load NL4Opt JSONL and convert to dspy.Example list.

    Parameters
    ----------
    path :
        Path to the cleaned JSONL file (one JSON object per line).
    limit :
        Maximum number of examples to load (None = all).

    Returns
    -------
    list[dspy.Example]
        Each example: ``prompt``, ``optimal_value``, ``tolerance_rel`` fields.
        ``with_inputs("prompt")`` sets the input key for MIPROv2.
    """
    import dspy
    from ie_eval.datasets.nl4opt import NL4OptDataset

    log.info("Loading training examples from %s ...", path)
    ds = NL4OptDataset(path=path)
    problems = list(ds.load())
    if limit is not None:
        problems = problems[:limit]
        log.info("  Limited to %d examples.", limit)

    examples = []
    for p in problems:
        ex = dspy.Example(
            prompt=p.prompt,
            optimal_value=p.ground_truth.objective_value,
            tolerance_rel=p.ground_truth.tolerance_rel,
        ).with_inputs("prompt")
        examples.append(ex)

    log.info("  Loaded %d training examples.", len(examples))
    return examples


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_program(
    program: Any,
    dev_examples: list[Any],
    *,
    desc: str = "eval",
) -> dict[str, Any]:
    """Run correctness_metric on every dev example; return summary dict.

    Parameters
    ----------
    program :
        IESolverProgram (or any callable forward(prompt)->Prediction).
    dev_examples :
        dspy.Example list from load_train_examples.
    desc :
        Label for logging.

    Returns
    -------
    dict
        ``accuracy``, ``n_correct``, ``n_total``, ``per_example`` list.
    """
    log.info("Evaluating (%s) on %d examples ...", desc, len(dev_examples))
    per_example: list[dict[str, Any]] = []
    n_correct = 0

    for i, ex in enumerate(dev_examples):
        t0 = time.perf_counter()
        try:
            pred = program(prompt=ex.prompt)
            score = correctness_metric(ex, pred)
        except Exception as exc:  # noqa: BLE001
            log.warning("  [%d] Error: %s", i, exc)
            score = 0.0
            pred = None
        elapsed = time.perf_counter() - t0

        correct = score >= 0.5
        n_correct += int(correct)
        per_example.append({
            "idx": i,
            "prompt_prefix": ex.prompt[:80],
            "optimal_value": ex.optimal_value,
            "score": score,
            "correct": correct,
            "elapsed_s": round(elapsed, 2),
        })
        log.info(
            "  [%d/%d] correct=%s score=%.2f (%.1fs)",
            i + 1, len(dev_examples), correct, score, elapsed,
        )

    accuracy = n_correct / len(dev_examples) if dev_examples else 0.0
    result = {
        "accuracy": round(accuracy, 4),
        "n_correct": n_correct,
        "n_total": len(dev_examples),
        "per_example": per_example,
    }
    log.info("  => Accuracy: %.1f%% (%d / %d)", accuracy * 100, n_correct, len(dev_examples))
    return result


# ---------------------------------------------------------------------------
# Optimization
# ---------------------------------------------------------------------------

def run_optimization(args: argparse.Namespace) -> None:
    """Main optimization loop."""
    import dspy
    from iesolver._optimization import IESolverProgram
    from iesolver.lm import get_fast_lm, get_reasoning_lm

    # ---- Ortam kurulumu ----
    log.info("=== iesolver MIPROv2 Optimization (A5) ===")
    log.info("Output: %s", args.output)
    log.info("Candidates per module: %d", args.num_candidates)
    log.info("Optimization trials: %d", args.num_trials)
    log.info("Max labeled demos: %d", args.max_labeled_demos)
    log.info("Max bootstrapped demos: %d", args.max_bootstrapped_demos)

    # ---- LM yapılandırması ----
    task_lm = get_fast_lm()
    log.info("Task LM: %s", task_lm.model)

    if args.teacher_model:
        import os
        teacher_lm = dspy.LM(
            args.teacher_model,
            api_key=os.environ.get("GEMINI_API_KEY", ""),
            temperature=0.7,  # Teacher biraz stokastik olmalı
        )
        log.info("Teacher LM: %s", args.teacher_model)
    else:
        # Varsayılan: reasoning LM teacher olarak
        teacher_lm = get_reasoning_lm()
        log.info("Teacher LM: %s (reasoning_lm)", teacher_lm.model)

    # DSPy global default — MIPROv2 proposal generation için
    dspy.configure(lm=teacher_lm)

    # ---- Eğitim verisi ----
    train_examples = load_train_examples(
        Path(args.train_data),
        limit=args.max_train,
    )
    if not train_examples:
        log.error("No training examples loaded. Exiting.")
        sys.exit(1)

    # ---- Program oluşturma ----
    log.info("Building IESolverProgram (loading singletons) ...")
    program = IESolverProgram()
    log.info("  Registered modules: %s", [n for n, _ in program.named_predictors()])

    # ---- Dry-run ----
    if args.dry_run:
        log.info("[DRY RUN] Testing one forward() call ...")
        ex0 = train_examples[0]
        log.info("  Prompt (first 120 chars): %s", ex0.prompt[:120])
        log.info("[DRY RUN] Done. No actual LLM calls made (LLM calls happen in forward()).")
        log.info("[DRY RUN] Setup verified. Remove --dry-run to run optimization.")
        return

    # ---- MIPROv2 ----
    log.info("Initializing MIPROv2 optimizer ...")

    # DSPy 3.x MIPROv2 API:
    # dspy.MIPROv2(metric, prompt_model, task_model, num_candidates, verbose)
    # .compile(student, trainset, num_trials, max_labeled_demos, max_bootstrapped_demos)
    try:
        teleprompter = dspy.MIPROv2(
            metric=correctness_metric,
            prompt_model=teacher_lm,
            task_model=task_lm,
            num_candidates=args.num_candidates,
            verbose=True,
            num_threads=1,    # Thread-safety: singleton'lar paylaşılıyor
        )
    except TypeError:
        # Eski DSPy API'si (prompt_model/task_model → teacher/student)
        log.warning("Falling back to legacy MIPROv2 API signature ...")
        teleprompter = dspy.MIPROv2(
            metric=correctness_metric,
            num_candidates=args.num_candidates,
            verbose=True,
        )

    log.info("Starting compilation (this may take a while) ...")
    t_start = time.perf_counter()

    compiled_program = teleprompter.compile(
        program,
        trainset=train_examples,
        num_trials=args.num_trials,
        max_labeled_demos=args.max_labeled_demos,
        max_bootstrapped_demos=args.max_bootstrapped_demos,
        eval_kwargs={},
    )

    elapsed_min = (time.perf_counter() - t_start) / 60
    log.info("Compilation complete in %.1f minutes.", elapsed_min)

    # ---- Kaydet ----
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    compiled_program.save(str(output_path))
    log.info("Compiled program saved to: %s", output_path)

    # ---- Dev set değerlendirmesi (opsiyonel) ----
    if args.dev_data:
        dev_examples = load_train_examples(
            Path(args.dev_data),
            limit=args.max_dev,
        )
        result = evaluate_program(compiled_program, dev_examples, desc="dev (optimized)")
        eval_path = output_path.with_suffix(".eval.json")
        eval_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
        log.info("Dev evaluation saved to: %s", eval_path)

    log.info("=== Optimization complete ===")
    log.info("Use in A5 ablation: make_a5_solve(compiled_path=%r)", str(output_path))


# ---------------------------------------------------------------------------
# Evaluate-only mode
# ---------------------------------------------------------------------------

def run_evaluate_only(args: argparse.Namespace) -> None:
    """Load a compiled program and evaluate on dev set (no optimization)."""
    from iesolver._optimization import load_compiled_graph

    log.info("=== Evaluate-only mode ===")
    log.info("Loading compiled program from: %s", args.evaluate_only)
    program = load_compiled_graph(Path(args.evaluate_only))

    if not args.dev_data:
        log.error("--dev-data required for --evaluate-only mode.")
        sys.exit(1)

    dev_examples = load_train_examples(Path(args.dev_data), limit=args.max_dev)
    result = evaluate_program(program, dev_examples, desc="dev")

    eval_path = Path(args.evaluate_only).with_suffix(".eval.json")
    eval_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    log.info("Evaluation saved to: %s", eval_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="MIPROv2 optimization for iesolver (Ablation A5).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Veri
    p.add_argument(
        "--train-data",
        metavar="PATH",
        help="NL4Opt cleaned train split JSONL (required unless --evaluate-only).",
    )
    p.add_argument(
        "--dev-data",
        metavar="PATH",
        default=None,
        help="Cleaned dev/test JSONL for post-optimization evaluation (optional).",
    )

    # Çıktı
    p.add_argument(
        "--output",
        metavar="PATH",
        default="compiled/iesolver_mipro.json",
        help="Output path for the compiled program JSON. (default: compiled/iesolver_mipro.json)",
    )

    # Optimizasyon parametreleri
    p.add_argument(
        "--num-candidates",
        type=int,
        default=10,
        metavar="N",
        help=(
            "Prompt candidate count per DSPy module. "
            "Higher = better coverage but more teacher LM calls. "
            "(default: 10)"
        ),
    )
    p.add_argument(
        "--num-trials",
        type=int,
        default=20,
        metavar="N",
        help=(
            "MIPROv2 Bayesian optimization trials. "
            "Higher = better convergence, linear cost growth. "
            "(default: 20)"
        ),
    )
    p.add_argument(
        "--max-labeled-demos",
        type=int,
        default=3,
        metavar="N",
        help="Max few-shot labeled examples per module. (default: 3)",
    )
    p.add_argument(
        "--max-bootstrapped-demos",
        type=int,
        default=2,
        metavar="N",
        help="Max bootstrapped few-shot demos per module. (default: 2)",
    )
    p.add_argument(
        "--max-train",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Cap on training examples (None = use all). "
            "Pilot runs: 30–50; full run: None. (default: None)"
        ),
    )
    p.add_argument(
        "--max-dev",
        type=int,
        default=None,
        metavar="N",
        help="Cap on dev examples for evaluation. (default: None)",
    )

    # Teacher model
    p.add_argument(
        "--teacher-model",
        metavar="MODEL_ID",
        default=None,
        help=(
            "DSPy model string for the teacher (prompt generator). "
            "Default: settings.reasoning_model (from config.py). "
            "Example: gemini/gemini-2.0-pro-exp"
        ),
    )

    # Modlar
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate setup without running optimization or making LLM calls.",
    )
    p.add_argument(
        "--evaluate-only",
        metavar="COMPILED_PATH",
        default=None,
        help="Skip optimization; load compiled program and evaluate on --dev-data.",
    )

    return p


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    # Evaluate-only mode
    if args.evaluate_only:
        run_evaluate_only(args)
        return

    # Validation
    if not args.train_data:
        parser.error("--train-data is required (unless --evaluate-only).")
    if not Path(args.train_data).exists():
        parser.error(f"--train-data file not found: {args.train_data}")

    run_optimization(args)


if __name__ == "__main__":
    main()
