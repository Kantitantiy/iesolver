# ie-eval

Evaluation harness for [`iesolver`](../src/iesolver). Implements the experimental
protocol described in `../EVALUATION_PLAN.MD` for the Q1 paper.

## Faz 4.5 MVP kapsamı

- `ie_eval.problem` — Problem dataclass
- `ie_eval.datasets` — Dataset protocol + IE-Case seed set
- `ie_eval.validator` — Deterministic feasibility check (DESIGN_REVIEW §3.2)
- `ie_eval.runner` — Batch runner around `iesolver.solve(auto_mode=True)`
- `ie_eval.store` — SQLite results store
- `ie_eval.metrics` — Numerical match + accuracy aggregation

## Design boundary

`ie_eval` **yalnızca** iesolver'ın public API'sini kullanır:

```python
from iesolver import solve, is_interrupted, SolverState, DataBundle
```

`iesolver.nodes`, `iesolver.signatures`, `iesolver.observability` gibi iç
modüllerden import **yasaktır** — bu sınır, "iesolver bir kütüphanedir"
tasarım hedefinin fiili doğrulamasıdır.

## Çalıştırma

```bash
# Kurulum (workspace root'tan)
uv sync --all-packages --extra dev

# Test
uv run --package ie-eval pytest
```

## NL4Opt kullanımı

```python
from pathlib import Path
from ie_eval.datasets import NL4OptDataset
from ie_eval.runner import run_dataset
from ie_eval.store import ResultStore

ds = NL4OptDataset(
    path=Path("data/nl4opt_test_cleaned.jsonl"),
    cleaning="cleaned",        # EVALUATION_PLAN §2: temiz sürüm zorunlu
    limit=None,                # veya küçük altkümede pilot: limit=20
)

store = ResultStore(Path("results/nl4opt.sqlite"))
recs = run_dataset(ds, config_id="baseline", n_runs=3, on_result=store.persist)
```

**Kaynak notu**: NL4Opt orijinal set 16 etiket hatası içerir. Temizlenmiş
sürüm için OptiMind (arxiv 2509.22979) supplementary'e bakın. Adaptör
`.jsonl` (satır başı obje) ve `.json` (top-level array) formatlarını okur;
alan adları esnektir (`document`/`question`/`prompt`; `optimal_value`/
`gold_optimal_value` vb.).

## Baseline'lar (EVALUATION_PLAN §4)

`iesolver.solve` yerine tek-atış LLM karşılaştırma noktaları:

```python
from ie_eval import single_shot_solve, single_shot_cot_solve
from ie_eval.runner import run_dataset
from ie_eval.datasets import NL4OptDataset

ds = NL4OptDataset(path=Path("data/nl4opt_test_cleaned.jsonl"))

# Pipeline (kontrol grubu)
run_dataset(ds, config_id="pipeline", solve_fn=None)   # None → iesolver.solve

# Baseline 1: tek atış LLM
run_dataset(ds, config_id="single_shot",
            solve_fn=lambda p, **kw: single_shot_solve(p))

# Baseline 2: tek atış + CoT
run_dataset(ds, config_id="single_shot_cot",
            solve_fn=lambda p, **kw: single_shot_cot_solve(p))
```

Baseline'lar iesolver'ın **public** LM ve sandbox helper'larını kullanır
(`get_fast_lm`, `run_code`) — iç modüllere dokunmaz. Aynı model, aynı
sandbox → pipeline'ın katma değeri izole edilir.

## Analysis (EVALUATION_PLAN §3, §7)

Koşulardan sonra store'dan konfigürasyon karşılaştırması:

```python
from ie_eval.analysis import (
    summarize_by_config, compare_configs, per_problem_correctness,
    mcnemar_test, bootstrap_diff_ci,
    format_summary, format_comparison,
)

print(format_summary(summarize_by_config(store, "pipeline")))
print(format_summary(summarize_by_config(store, "single_shot")))

comp = compare_configs(store, "pipeline", "single_shot", policy="majority")
mc = mcnemar_test(comp.only_a_correct, comp.only_b_correct)

a = list(per_problem_correctness(store, "pipeline",   policy="majority").values())
b = list(per_problem_correctness(store, "single_shot", policy="majority").values())
ci = bootstrap_diff_ci(a, b, n_iterations=10_000, seed=42)

print(format_comparison(comp, mc, ci))
```

**Notlar:**
- **McNemar**: n ≥ 25'te sürekli düzeltmeli χ² (df=1); n < 25'te exact binomial. scipy'siz.
- **Bootstrap CI**: percentile method, seedable → tekrarlanabilir (§3.7).
- **Aggregation policy**: `"majority"` (default), `"all"`, `"any"`, `"first"`. Her problem başına tek 0/1'e indirir.

### Benchmark/problem_type kırılımı

`runs.metadata_json` sütunu problem.metadata'yı taşır. Analysis fonksiyonlarında
`metadata_filter` ile kırılımlı raporlar:

```python
# Sadece NL4Opt
s_nl = summarize_by_config(store, "pipeline",
                            metadata_filter={"benchmark": "NL4Opt"})

# Yalnızca LP + MILP problem tipleri (callable form)
s_lp = summarize_by_config(store, "pipeline",
                            metadata_filter=lambda md:
                                md.get("problem_type") in {"LP", "MILP"})

# Kırılımlı karşılaştırma
comp_nl = compare_configs(store, "pipeline", "single_shot",
                          metadata_filter={"benchmark": "NL4Opt"})
```

### IndustryOR

```python
from ie_eval.datasets import IndustryORDataset

ds = IndustryORDataset(path=Path("data/industryor_cleaned.jsonl"),
                        cleaning="cleaned")   # 23 invalid dışlanmış, 50 düzeltilmiş
```

NL4Opt ile aynı şema toleransı (`_jsonl_common` ortak taban); IndustryOR'a
özgü ek metadata: `sector`, `industry`. SOTA ~%37 — zorluk kanıtı burada.