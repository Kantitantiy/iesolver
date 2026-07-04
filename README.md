# iesolver

End-to-end LLM-driven solver for industrial engineering (IE) problems, built on
**DSPy** (reasoning units) and **LangGraph** (workflow orchestration). Developed
as the reference implementation for an academic paper on agentic LLM systems
for operations research / IE problem solving.

```python
from iesolver import solve

result = solve("EOQ problem: annual demand D=10000, order cost S=50, holding cost H=2")
print(result["executive_summary"])
```

## Overview

Given a natural-language problem statement and an optional single data file
(`.csv` / `.xlsx` / `.sqlite`), iesolver extracts requirements, decides whether
the problem needs generated code or a purely analytical answer, produces and
validates a solution, runs a sensitivity analysis, and writes a three-tier
report (Executive Summary / Technical Output / Action Directives) in HTML,
DOCX, or PDF.

**Architectural split:**

| Layer | Responsibility | Implementation |
|---|---|---|
| **DSPy** | "What should one LLM call do?" | Signature (typed I/O contract) + Module (`Predict` / `ChainOfThought` / `ReAct`) |
| **LangGraph** | "How do the stages connect?" | `TypedDict` state, conditional edges, retry loops, SQLite checkpointing, human-in-the-loop `interrupt()` |

Every LangGraph node runs exactly one DSPy module. State lives in LangGraph;
reasoning lives in DSPy — this separation ("reasoning units" vs. "workflow
engine") is the core methodological argument of the paper.

## Pipeline

```
START → intake → requirement ──[incomplete]──▶ clarify ─┬─[interactive]→ requirement (loop)
             │ [complete]                                 │
             ▼                                            └─[auto_mode]──▶ refine
           refine ◀──────────────────────────────────────────────────────────┘
             │
             ▼
           route ──[NO_CODE]──▶ chain_branch ─────────────────────────────┐
             │                                                             │
             └──[CODE]──▶ code_branch (algo_select → constraint_adapt →   │
                            output_spec → generate[ReAct + sandbox])       │
                                     │                                     │
                                     ▼                                     │
                                 validate                                  │
                                /       \                                  │
                      [invalid+retry]  [valid]   [invalid+max_retry]      │
                            │             │            │                  │
                      code_branch    sensitivity     report ◀──────────────┘
                      (max 3x)            │             ▲
                                          ▼             │
                                     artifacts ─────────┘
                                                          ▼
                                                         END
```

* **Inner recovery loop:** the ReAct code generator writes code, runs it in a
  sandboxed subprocess, reads the error, and retries — up to 3 iterations —
  without ever leaving the `code_branch` node.
* **Outer recovery loop:** if `validate` flags the result invalid, the graph
  routes back to `code_branch` for a full re-attempt (algorithm choice
  included), bounded by `MAX_RETRIES = 3`. When retries are exhausted, a
  report is still written (graceful degradation).

See [`GUIDE.md`](GUIDE.md) §12 for the full node-by-node table (DSPy module,
LM tier, state fields read/written).

## Installation

```bash
uv sync --extra report      # core + HTML/DOCX/PDF report writer
uv sync --all-extras        # everything (viz, report, ui, dev)
```

Create a `.env` file at the project root:

```bash
GOOGLE_API_KEY=your-api-key
IESOLVER_FAST_MODEL=gemini/gemini-2.0-flash
IESOLVER_REASONING_MODEL=gemini/gemini-2.0-flash
```

Any [LiteLLM](https://docs.litellm.ai/)-compatible model string works, e.g.
`openai/gpt-4o-mini` or `anthropic/claude-3-5-haiku-20241022`.

## Usage

```bash
# Terminal CLI
uv run python scripts/run_problem.py "EOQ: D=10000, S=50, H=2"
uv run python scripts/run_problem.py "EOQ" --data data/input.csv --format pdf
uv run python scripts/run_problem.py "EOQ" --show-llm-log   # inspect the prompts sent to the LLM
```

```python
# Python API
from iesolver import solve, stream_solve, show_llm_history
from iesolver.report import write_report

state = solve("EOQ: D=10000, S=50, H=2")
write_report(state, "report.html", format="html")

for node_name, partial in stream_solve("EOQ: D=10000, S=50, H=2"):
    print(node_name, list(partial.keys()))
```

Full walkthrough — problem authoring, step-by-step tracing, LLM history
inspection, prompt engineering workflow, experiment design — lives in
[`GUIDE.md`](GUIDE.md).

## Reproducibility & ablations

All LM calls use `temperature=0` and a fixed seed. Every run is checkpointed
to SQLite (resumable by `thread_id`). Five ablation flags isolate each
architectural component's contribution:

| Ablation | Mechanism | What it isolates |
|---|---|---|
| A1 | `solve(..., enable_refiner=False)` | PromptRefiner node contribution |
| A2 | `solve(..., enable_validator_retry=False)` | Retry loop contribution |
| A3 | `ie_eval.ablations.make_a3_correctness_fn()` | Deterministic validator layer contribution |
| A4 | `solve(..., fast_only=True)` | Fast vs. reasoning LM tier switching |
| A5 | `ie_eval.ablations.make_a5_solve(compiled_path)` | DSPy MIPROv2 prompt optimization |

## Prompt optimization (DSPy MIPROv2)

Signature prompts are compilable, not hand-frozen strings. `IESolverProgram`
(`src/iesolver/_optimization.py`) exposes the pipeline's DSPy module
singletons to MIPROv2, which searches over instruction text and few-shot
demonstrations against a labeled training split, scored end-to-end by the
final numerical answer:

```bash
uv run python scripts/optimize_mipro.py \
    --train-data data/nl4opt_train_cleaned.jsonl \
    --output compiled/iesolver_mipro.json \
    --max-train 40 --num-candidates 5 --num-trials 10
```

The compiled program loads back into the same live objects
(`load_compiled_graph`), so `solve()` transparently uses the optimized
prompts. See `GUIDE.md` §13 for the full mechanism (why singleton sharing
matters, what MIPROv2 does internally, cost estimates).

## Evaluation harness (`ie-eval`)

A separate `uv` workspace member for benchmark-scale evaluation:

```bash
cd ie-eval/
uv run python -m ieeval.runner --dataset datasets/nl4opt_sample.jsonl --config configs/full.yaml --out results/
```

Includes NL4Opt / IndustryOR dataset adapters, a custom 6-problem IE-Case
benchmark (EOQ, transportation, assignment, multi-product inventory,
job-shop-style, one NO_CODE conceptual question), a deterministic numerical
validator, a SQLite result store, single-shot/CoT baselines, and analysis
utilities (McNemar test, bootstrap CI — no `scipy` dependency).

## Project layout

```
src/iesolver/
  state.py            SolverState (TypedDict) + DataBundle
  graph.py             LangGraph DAG assembly + conditional routing
  lm.py                 Fast/reasoning LM context switching
  config.py            Pydantic settings (.env-backed)
  signatures/          13 DSPy Signatures — the system's prompts
  nodes/                11 LangGraph node functions (business logic)
  sandbox/              Isolated subprocess code execution
  report/               HTML / DOCX / PDF report writer
  observability/        Per-node token/cost/latency metrics
ie-eval/                 Benchmark evaluation harness (separate uv package)
scripts/                 CLI entry points (run_problem.py, optimize_mipro.py)
tests/                   iesolver test suite
```

## Testing

```bash
uv run pytest -m "not slow"     # fast suite, no live LLM calls
cd ie-eval && uv run pytest     # evaluation harness suite
```

## Documentation

| File | Contents |
|---|---|
| [`GUIDE.md`](GUIDE.md) | Researcher's guide: running the system, tracing execution, editing prompts, ablation flags, DSPy/MIPROv2 optimization walkthrough |
| [`PLAN.md`](PLAN.md) | Architecture decisions and the 5-phase build roadmap |
| [`METHODOLOGY_NOTES.md`](METHODOLOGY_NOTES.md) | Notes feeding the paper's methodology section |
| [`EVALUATION_PLAN.MD`](EVALUATION_PLAN.MD) | Experiment design: research questions, datasets, ablations, metrics |
| [`SYSTEM.md`](SYSTEM.md) | Full AI-oriented system documentation |
| [`MAKALE_YOL_HARITASI.md`](MAKALE_YOL_HARITASI.md) | Post-implementation roadmap to publication (Turkish) |

## Status

Phases 1–4.5 complete (foundation, DSPy signature transfer, code-generation
branch with sandboxed ReAct execution, sensitivity analysis + artifacts,
evaluation harness with A1–A5 ablations). Report writer (HTML/DOCX/PDF)
complete. Remaining work before the main benchmark runs: acquiring cleaned
NL4Opt/IndustryOR datasets, MIPROv2 optimization pass, and figure-generation
scripts for the paper's analysis section.

## License

MIT.
