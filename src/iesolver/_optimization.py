"""
iesolver._optimization — MIPROv2 optimizasyon desteği (Ablation A5).

Bu modül scripts/optimize_mipro.py ve ie_eval.ablations.make_a5_solve
tarafından kullanılır; doğrudan çağrılmamalıdır.

Tasarım kararı — singleton referans paylaşımı:
    Her node (intake.py, requirement.py vb.) modül yüklenince bir DSPy
    singleton oluşturur (ör. ``_gatekeeper = dspy.ChainOfThought(...)``).
    ``IESolverProgram`` bu singleton'ları attribute olarak tutar:

        self.gatekeeper = _gatekeeper   # AYNI nesne

    MIPROv2, ``self.gatekeeper`` üzerinden prompt'u güncellediğinde
    ``call_with_fast_lm(_gatekeeper, ...)`` çağrısı da güncellenmiş
    prompt'u kullanır — iki ayrı nesne değil, aynı Python nesnesi.

    ``program.save(path)`` bu prompt/few-shot ağırlıklarını JSON'a yazar.
    ``program.load(path)`` onları geri yükler; graph tekrar derlenmez.
"""

from __future__ import annotations

import tempfile
import uuid
from pathlib import Path

import dspy


class IESolverProgram(dspy.Module):
    """All iesolver DSPy module singletons as one trainable program.

    MIPROv2 sub-modülleri keşfetmek için attribute traversal yapar.
    ``forward()`` iesolver.solve() pipeline'ını auto_mode=True ile çalıştırır.

    Attributes
    ----------
    gatekeeper : dspy.ChainOfThought
        GateKeeper — Phase 0
    analyst : dspy.Predict
        RequirementAnalyst — Phase 1
    refiner : dspy.ChainOfThought
        PromptRefiner — Phase 2
    router : dspy.ChainOfThought
        StrategyRouter — Phase 3
    selector : dspy.ChainOfThought
        AlgoSelector — Phase 4B.1
    adapter : dspy.Predict
        ConstraintAdapter — Phase 4B.2
    spec : dspy.Predict
        OutputSpecEngineer — Phase 4B.3
    react : dspy.ReAct
        ReActCodeGenerator — Phase 4B.4
    validator : dspy.ChainOfThought
        ResultValidator — Phase 4B.5
    sens_gen : dspy.ChainOfThought
        SensitivityAnalysis — Phase 4C
    reporter : dspy.ChainOfThought
        FinalReportGenerator — Phase 5
    """

    def __init__(self) -> None:
        super().__init__()
        # Her import, node modülündeki singleton'ı döndürür.
        # "from X import Y" her seferinde aynı nesneyi verir (Python module cache).
        from iesolver.nodes.intake import _gatekeeper
        from iesolver.nodes.requirement import _analyst
        from iesolver.nodes.refine import _refiner
        from iesolver.nodes.route import _router
        from iesolver.nodes.code_branch.algo_select import _selector
        from iesolver.nodes.code_branch.constraint_adapt import _adapter
        from iesolver.nodes.code_branch.output_spec import _spec
        from iesolver.nodes.code_branch.generate import _react
        from iesolver.nodes.validate import _validator
        from iesolver.nodes.sensitivity import _sens_gen
        from iesolver.nodes.report import _reporter

        self.gatekeeper = _gatekeeper
        self.analyst = _analyst
        self.refiner = _refiner
        self.router = _router
        self.selector = _selector
        self.adapter = _adapter
        self.spec = _spec
        self.react = _react
        self.validator = _validator
        self.sens_gen = _sens_gen
        self.reporter = _reporter

    def forward(self, prompt: str, data_path: Path | None = None) -> dspy.Prediction:
        """Run the iesolver pipeline; return execution_result for metric scoring.

        Her optimizasyon değerlendirmesi ayrı bir geçici checkpoint DB'de
        çalışır — paralel veya tekrarlı koşular birbiriyle çakışmaz.

        Parameters
        ----------
        prompt :
            Natural-language IE problem statement.
        data_path :
            Optional data file path.

        Returns
        -------
        dspy.Prediction
            ``execution_result``, ``is_valid``, ``executive_summary`` fields.
        """
        from iesolver import solve

        with tempfile.TemporaryDirectory(prefix="iesolver_mipro_") as tmpdir:
            state = solve(
                prompt,
                data_path=data_path,
                auto_mode=True,
                thread_id=f"mipro-{uuid.uuid4().hex[:8]}",
                checkpoint_db=Path(tmpdir) / "ckpt.sqlite",
            )

        return dspy.Prediction(
            execution_result=state.get("execution_result", "") or "",
            is_valid=bool(state.get("is_valid", False)),
            executive_summary=state.get("executive_summary", "") or "",
        )


def load_compiled_graph(compiled_path: Path) -> IESolverProgram:
    """Load a MIPROv2-compiled program and apply it to the live singletons.

    Adımlar:
        1. ``IESolverProgram()`` oluşturur (mevcut singleton'lara bağlı).
        2. ``program.load(path)`` ile prompt/few-shot ağırlıklarını yükler.
        3. Artık ``iesolver.solve()`` güncellenmiş prompt'larla çalışır.

    Parameters
    ----------
    compiled_path :
        Path to the JSON file saved by ``scripts/optimize_mipro.py``.

    Returns
    -------
    IESolverProgram
        Loaded program — same singleton references, updated prompts.

    Raises
    ------
    FileNotFoundError
        If ``compiled_path`` does not exist.
    """
    compiled_path = Path(compiled_path)
    if not compiled_path.exists():
        raise FileNotFoundError(
            f"Compiled program not found: {compiled_path}. "
            "Run `uv run python scripts/optimize_mipro.py` first."
        )
    program = IESolverProgram()
    program.load(str(compiled_path))
    return program


__all__ = ["IESolverProgram", "load_compiled_graph"]
