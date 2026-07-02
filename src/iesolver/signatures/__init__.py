"""Pure DSPy Signature contracts.

Bu paket eski ``signatures.py`` dosyasının 11 sınıfını ayrı dosyalara
böler. Her dosya yalnızca bir ``dspy.Signature`` sınıfı barındırır:
docstring (prompt) + ``InputField`` / ``OutputField`` deklarasyonları.

Mimari karar — neden bu kadar dar dosyalar?
    Signature'lar makalede ayrı ayrı analiz, optimize ve test
    edilecek "atomic prompts". Tek tek dosyada olmaları, ileride
    ``dspy.MIPRO`` / ``dspy.BootstrapFewShot`` gibi optimizer'lar
    tek tek koşturulurken hangi prompt'un değiştiğini diff'lemeyi
    kolaylaştırır. CoT vs Predict tercihi burada DEĞİL, node'larda
    yapılır (bkz. ``iesolver/nodes/``).
"""

from iesolver.signatures.algo_selector import AlgoSelectorSignature
from iesolver.signatures.analytical_engine import AnalyticalEngineSignature
from iesolver.signatures.constraint_adapter import ConstraintAdapterSignature
from iesolver.signatures.final_report import FinalReportSignature
from iesolver.signatures.gatekeeper import GateKeeperSignature
from iesolver.signatures.output_spec import OutputSpecEngineerSignature
from iesolver.signatures.prompt_refiner import PromptRefinerSignature
from iesolver.signatures.react_code import ReActCodeSignature
from iesolver.signatures.requirement_analyst import RequirementAnalystSignature
from iesolver.signatures.sensitivity import SensitivityCodeSignature
from iesolver.signatures.strategy_router import StrategyRouterSignature
from iesolver.signatures.tornado_chart import TornadoChartSignature
from iesolver.signatures.validator import ResultValidatorSignature

__all__ = [
    "AlgoSelectorSignature",
    "AnalyticalEngineSignature",
    "ConstraintAdapterSignature",
    "FinalReportSignature",
    "GateKeeperSignature",
    "OutputSpecEngineerSignature",
    "PromptRefinerSignature",
    "ReActCodeSignature",
    "RequirementAnalystSignature",
    "ResultValidatorSignature",
    "SensitivityCodeSignature",
    "StrategyRouterSignature",
    "TornadoChartSignature",
]
