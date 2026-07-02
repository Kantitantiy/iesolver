"""Dataset protokolü — bir Problem koleksiyonu döndüren her nesne uyar."""

from __future__ import annotations

from typing import Iterable, Protocol, runtime_checkable

from ie_eval.problem import Problem


@runtime_checkable
class Dataset(Protocol):
    """Read-only sequence of ``Problem``s with a stable name.

    Adaptörler (NL4Opt, IndustryOR, IE-Case) bu protokolü uygular. Runner
    yalnızca ``load()``'u çağırır, veri kaynağına bakmaz.
    """

    name: str

    def load(self) -> Iterable[Problem]:
        """Yield problems one by one; iteration must be idempotent."""
        ...