"""HTML renderer — Jinja2 + markdown-it-py."""

from __future__ import annotations

import base64
from pathlib import Path

from iesolver.state import SolverState


def _md(text: str) -> str:
    """Render markdown to HTML."""
    from markdown_it import MarkdownIt
    return MarkdownIt().render(text)


def _b64_figure(path: Path) -> str | None:
    if not path.exists():
        return None
    raw = path.read_bytes()
    ext = path.suffix.lstrip(".").lower() or "png"
    return f"data:image/{ext};base64,{base64.b64encode(raw).decode()}"


def write_html(state: SolverState, output_path: Path) -> Path:
    """Render *state* to a self-contained HTML file at *output_path*."""
    from jinja2 import BaseLoader, Environment

    tmpl_src = (Path(__file__).parent / "templates" / "report.html.j2").read_text(encoding="utf-8")
    env = Environment(loader=BaseLoader(), autoescape=False)
    template = env.from_string(tmpl_src)

    fig_data = [d for p in (state.get("figures") or []) if (d := _b64_figure(Path(p)))]

    html = template.render(
        goal=state.get("explicit_goal", "IE Problem") or "IE Problem",
        executive_summary=_md(state.get("executive_summary", "") or ""),
        technical_output=_md(state.get("technical_output", "") or ""),
        action_directives=_md(state.get("action_directives", "") or ""),
        sensitivity_results=_md(state.get("sensitivity_results", "") or ""),
        has_sensitivity=bool(state.get("sensitivity_results")),
        figures=fig_data,
        metrics=state.get("metrics") or {},
        is_valid=state.get("is_valid"),
        execution_result=state.get("execution_result", "") or "",
        generated_at=_now(),
    )
    output_path.write_text(html, encoding="utf-8")
    return output_path


def _now() -> str:
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M")
