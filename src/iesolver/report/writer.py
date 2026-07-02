"""
iesolver.report.writer — format dispatcher (HTML / DOCX / PDF).

Kullanım::

    from iesolver import solve, write_report

    state = solve("EOQ problem ...", auto_mode=True)
    html_path = write_report(state, "output/report.html", format="html")
    docx_path = write_report(state, "output/report.docx", format="docx")
    pdf_path  = write_report(state, "output/report.pdf",  format="pdf")
"""

from __future__ import annotations

from pathlib import Path

from iesolver.state import OutputFormat, SolverState


class ReportWriter:
    """Dispatch *state* → file for the requested *format*.

    Parameters
    ----------
    state :
        Completed ``SolverState`` from ``iesolver.solve()``.
    """

    def __init__(self, state: SolverState) -> None:
        self._state = state

    def write(
        self,
        output_path: Path | str,
        format: OutputFormat = "html",
    ) -> Path:
        """Write the report and return the resolved output path.

        Parameters
        ----------
        output_path :
            Destination file.  Parent directories are created automatically.
        format :
            ``"html"`` | ``"docx"`` | ``"pdf"``
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if format == "html":
            from iesolver.report._html import write_html
            return write_html(self._state, path)
        if format == "docx":
            from iesolver.report._docx import write_docx
            return write_docx(self._state, path)
        if format == "pdf":
            from iesolver.report._pdf import write_pdf
            return write_pdf(self._state, path)

        raise ValueError(f"Unsupported format {format!r}. Choose 'html', 'docx', or 'pdf'.")


def write_report(
    state: SolverState,
    output_path: Path | str,
    format: OutputFormat = "html",
) -> Path:
    """Convenience wrapper around :class:`ReportWriter`.

    Parameters
    ----------
    state :
        Completed ``SolverState`` from ``iesolver.solve()``.
    output_path :
        Destination file path.
    format :
        ``"html"`` | ``"docx"`` | ``"pdf"``

    Returns
    -------
    Path
        Resolved path of the written file.
    """
    return ReportWriter(state).write(output_path, format)
