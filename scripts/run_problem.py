"""
run_problem.py — iesolver için interaktif CLI çalıştırıcı.

Kullanım:
    uv run python scripts/run_problem.py "EOQ: D=10000, S=50, H=2"
    uv run python scripts/run_problem.py "EOQ problemi" --data data/orders.csv
    uv run python scripts/run_problem.py "LP problemi" --show-llm-log
    uv run python scripts/run_problem.py "..." --format pdf --out rapor.pdf
    uv run python scripts/run_problem.py "..." --auto   # interrupt olmadan
    uv run python scripts/run_problem.py "..." --no-refiner  # A1 ablasyon
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from datetime import datetime
from pathlib import Path

# Windows terminali cp1252 kullanabilir; UTF-8'e zorla.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# v4/src altında çalışabilmesi için sys.path ayarla
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="run_problem",
        description="iesolver — IE probleminizi LLM ile çözün.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Örnekler:
              uv run python scripts/run_problem.py "EOQ: D=10000, S=50, H=2"
              uv run python scripts/run_problem.py "Üretim planlama LP" --show-llm-log
              uv run python scripts/run_problem.py "Stok analizi" --data data/inv.csv --format pdf
        """),
    )
    p.add_argument("prompt", help="Çözmek istediğiniz IE problemi (doğal dilde)")
    p.add_argument("--data", metavar="PATH", help="Veri dosyası (.csv / .xlsx / .sqlite)")
    p.add_argument(
        "--format",
        choices=["html", "docx", "pdf"],
        default="html",
        help="Rapor formatı (default: html)",
    )
    p.add_argument("--out", metavar="PATH", help="Rapor çıktı yolu (default: otomatik)")
    p.add_argument(
        "--auto",
        action="store_true",
        help="Eksik bilgiler için interrupt olmadan devam et (batch mod)",
    )
    p.add_argument(
        "--show-llm-log",
        action="store_true",
        help="Çalıştırma sonrası LLM konuşma geçmişini göster",
    )
    p.add_argument(
        "--llm-log-n",
        type=int,
        default=5,
        metavar="N",
        help="Kaç LLM çağrısı gösterilsin (default: 5, --show-llm-log ile birlikte)",
    )
    # Ablasyon bayrakları
    p.add_argument("--no-refiner", action="store_true", help="A1: PromptRefiner'ı devre dışı bırak")
    p.add_argument("--no-retry", action="store_true", help="A2: Validator retry döngüsünü devre dışı bırak")
    p.add_argument("--fast-only", action="store_true", help="A4: Tüm LLM çağrıları için hızlı modeli kullan")
    return p.parse_args()


def _default_out(fmt: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    reports_dir = _ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    return reports_dir / f"report_{ts}.{fmt}"


NODE_LABELS: dict[str, str] = {
    "intake":            "Girdi analizi",
    "requirement_analyst": "Gereksinim analizi",
    "refiner":           "Prompt iyileştirme",
    "strategy_router":   "Strateji seçimi",
    "algo_select":       "Algoritma seçimi",
    "constraint_adapt":  "Kısıt adaptasyonu",
    "output_spec":       "Çıktı tanımı",
    "generate":          "Kod üretimi (ReAct)",
    "execute":           "Kod çalıştırma",
    "validate":          "Sonuç doğrulama",
    "sensitivity":       "Duyarlılık analizi",
    "report":            "Rapor derleme",
}


def _node_label(name: str) -> str:
    return NODE_LABELS.get(name, name)


def _print_partial(node_name: str, partial: dict) -> None:
    label = _node_label(node_name)
    keys = [k for k in partial if not k.startswith("_")]
    print(f"  ✓ {label:<30}  [{', '.join(keys)}]")


def main() -> None:
    args = _parse_args()

    try:
        from iesolver import show_llm_history, stream_solve
        from iesolver.report import write_report
    except ImportError as e:
        print(f"[HATA] iesolver paketi yüklenemedi: {e}")
        print("       'uv sync --extra report' komutunu çalıştırın.")
        sys.exit(1)

    out_path = Path(args.out) if args.out else _default_out(args.format)

    print()
    print("=" * 60)
    print("  iesolver — IE Problem Çözücü")
    print("=" * 60)
    print(f"  Problem : {args.prompt[:80]}")
    if args.data:
        print(f"  Veri    : {args.data}")
    print(f"  Rapor   : {out_path}")
    print(f"  Mod     : {'otomatik' if args.auto else 'interaktif'}")
    print("=" * 60)
    print()
    print("Çözüm aşamaları:")
    print()

    final_state: dict = {}
    interrupted = False

    try:
        for node_name, partial in stream_solve(
            prompt=args.prompt,
            data_path=args.data,
            auto_mode=args.auto,
            enable_refiner=not args.no_refiner,
            enable_validator_retry=not args.no_retry,
            fast_only=args.fast_only,
        ):
            if node_name == "__interrupt__":
                interrupted = True
                questions = partial if isinstance(partial, list) else [partial]
                print()
                print("  [?] Program ek bilgiye ihtiyaç duyuyor:")
                for q in questions:
                    value = q.value if hasattr(q, "value") else str(q)
                    print(f"      {value}")
                print()
                print("  (Otomatik mod için --auto bayrağını kullanın)")
                break
            _print_partial(node_name, partial)
            final_state.update(partial)

    except KeyboardInterrupt:
        print("\n  [!] Kullanıcı tarafından durduruldu.")
        sys.exit(0)

    if interrupted:
        print("[!] Çalıştırma interrupt ile durdu. Rapor oluşturulmadı.")
        sys.exit(0)

    print()

    # Özet
    goal = final_state.get("explicit_goal") or args.prompt[:60]
    result = final_state.get("execution_result") or ""
    is_valid = final_state.get("is_valid")
    valid_str = "DOĞRULANDI" if is_valid else ("DOĞRULANAMADI" if is_valid is False else "—")

    print("─" * 60)
    print(f"  Hedef      : {goal}")
    if result:
        print(f"  Sonuç      : {result[:100]}")
    print(f"  Doğrulama  : {valid_str}")
    print("─" * 60)
    print()

    # Rapor
    try:
        write_report(final_state, out_path, format=args.format)
        print(f"  Rapor kaydedildi: {out_path}")
    except Exception as e:
        print(f"  [UYARI] Rapor oluşturulamadı: {e}")

    # LLM geçmişi
    if args.show_llm_log:
        print()
        print("─" * 60)
        print("  LLM Konuşma Geçmişi")
        print("─" * 60)
        show_llm_history(n=args.llm_log_n)

    print()


if __name__ == "__main__":
    main()
