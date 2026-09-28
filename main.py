#!/usr/bin/env python3
"""Launch the command-line or Streamlit interface."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent


def main() -> None:
    """Run the requested interface."""
    if "--web" not in sys.argv[1:]:
        from bridge_simulator.cli import cli_main

        cli_main()
        return

    try:
        import streamlit.web.cli as streamlit_cli
    except ImportError as exc:
        print("Error: Streamlit is not installed. Run: pip install -r requirements.txt")
        raise SystemExit(1) from exc

    extra_args = [arg for arg in sys.argv[1:] if arg != "--web"]
    app_path = PROJECT_ROOT / "bridge_simulator" / "streamlit_app.py"
    sys.argv = ["streamlit", "run", str(app_path), *extra_args]
    streamlit_cli.main()


if __name__ == "__main__":
    main()
