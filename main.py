#!/usr/bin/env python3
"""
Main entry point for the Bridge Monte Carlo Contract Simulator.

This script provides easy access to both CLI and web interfaces.
"""

import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def main():
    """Main entry point with interface selection."""
    if len(sys.argv) > 1 and sys.argv[1] == '--web':
        # Launch Streamlit web interface
        try:
            import streamlit.web.cli as stcli
            from bridge_simulator import streamlit_app
            
            # Remove the --web argument so streamlit doesn't see it
            sys.argv = [sys.argv[0]] + sys.argv[2:]
            
            # Run streamlit
            sys.argv = [
                "streamlit", "run", 
                str(project_root / "bridge_simulator" / "streamlit_app.py")
            ] + sys.argv[1:]
            
            stcli.main()
            
        except ImportError:
            print("Error: Streamlit not installed. Install with:")
            print("pip install streamlit pandas plotly")
            sys.exit(1)
    else:
        # Launch CLI interface
        from bridge_simulator.cli import cli_main
        cli_main()


if __name__ == "__main__":
    main() 