"""Launch the live runs dashboard.

Usage:
    python scripts/run_dashboard.py [--port 5000]
"""

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from webapp.app import create_app


def main():
    parser = argparse.ArgumentParser(description="Market instability dashboard")
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=5000)
    args = parser.parse_args()

    app = create_app()
    print(f"Dashboard running at http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
