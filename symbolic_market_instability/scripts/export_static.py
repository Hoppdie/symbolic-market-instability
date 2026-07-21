"""Export the dashboard as a static, read-only site (for GitHub Pages).

Renders the live Flask page, embeds every archived run's data as
window.STATIC_DATA, and copies the bundled assets so the result works
with no server at all. The launch form is hidden automatically by app.js
when it detects static mode.

Usage:
    python scripts/export_static.py [--out site]
"""

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.run_manager import RunManager
from webapp.app import create_app

PROJECT_ROOT = Path(__file__).parent.parent


def build_static_data(manager: RunManager) -> dict:
    """Collect the run list plus per-run detail (metrics + timeline)."""
    runs = manager.list_runs()
    details = {}
    for run in runs:
        run_id = run['run_id']
        detail = dict(run)
        detail['metrics'] = manager.load_metrics(run_id)
        detail['timeline'] = manager.load_timeline(run_id)
        details[run_id] = detail
    return {'runs': runs, 'details': details}


def export(out_dir: Path, manager: Optional[RunManager] = None) -> Path:
    manager = manager or RunManager()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Render the page exactly as Flask serves it, then relativize asset URLs
    app = create_app(run_manager=manager)
    client = app.test_client()
    html = client.get('/').get_data(as_text=True)
    html = html.replace('/static/', 'static/')

    # Embed run data before app.js so it boots in static mode
    data = build_static_data(manager)
    payload = json.dumps(data, default=str)
    html = html.replace(
        '<script src="static/app.js"></script>',
        f'<script>window.STATIC_DATA = {payload};</script>\n'
        '<script src="static/app.js"></script>',
    )
    (out_dir / 'index.html').write_text(html)
    (out_dir / '.nojekyll').write_text('')

    # Bundle the assets (css, js, fonts)
    static_src = PROJECT_ROOT / 'webapp' / 'static'
    static_dst = out_dir / 'static'
    if static_dst.exists():
        shutil.rmtree(static_dst)
    shutil.copytree(static_src, static_dst)

    return out_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default=str(PROJECT_ROOT / 'site'),
                        help='Output directory (default: <project>/site)')
    args = parser.parse_args()

    out = export(Path(args.out))
    n_runs = len(build_static_data(RunManager())['runs'])
    print(f"Static dashboard exported to {out} ({n_runs} runs embedded)")


if __name__ == '__main__':
    main()
