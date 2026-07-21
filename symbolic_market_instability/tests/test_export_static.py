"""Tests for the static dashboard export (GitHub Pages build)."""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.run_manager import RunManager
from scripts.export_static import build_static_data, export


@pytest.fixture
def manager(tmp_path):
    return RunManager(results_dir=str(tmp_path / "results"))


@pytest.fixture
def seeded_manager(manager):
    run_id, _ = manager.create_run({
        'ticker': '^GSPC', 'start_date': '2020-01-01',
        'end_date': '2020-01-10',
    })
    dates = pd.date_range('2020-01-01', periods=6, freq='D')
    df = pd.DataFrame({'instability_level':
                       ['Low', 'Low', 'Medium', 'High', 'High', 'Low']},
                      index=pd.Index(dates, name='date'))
    manager.save_results(run_id, df)
    manager.save_metrics(run_id, {'2020_covid_crash': {'lead_time_days': 0}})
    manager.finalize_run(run_id, summary={'days_analyzed': 6, 'high_days': 2,
                                          'medium_days': 1, 'low_days': 3})
    return manager


def test_build_static_data(seeded_manager):
    data = build_static_data(seeded_manager)
    assert len(data['runs']) == 1
    run_id = data['runs'][0]['run_id']
    detail = data['details'][run_id]
    assert detail['timeline']['levels'] == 'LLMHHL'
    assert detail['metrics']['2020_covid_crash']['lead_time_days'] == 0


def test_export_writes_selfcontained_site(seeded_manager, tmp_path):
    out = export(tmp_path / "site", manager=seeded_manager)

    html = (out / 'index.html').read_text()
    assert 'window.STATIC_DATA' in html
    assert '"levels": "LLMHHL"' in html
    assert '/static/' not in html          # all asset URLs relativized
    assert (out / '.nojekyll').exists()
    assert (out / 'static' / 'style.css').exists()
    assert (out / 'static' / 'app.js').exists()
    assert any((out / 'static' / 'fonts').glob('*.woff2'))


def test_export_with_no_runs(manager, tmp_path):
    out = export(tmp_path / "site", manager=manager)
    html = (out / 'index.html').read_text()
    assert 'window.STATIC_DATA' in html
    assert json.loads(
        html.split('window.STATIC_DATA = ')[1].split(';</script>')[0]
    ) == {'runs': [], 'details': {}}
