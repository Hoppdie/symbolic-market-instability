"""Tests for RunManager."""

import json
import pytest
import pandas as pd
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.run_manager import RunManager


@pytest.fixture
def manager(tmp_path):
    return RunManager(results_dir=str(tmp_path))


@pytest.fixture
def results_df():
    dates = pd.date_range('2020-01-01', periods=6, freq='D')
    return pd.DataFrame({
        'instability_level': ['Low', 'Low', 'Medium', 'High', 'High', 'Low'],
        'close_price': [3000.0, 3010.0, 2990.0, 2900.0, 2850.0, 2950.0],
    }, index=pd.Index(dates, name='date'))


# --- create_run / get_run ---

def test_create_run_makes_directory_and_manifest(manager, tmp_path):
    run_id, run_dir = manager.create_run({'ticker': '^GSPC'})
    assert run_dir.exists()
    assert (run_dir / 'run_metadata.json').exists()
    metadata = manager.get_run(run_id)
    assert metadata['status'] == 'running'
    assert metadata['params']['ticker'] == '^GSPC'


def test_create_run_ids_unique_within_same_second(manager):
    ids = {manager.create_run({})[0] for _ in range(3)}
    assert len(ids) == 3


def test_get_run_unknown_returns_none(manager):
    assert manager.get_run('nope') is None


# --- finalize_run ---

def test_finalize_run_merges_summary(manager):
    run_id, _ = manager.create_run({'ticker': 'X'})
    metadata = manager.finalize_run(run_id, summary={'days_analyzed': 10})
    assert metadata['status'] == 'completed'
    assert metadata['summary']['days_analyzed'] == 10
    assert 'finished_at' in metadata
    # params from creation survive the merge
    assert metadata['params']['ticker'] == 'X'


def test_finalize_run_failed_records_error(manager):
    run_id, _ = manager.create_run({})
    metadata = manager.finalize_run(run_id, status='failed', error='boom')
    assert metadata['status'] == 'failed'
    assert metadata['error'] == 'boom'


def test_finalize_unknown_run_raises(manager):
    with pytest.raises(ValueError, match="Unknown run"):
        manager.finalize_run('missing')


# --- list_runs ---

def test_list_runs_newest_first(manager):
    first, _ = manager.create_run({})
    second, _ = manager.create_run({})
    ids = [r['run_id'] for r in manager.list_runs()]
    assert ids == sorted([first, second], reverse=True)


def test_list_runs_skips_corrupt_manifest(manager):
    good_id, _ = manager.create_run({})
    bad_dir = manager.runs_dir / 'corrupt_run'
    bad_dir.mkdir()
    (bad_dir / 'run_metadata.json').write_text('{not json')
    ids = [r['run_id'] for r in manager.list_runs()]
    assert ids == [good_id]


def test_list_runs_empty(manager):
    assert manager.list_runs() == []


# --- results / metrics / timeline ---

def test_save_and_load_timeline(manager, results_df):
    run_id, _ = manager.create_run({})
    manager.save_results(run_id, results_df)
    timeline = manager.load_timeline(run_id)
    assert timeline['levels'] == 'LLMHHL'
    assert timeline['start_date'] == '2020-01-01'
    assert timeline['end_date'] == '2020-01-06'


def test_load_timeline_missing_returns_none(manager):
    run_id, _ = manager.create_run({})
    assert manager.load_timeline(run_id) is None


def test_save_and_load_metrics(manager):
    run_id, _ = manager.create_run({})
    manager.save_metrics(run_id, {'2008_crisis': {'lead_time_days': 8}})
    metrics = manager.load_metrics(run_id)
    assert metrics['2008_crisis']['lead_time_days'] == 8


def test_load_metrics_missing_returns_none(manager):
    run_id, _ = manager.create_run({})
    assert manager.load_metrics(run_id) is None
