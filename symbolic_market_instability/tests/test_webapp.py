"""Tests for the Flask dashboard API."""

import threading
import time

import pytest
import pandas as pd
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.run_manager import RunManager
from webapp.app import create_app


@pytest.fixture
def manager(tmp_path):
    return RunManager(results_dir=str(tmp_path))


def make_client(manager, executor=None):
    app = create_app(run_manager=manager, executor=executor)
    app.config['TESTING'] = True
    return app.test_client()


def fake_executor(ticker, start, end, run_manager, run_id):
    """Instant pipeline: writes results and finalizes the announced run."""
    dates = pd.date_range(start, periods=4, freq='D')
    df = pd.DataFrame({
        'instability_level': ['Low', 'Medium', 'High', 'Low'],
    }, index=pd.Index(dates, name='date'))
    run_manager.save_results(run_id, df)
    run_manager.finalize_run(run_id, summary={
        'days_analyzed': 4, 'high_days': 1, 'medium_days': 1, 'low_days': 2,
    })
    return run_id


# --- GET / ---

def test_index_renders(manager):
    client = make_client(manager)
    res = client.get('/')
    assert res.status_code == 200
    assert b'MARKET' in res.data


# --- GET /api/runs ---

def test_list_runs_empty(manager):
    client = make_client(manager)
    res = client.get('/api/runs')
    assert res.status_code == 200
    assert res.get_json() == {'runs': []}


def test_list_runs_populated(manager):
    run_id, _ = manager.create_run({'ticker': '^GSPC'})
    manager.finalize_run(run_id, summary={'days_analyzed': 5})
    client = make_client(manager)
    runs = client.get('/api/runs').get_json()['runs']
    assert len(runs) == 1
    assert runs[0]['run_id'] == run_id
    assert runs[0]['status'] == 'completed'


# --- GET /api/runs/<id> ---

def test_run_detail_includes_timeline_and_metrics(manager):
    run_id, _ = manager.create_run({})
    dates = pd.date_range('2020-01-01', periods=3, freq='D')
    manager.save_results(run_id, pd.DataFrame(
        {'instability_level': ['Low', 'High', 'Low']},
        index=pd.Index(dates, name='date')))
    manager.save_metrics(run_id, {'2008_crisis': {'lead_time_days': 8}})
    manager.finalize_run(run_id, summary={'days_analyzed': 3})

    client = make_client(manager)
    data = client.get(f'/api/runs/{run_id}').get_json()
    assert data['timeline']['levels'] == 'LHL'
    assert data['metrics']['2008_crisis']['lead_time_days'] == 8


def test_run_detail_unknown_404(manager):
    client = make_client(manager)
    assert client.get('/api/runs/nope').status_code == 404


# --- POST /api/runs ---

def test_trigger_run_executes_pipeline(manager):
    client = make_client(manager, executor=fake_executor)
    res = client.post('/api/runs', json={
        'ticker': '^GSPC', 'start_date': '2020-01-01', 'end_date': '2020-06-01',
    })
    assert res.status_code == 202
    run_id = res.get_json()['run_id']

    # background thread is near-instant with the fake executor
    for _ in range(50):
        metadata = manager.get_run(run_id)
        if metadata['status'] != 'running':
            break
        time.sleep(0.05)

    assert metadata['status'] == 'completed'
    assert metadata['summary']['days_analyzed'] == 4
    assert manager.load_timeline(run_id)['levels'] == 'LMHL'


@pytest.mark.parametrize('payload,expected_error', [
    ({'ticker': '', 'start_date': '2020-01-01', 'end_date': '2020-06-01'}, 'ticker'),
    ({'ticker': 'bad ticker!', 'start_date': '2020-01-01', 'end_date': '2020-06-01'}, 'ticker'),
    ({'ticker': '^GSPC', 'start_date': 'not-a-date', 'end_date': '2020-06-01'}, 'start_date'),
    ({'ticker': '^GSPC', 'start_date': '2020-06-01', 'end_date': '2020-01-01'}, 'before'),
    ({'ticker': '^GSPC', 'start_date': '2020-01-01', 'end_date': '2099-01-01'}, 'future'),
])
def test_trigger_run_invalid_params_400(manager, payload, expected_error):
    client = make_client(manager, executor=fake_executor)
    res = client.post('/api/runs', json=payload)
    assert res.status_code == 400
    assert any(expected_error in e for e in res.get_json()['errors'])


def test_trigger_run_concurrent_409(manager):
    release = threading.Event()

    def slow_executor(ticker, start, end, run_manager, run_id):
        release.wait(timeout=5)
        run_manager.finalize_run(run_id, summary={})
        return run_id

    client = make_client(manager, executor=slow_executor)
    first = client.post('/api/runs', json={
        'ticker': '^GSPC', 'start_date': '2020-01-01', 'end_date': '2020-06-01',
    })
    assert first.status_code == 202

    second = client.post('/api/runs', json={
        'ticker': '^GSPC', 'start_date': '2020-01-01', 'end_date': '2020-06-01',
    })
    assert second.status_code == 409
    assert second.get_json()['run_id'] == first.get_json()['run_id']

    release.set()


def test_trigger_run_executor_crash_marks_failed(manager):
    def crashing_executor(ticker, start, end, run_manager, run_id):
        raise RuntimeError('download exploded')

    client = make_client(manager, executor=crashing_executor)
    res = client.post('/api/runs', json={
        'ticker': '^GSPC', 'start_date': '2020-01-01', 'end_date': '2020-06-01',
    })
    run_id = res.get_json()['run_id']

    for _ in range(50):
        metadata = manager.get_run(run_id)
        if metadata['status'] != 'running':
            break
        time.sleep(0.05)

    assert metadata['status'] == 'failed'
    assert 'download exploded' in metadata['error']
