"""Tests for the generic crash evaluator and event registry."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from evaluation.case_studies import CaseStudyEvaluator, CRASH_EVENTS


@pytest.fixture
def evaluator(tmp_path):
    return CaseStudyEvaluator(results_dir=str(tmp_path))


def make_results(start, end, high_dates=()):
    idx = pd.date_range(start, end, freq='B')
    levels = ['High' if str(d.date()) in high_dates else 'Low' for d in idx]
    return pd.DataFrame({'instability_level': levels}, index=idx)


def test_registry_has_four_events():
    keys = {e['key'] for e in CRASH_EVENTS}
    assert keys == {'2000_dotcom_crash', '2008_crisis',
                    '2020_covid_crash', '2022_rate_hike_drawdown'}
    for e in CRASH_EVENTS:
        assert e['window_start'] < e['crash_date'] <= e['window_end']


def test_evaluate_crash_lead_time_and_counts(evaluator):
    results = make_results('2008-08-15', '2008-12-31',
                           high_dates=('2008-09-10', '2008-09-22'))
    m = evaluator.evaluate_crash(results, '2008-09-15',
                                 '2008-08-15', '2008-12-31',
                                 name='2008 FINANCIAL CRISIS')
    assert m['name'] == '2008 FINANCIAL CRISIS'
    assert m['lead_time_days'] == 5          # 09-10 -> 09-15
    assert m['warnings_before_crash'] == 1
    assert m['total_warnings'] == 2


def test_evaluate_crash_outside_range_returns_note(evaluator):
    results = make_results('2015-01-01', '2015-06-30')
    m = evaluator.evaluate_crash(results, '2008-09-15',
                                 '2008-08-15', '2008-12-31')
    assert 'note' in m
    assert m['total_warnings'] == 0


def test_evaluate_all_only_overlapping_events(evaluator):
    results = make_results('2019-01-01', '2021-01-01')
    metrics = evaluator.evaluate_all(results)
    assert list(metrics.keys()) == ['2020_covid_crash']


def test_evaluate_all_full_range_covers_all_events(evaluator):
    results = make_results('1999-01-01', '2023-01-01')
    metrics = evaluator.evaluate_all(results)
    assert set(metrics.keys()) == {e['key'] for e in CRASH_EVENTS}


def test_legacy_wrappers_still_work(evaluator):
    results = make_results('2008-08-15', '2008-12-31',
                           high_dates=('2008-09-10',))
    legacy = evaluator.evaluate_2008_crisis(results)
    assert legacy['crash_date'] == '2008-09-15'
    assert legacy['lead_time_days'] == 5

    covid = evaluator.evaluate_covid_crash(
        make_results('2020-02-11', '2020-04-30', high_dates=('2020-03-05',)))
    assert covid['crash_date'] == '2020-03-12'
    assert covid['lead_time_days'] == 7
