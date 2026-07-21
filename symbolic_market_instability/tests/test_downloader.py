"""Tests for MarketDataDownloader cache handling (no network required)."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_acquisition.downloader import MarketDataDownloader


@pytest.fixture
def downloader(tmp_path):
    return MarketDataDownloader(cache_dir=str(tmp_path))


def _write_cached_index(cache_dir, name, index):
    df = pd.DataFrame(
        {'Open': 1.0, 'High': 1.0, 'Low': 1.0, 'Close': 1.0, 'Volume': 100},
        index=index,
    )
    df.to_csv(Path(cache_dir) / name)


def _write_cached_vix(cache_dir, name, index):
    pd.Series(20.0, index=index, name='VIX').to_frame().to_csv(
        Path(cache_dir) / name
    )


def test_combined_data_survives_dst_spanning_cache(downloader, tmp_path):
    """
    A cached range crossing a daylight-saving change contains mixed UTC
    offsets; pandas re-reads those as strings, which used to make the
    index/VIX inner join empty. Regression test for that failure.
    """
    idx = pd.date_range('2020-02-24', periods=40, freq='B',
                        tz='America/New_York')
    _write_cached_index(tmp_path, '^GSPC_2020-01-15_2021-01-31.csv', idx)
    _write_cached_vix(tmp_path, 'VIX_US_2020-01-15_2021-01-31.csv', idx)

    combined = downloader.get_combined_data('^GSPC', '2020-01-15',
                                            '2021-01-31')

    assert len(combined) == 40
    assert 'VIX' in combined.columns
    assert combined.index.tz is None


def test_cached_index_is_tz_naive_midnights(downloader, tmp_path):
    idx = pd.date_range('2020-01-02', periods=5, freq='B',
                        tz='America/New_York')
    _write_cached_index(tmp_path, '^GSPC_2020-01-01_2020-01-10.csv', idx)

    data = downloader.download_index('^GSPC', '2020-01-01', '2020-01-10')

    assert data.index.tz is None
    assert (data.index == data.index.normalize()).all()
    assert list(data.index.date) == list(idx.tz_localize(None).date)


def test_cached_vix_returns_series(downloader, tmp_path):
    idx = pd.date_range('2020-01-02', periods=5, freq='B',
                        tz='America/New_York')
    _write_cached_vix(tmp_path, 'VIX_US_2020-01-01_2020-01-10.csv', idx)

    vix = downloader.download_vix('2020-01-01', '2020-01-10')

    assert isinstance(vix, pd.Series)
    assert len(vix) == 5
    assert vix.index.tz is None
