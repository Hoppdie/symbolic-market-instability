"""Run persistence: each pipeline execution saved as a timestamped run."""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

LEVEL_CODES = {'Low': 'L', 'Medium': 'M', 'High': 'H'}


class RunManager:
    """Manages run directories under results/runs/<run_id>/."""

    def __init__(self, results_dir: Optional[str] = None):
        """
        Args:
            results_dir: Directory holding runs/. Defaults to <project>/results.
        """
        if results_dir is None:
            project_root = Path(__file__).parent.parent.parent
            results_dir = project_root / "results"

        self.results_dir = Path(results_dir)
        self.runs_dir = self.results_dir / "runs"
        self.runs_dir.mkdir(parents=True, exist_ok=True)

    def _run_dir(self, run_id: str) -> Path:
        return self.runs_dir / run_id

    def _metadata_file(self, run_id: str) -> Path:
        return self._run_dir(run_id) / "run_metadata.json"

    def create_run(self, params: Dict) -> Tuple[str, Path]:
        """
        Create a new run directory with an initial metadata manifest.

        Args:
            params: Run parameters (ticker, start_date, end_date, ...)

        Returns:
            Tuple of (run_id, run_dir)
        """
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = self._run_dir(run_id)
        # Guard against two runs created within the same second
        suffix = 1
        while run_dir.exists():
            run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{suffix}"
            run_dir = self._run_dir(run_id)
            suffix += 1
        run_dir.mkdir(parents=True)

        metadata = {
            'run_id': run_id,
            'status': 'running',
            'created_at': datetime.now().isoformat(timespec='seconds'),
            'params': params,
        }
        self._write_metadata(run_id, metadata)
        return run_id, run_dir

    def finalize_run(
        self,
        run_id: str,
        summary: Optional[Dict] = None,
        status: str = 'completed',
        error: Optional[str] = None,
    ) -> Dict:
        """
        Merge summary stats into the manifest and set final status.

        Args:
            run_id: Run to finalize
            summary: Stats to merge (day counts, metrics, duration...)
            status: 'completed' or 'failed'
            error: Error message when status is 'failed'

        Returns:
            The updated metadata dictionary
        """
        metadata = self.get_run(run_id)
        if metadata is None:
            raise ValueError(f"Unknown run: {run_id}")

        metadata['status'] = status
        metadata['finished_at'] = datetime.now().isoformat(timespec='seconds')
        if summary:
            metadata['summary'] = summary
        if error:
            metadata['error'] = error

        self._write_metadata(run_id, metadata)
        return metadata

    def list_runs(self) -> List[Dict]:
        """
        List all runs, newest first. Corrupt or missing manifests are skipped.
        """
        runs = []
        for manifest in self.runs_dir.glob("*/run_metadata.json"):
            try:
                with open(manifest, 'r') as f:
                    runs.append(json.load(f))
            except (json.JSONDecodeError, OSError):
                continue
        runs.sort(key=lambda r: r.get('run_id', ''), reverse=True)
        return runs

    def get_run(self, run_id: str) -> Optional[Dict]:
        """Return one run's metadata, or None if it doesn't exist."""
        manifest = self._metadata_file(run_id)
        if not manifest.exists():
            return None
        try:
            with open(manifest, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return None

    def save_results(self, run_id: str, results_df: pd.DataFrame) -> Path:
        """Save the analysis results CSV into the run directory."""
        path = self._run_dir(run_id) / "analysis_results.csv"
        results_df.to_csv(path)
        return path

    def save_metrics(self, run_id: str, metrics: Dict) -> Path:
        """Save evaluation metrics JSON into the run directory."""
        path = self._run_dir(run_id) / "evaluation_metrics.json"
        with open(path, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
        return path

    def load_metrics(self, run_id: str) -> Optional[Dict]:
        """Load evaluation metrics for a run, or None if absent."""
        path = self._run_dir(run_id) / "evaluation_metrics.json"
        if not path.exists():
            return None
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return None

    def load_timeline(self, run_id: str) -> Optional[Dict]:
        """
        Load a compact timeline for a run from its analysis_results.csv.

        Returns:
            Dict with 'start_date', 'end_date', 'levels' (one char per day:
            L/M/H), and 'dates' omitted to stay small — or None if no results.
        """
        path = self._run_dir(run_id) / "analysis_results.csv"
        if not path.exists():
            return None

        df = pd.read_csv(path, index_col=0, parse_dates=True,
                         usecols=lambda c: c in ('date', 'instability_level'))
        if df.empty or 'instability_level' not in df.columns:
            return None

        levels = ''.join(LEVEL_CODES.get(v, 'L') for v in df['instability_level'])
        return {
            'start_date': str(df.index.min().date()),
            'end_date': str(df.index.max().date()),
            'levels': levels,
        }

    def _write_metadata(self, run_id: str, metadata: Dict) -> None:
        # Atomic replace: the webapp polls this file from another thread
        # while the pipeline updates it, so readers must never see a
        # half-written manifest.
        target = self._metadata_file(run_id)
        tmp = target.with_suffix('.json.tmp')
        with open(tmp, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        os.replace(tmp, target)
