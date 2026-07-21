"""Case study evaluation for historical crashes."""

from typing import Dict, List, Optional
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


class CaseStudyEvaluator:
    """Evaluates system performance on historical crash events."""
    
    def __init__(self, results_dir: Optional[str] = None):
        """
        Initialize evaluator.
        
        Args:
            results_dir: Directory to save results. Defaults to results/
        """
        if results_dir is None:
            # Go up from src/evaluation/case_studies.py -> src/evaluation -> src -> symbolic_market_instability
            project_root = Path(__file__).parent.parent.parent
            results_dir = project_root / "results"
        
        self.results_dir = Path(results_dir)
        self.figures_dir = self.results_dir / "figures"
        self.figures_dir.mkdir(parents=True, exist_ok=True)
    
    def evaluate_2008_crisis(self, results: pd.DataFrame) -> Dict:
        """
        Evaluate performance on 2008 financial crisis.
        
        Crash date: 2008-09-15 (Lehman Brothers bankruptcy)
        Warning window: 30 days before crash
        
        Args:
            results: DataFrame with columns: date, instability_level, ...
        
        Returns:
            Dictionary with evaluation metrics
        """
        crash_date = '2008-09-15'
        warning_window_start = pd.to_datetime('2008-08-15')
        warning_window_end = pd.to_datetime('2008-09-15')
        
        # Work with a copy and normalize index timezone
        results = results.copy()
        # Ensure index is DatetimeIndex and timezone-naive
        if not isinstance(results.index, pd.DatetimeIndex):
            results.index = pd.to_datetime(results.index, utc=True)
        if results.index.tz is not None:
            results.index = results.index.tz_localize(None)
        
        # Filter using timezone-naive dates
        warning_window_start = pd.Timestamp('2008-08-15')
        warning_window_end = pd.Timestamp('2008-12-31')
        results_filtered = results[
            (results.index >= warning_window_start) & 
            (results.index <= warning_window_end)
        ].copy()
        
        if results_filtered.empty:
            return {
                'crash_date': crash_date,
                'lead_time': None,
                'precision': None,
                'recall': None,
                'warnings_before_crash': 0,
                'total_warnings': 0,
                'note': 'No data available for 2008 crisis period'
            }
        
        # Create warning series (High or Medium instability)
        warnings = (results_filtered['instability_level'].isin(['High', 'Medium']))
        
        # Compute metrics
        from .metrics import compute_lead_time, compute_precision, compute_recall
        
        lead_time = compute_lead_time(warnings, crash_date)
        precision = compute_precision(
            warnings, 
            (warning_window_start, warning_window_end)
        )
        recall = compute_recall(
            warnings,
            (warning_window_start, warning_window_end)
        )
        
        crash_date_ts = pd.to_datetime(crash_date)
        warnings_before = warnings[warnings.index < crash_date_ts].sum()
        total_warnings = warnings.sum()
        
        return {
            'crash_date': crash_date,
            'lead_time_days': lead_time,
            'precision': precision,
            'recall': recall,
            'warnings_before_crash': int(warnings_before),
            'total_warnings': int(total_warnings),
            'false_positive_rate': 1.0 - precision if precision > 0 else None
        }
    
    def evaluate_covid_crash(self, results: pd.DataFrame) -> Dict:
        """
        Evaluate performance on 2020 COVID-19 crash.
        
        Crash date: 2020-03-12 (Black Thursday)
        Warning window: 30 days before crash
        
        Args:
            results: DataFrame with columns: date, instability_level, ...
        
        Returns:
            Dictionary with evaluation metrics
        """
        crash_date = '2020-03-12'
        warning_window_start = pd.to_datetime('2020-02-11')
        warning_window_end = pd.to_datetime('2020-03-12')
        
        # Work with a copy and normalize index timezone
        results = results.copy()
        # Ensure index is DatetimeIndex and timezone-naive
        if not isinstance(results.index, pd.DatetimeIndex):
            results.index = pd.to_datetime(results.index, utc=True)
        if results.index.tz is not None:
            results.index = results.index.tz_localize(None)
        
        # Filter using timezone-naive dates
        warning_window_start = pd.Timestamp('2020-02-11')
        warning_window_end = pd.Timestamp('2020-04-30')
        results_filtered = results[
            (results.index >= warning_window_start) & 
            (results.index <= warning_window_end)
        ].copy()
        
        if results_filtered.empty:
            return {
                'crash_date': crash_date,
                'lead_time': None,
                'precision': None,
                'recall': None,
                'warnings_before_crash': 0,
                'total_warnings': 0,
                'note': 'No data available for 2020 COVID crash period'
            }
        
        # Create warning series
        warnings = (results_filtered['instability_level'].isin(['High', 'Medium']))
        
        # Compute metrics
        from .metrics import compute_lead_time, compute_precision, compute_recall
        
        lead_time = compute_lead_time(warnings, crash_date)
        precision = compute_precision(
            warnings, 
            (warning_window_start, warning_window_end)
        )
        recall = compute_recall(
            warnings,
            (warning_window_start, warning_window_end)
        )
        
        crash_date_ts = pd.to_datetime(crash_date)
        warnings_before = warnings[warnings.index < crash_date_ts].sum()
        total_warnings = warnings.sum()
        
        return {
            'crash_date': crash_date,
            'lead_time_days': lead_time,
            'precision': precision,
            'recall': recall,
            'warnings_before_crash': int(warnings_before),
            'total_warnings': int(total_warnings),
            'false_positive_rate': 1.0 - precision if precision > 0 else None
        }
    
    def plot_timeline(
        self, 
        results: pd.DataFrame, 
        crash_dates: List[str],
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot instability timeline with crash dates marked.
        
        Args:
            results: DataFrame with instability_level column
            crash_dates: List of crash dates in 'YYYY-MM-DD' format
            save_path: Path to save figure. If None, saves to results/figures/
        """
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Map instability levels to numeric values
        level_map = {'Low': 1, 'Medium': 2, 'High': 3}
        results['instability_numeric'] = results['instability_level'].map(level_map)
        
        # Plot instability level over time
        ax.plot(results.index, results['instability_numeric'], 
               linewidth=1.5, alpha=0.7, label='Instability Level')
        
        # Mark crash dates
        for crash_date in crash_dates:
            crash_idx = pd.to_datetime(crash_date)
            if crash_idx in results.index:
                ax.axvline(x=crash_idx, color='red', linestyle='--', 
                          linewidth=2, alpha=0.7, label=f'Crash: {crash_date}')
        
        ax.set_ylabel('Instability Level', fontsize=12)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_title('Market Instability Timeline', fontsize=14, fontweight='bold')
        ax.set_yticks([1, 2, 3])
        ax.set_yticklabels(['Low', 'Medium', 'High'])
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.figures_dir / "instability_timeline.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
