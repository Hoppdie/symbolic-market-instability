"""Visualization utilities."""

from typing import Optional, List
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def plot_features(features: pd.DataFrame, save_path: Optional[str] = None) -> None:
    """
    Plot all features over time.
    
    Args:
        features: DataFrame with feature columns
        save_path: Path to save figure. If None, saves to results/figures/
    """
    feature_cols = ['rolling_return', 'rolling_volatility', 'price_slope', 
                    'volume_trend', 'vix']
    
    fig, axes = plt.subplots(len(feature_cols), 1, figsize=(14, 12))
    
    for i, col in enumerate(feature_cols):
        if col in features.columns:
            axes[i].plot(features.index, features[col], linewidth=1.5, alpha=0.7)
            axes[i].set_ylabel(col.replace('_', ' ').title(), fontsize=10)
            axes[i].grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Date', fontsize=12)
    fig.suptitle('Technical Features Over Time', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path is None:
        # Go up from src/utils/visualization.py -> src/utils -> src -> symbolic_market_instability
        project_root = Path(__file__).parent.parent.parent
        save_path = project_root / "results" / "figures" / "features.png"
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_symbols_over_time(
    symbolized: pd.DataFrame, 
    save_path: Optional[str] = None
) -> None:
    """
    Plot active symbols over time.
    
    Args:
        symbolized: DataFrame with 'symbols' column (Set[str] per row)
        save_path: Path to save figure
    """
    # Get all unique symbols
    all_symbols = set()
    for symbol_set in symbolized['symbols']:
        if isinstance(symbol_set, set):
            all_symbols.update(symbol_set)
    
    # Create binary matrix: 1 if symbol is active, 0 otherwise
    symbol_matrix = pd.DataFrame(index=symbolized.index, columns=sorted(all_symbols))
    for date, symbol_set in symbolized['symbols'].items():
        if isinstance(symbol_set, set):
            for symbol in symbol_set:
                symbol_matrix.loc[date, symbol] = 1
    
    symbol_matrix = symbol_matrix.fillna(0)
    
    # Plot heatmap
    fig, ax = plt.subplots(figsize=(14, max(8, len(all_symbols) * 0.3)))
    im = ax.imshow(symbol_matrix.T, aspect='auto', cmap='YlOrRd', interpolation='nearest')
    
    ax.set_yticks(range(len(all_symbols)))
    ax.set_yticklabels(symbol_matrix.columns)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_title('Active Symbols Over Time', fontsize=14, fontweight='bold')
    
    # Format x-axis dates
    n_dates = len(symbolized)
    step = max(1, n_dates // 10)
    ax.set_xticks(range(0, n_dates, step))
    ax.set_xticklabels([str(d)[:10] for d in symbolized.index[::step]], rotation=45)
    
    plt.colorbar(im, ax=ax, label='Active')
    plt.tight_layout()
    
    if save_path is None:
        # Go up from src/utils/visualization.py -> src/utils -> src -> symbolic_market_instability
        project_root = Path(__file__).parent.parent.parent
        save_path = project_root / "results" / "figures" / "symbols_over_time.png"
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_instability_timeline(
    results: pd.DataFrame, 
    crash_dates: List[str],
    save_path: Optional[str] = None
) -> None:
    """
    Plot instability timeline with crash dates marked.
    
    Args:
        results: DataFrame with instability_level column
        crash_dates: List of crash dates in 'YYYY-MM-DD' format
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Map instability levels to numeric values
    level_map = {'Low': 1, 'Medium': 2, 'High': 3}
    results['instability_numeric'] = results['instability_level'].map(level_map)
    
    # Plot instability level over time
    ax.plot(results.index, results['instability_numeric'], 
           linewidth=2, alpha=0.8, label='Instability Level', color='blue')
    
    # Mark crash dates
    for crash_date in crash_dates:
        crash_idx = pd.to_datetime(crash_date)
        if crash_idx >= results.index.min() and crash_idx <= results.index.max():
            ax.axvline(x=crash_idx, color='red', linestyle='--', 
                      linewidth=2, alpha=0.7)
            ax.text(crash_idx, ax.get_ylim()[1] * 0.95, crash_date, 
                   rotation=90, verticalalignment='top', fontsize=9)
    
    ax.set_ylabel('Instability Level', fontsize=12)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_title('Market Instability Timeline', fontsize=14, fontweight='bold')
    ax.set_yticks([1, 2, 3])
    ax.set_yticklabels(['Low', 'Medium', 'High'])
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    
    if save_path is None:
        # Go up from src/utils/visualization.py -> src/utils -> src -> symbolic_market_instability
        project_root = Path(__file__).parent.parent.parent
        save_path = project_root / "results" / "figures" / "instability_timeline.png"
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
