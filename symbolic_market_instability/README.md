# Symbolic Reasoning Models for Stock Market Instability

## Overview

This project implements a theory-driven symbolic AI system for detecting stock market instability. The system uses forward-chaining inference with economic theory rules (based on Minsky's Financial Instability Hypothesis) to provide interpretable and explainable early-warning signals for market crashes.

**Key Features:**
- 100% interpretable (no black boxes)
- Theory-driven rules (Minsky's Financial Instability Hypothesis)
- Explainable outputs with inference chains
- Historical validation on 2008 and 2020 crashes
- No machine learning frameworks (pure symbolic reasoning)

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### 1. Calibrate Thresholds

First, download historical data and calibrate feature thresholds:

```bash
python scripts/run_calibration.py
```

This will:
- Download S&P 500 data from 2000-2015 (calibration period)
- Compute technical features
- Calibrate percentile thresholds
- Save thresholds to `config/thresholds.json`

### 2. Run Analysis

Run the full pipeline on test data:

```bash
python scripts/run_analysis.py
```

This will:
- Load calibrated thresholds
- Download test data (2016-2024)
- Compute features and symbolize them
- Run forward-chaining inference for each day
- Save results to `results/analysis_results.csv`

### 3. Evaluate

Evaluate performance on historical crashes:

```bash
python scripts/run_evaluation.py
```

This will:
- Evaluate 2008 financial crisis
- Evaluate 2020 COVID-19 crash
- Compute metrics (lead time, precision, recall)
- Generate timeline plots
- Save metrics to `results/metrics/evaluation_metrics.json`

### 4. Live Dashboard

Browse every analysis run — and launch new ones — from a local web dashboard:

```bash
python scripts/run_dashboard.py          # http://127.0.0.1:5000
```

Features:
- **Run archive**: every pipeline execution is persisted under
  `results/runs/<run_id>/` with its parameters, status, day counts, and
  crash-evaluation metrics
- **Instability timeline**: per-run strip chart (green/amber/red per day)
  with hover tooltips, plus an accessible table view
- **Live mode**: launch a new run from the browser — pick an index from the
  dropdown (S&P 500, Nasdaq, Dow, FTSE, DAX, Nikkei, Nifty 50, …) or enter
  any custom Yahoo Finance ticker; the run executes in the background and
  the page tracks it to completion
- **Crash evaluation**: every run is scored against all registered
  historical crashes that overlap its date range — the 2000 dot-com bust,
  the 2008 financial crisis, the 2020 COVID crash, and the 2022 rate-hike
  drawdown (see `CRASH_EVENTS` in `src/evaluation/case_studies.py` to add
  more)
- Fully offline: fonts and assets are bundled, no CDN required

### 5. Static Export & CI

Export a read-only snapshot of the dashboard (no server needed — run data
is embedded into the page). Open `site/index.html` directly, or drop the
`site/` folder onto any static host (Netlify, Vercel, an S3 bucket, or
GitHub Pages on a public repo):

```bash
python scripts/export_static.py --out site   # open site/index.html
```

Continuous integration (`.github/workflows/ci.yml`) runs the full pytest
suite on every push and pull request.

> **Note on GitHub Pages:** publishing straight from Actions needs Pages
> enabled, which requires a **public** repository or a paid plan for a
> private one. This repo therefore ships CI only; use the local
> `export_static.py` snapshot above for hosting.

## Architecture

The system follows a pipeline architecture:

```
Data Acquisition → Feature Engineering → Symbolization → Knowledge Base → Reasoning Engine → Evaluation
```

### Data Acquisition
- Downloads market data (S&P 500, VIX) from Yahoo Finance
- Preprocesses and cleans data (handles missing values, outliers)

### Feature Engineering
- Computes 5 technical features:
  - Rolling return (30-day)
  - Rolling volatility (30-day std of returns)
  - Price slope (linear regression over 30 days)
  - Volume trend (deviation from 30-day MA)
  - VIX (volatility index)

### Symbolization
- Converts numeric features to symbolic states using percentile thresholds
- Creates symbols like: `RisingPrices`, `HighVolatility`, `TightLiquidity`, etc.

### Knowledge Base
- Contains 8 inference rules based on Minsky's Financial Instability Hypothesis
- Rules encode economic theory (e.g., "RisingPrices ∧ HighVolume → SpeculativeActivity")

### Reasoning Engine
- Forward-chaining inference engine
- Derives high-level states from observed symbols
- Determines instability level: Low, Medium, or High

### Evaluation
- Computes metrics: lead time, precision, recall
- Validates on historical crashes (2008, 2020)
- Generates explainable outputs

## Example Output

When running analysis on 2008-09-12 (3 days before Lehman Brothers bankruptcy):

```
Date: 2008-09-12
Instability Level: High

Active Symbols:
  - FallingPrices (price_slope=-0.0038, below 25th percentile)
  - HighVolatility (volatility=0.042, above 80th percentile)
  - TightLiquidity (VIX=31.7, above 75th percentile)
  - LowVolume (volume_trend=-0.22, below 30th percentile)

Fired Rules:
  1. [R5] FallingPrices ∧ HighVolatility ∧ TightLiquidity → PanicState
  2. [R6] PanicState ∧ LowVolume → LiquidityCrisis
  3. [R8] LiquidityCrisis ∧ FallingPrices → ImminentCrash

CONCLUSION: Imminent crash detected due to combination of high crash risk
or liquidity crisis with falling prices and excessive losses.

Lead Time: 3 days before major crash event
```

## Directory Structure

```
symbolic_market_instability/
├── README.md
├── requirements.txt
├── config/
│   ├── symbols.yaml          # Symbol definitions
│   ├── rules.yaml            # Inference rules
│   └── thresholds.json       # Calibrated thresholds
├── data/
│   ├── raw/                  # Downloaded data
│   └── processed/            # Processed data
├── src/
│   ├── data_acquisition/     # Data download and preprocessing
│   ├── feature_engineering/  # Feature computation and calibration
│   ├── symbolization/        # Feature to symbol conversion
│   ├── knowledge_base/       # Rules and ontology
│   ├── reasoning_engine/     # Forward-chaining inference
│   ├── evaluation/           # Metrics and case studies
│   └── utils/                 # Logging and visualization
├── notebooks/                # Jupyter notebooks for exploration
├── tests/                    # Unit tests
├── results/                  # Output files
│   ├── figures/             # Plots
│   ├── explanations/        # Generated explanations
│   └── metrics/             # Evaluation metrics
└── scripts/                  # Main execution scripts
```

## Key Contributions

1. **Novel Application**: First application of symbolic reasoning to financial instability detection
2. **Interpretability**: 100% interpretable alternative to black-box ML models
3. **Theory-Grounded**: Rules based on established economic theory (Minsky's Hypothesis)
4. **Explainable**: Provides causal inference chains for regulatory/policy use
5. **Validated**: Tested on major historical crashes (2008, 2020)

## Methodology

- **Threshold-based symbolization** of market indicators
- **Forward-chaining inference** with economic theory rules
- **Temporal validation** on out-of-sample crashes
- **Explanation generation** for regulatory/policy use

## Evaluation Results

Measured on an S&P 500 (`^GSPC`) run against the two crashes the system was
originally validated on:

| Event | Precision | Recall | Lead time |
|-------|-----------|--------|-----------|
| 2008 financial crisis | 100% | 12.5% | 0 days |
| 2020 COVID crash | 100% | 14.3% | 0 days |

**Honest read:** precision is high — when the system raises a signal it is
almost always inside a genuine crash window, so it rarely cries wolf. But
recall is low and **lead time is effectively zero**: the current rules are
*reactive*. A rule like `FallingPrices ∧ HighVolatility ∧ TightLiquidity →
PanicState` can only fire once prices are already falling — during the crash,
not ahead of it. Genuine *early* warning would require the bubble-phase rules
(R1–R4) to escalate to a Medium/High signal before the drawdown begins. The
generic crash evaluator now also scores the 2000 dot-com bust and the 2022
rate-hike drawdown so this can be tested across more regimes (see Future Work).

## Limitations

1. Limited to daily data (no intraday signals)
2. Rules require periodic recalibration for regime changes
3. Threshold-based symbolization loses granularity
4. Does not incorporate news/sentiment (only price/volume)
5. Validated on only a handful of major crashes (small sample)
6. Current rules are **reactive** — measured crash lead time is ~0 days
   (see Evaluation Results); genuine early warning needs bubble-phase escalation

## Future Work

1. Extend to multi-market contagion analysis
2. Incorporate text/news sentiment symbolization
3. Develop adaptive threshold mechanisms
4. Add probabilistic reasoning (Bayesian networks)
5. Create real-time monitoring dashboard
6. Validate on emerging market crashes
7. Compare with ML baselines systematically

## Citation

If you use this code in your research, please cite:

```
[Add paper citation once published]
```

## License

[Add license information]

## Contact

[Add contact information]
