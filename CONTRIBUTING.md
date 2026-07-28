# Contributing

Thanks for your interest in improving this project! This guide covers the
local setup and the conventions used here.

## Development setup

All code lives under `symbolic_market_instability/`.

```bash
cd symbolic_market_instability
pip install -r requirements.txt
```

## Running the pipeline

```bash
python scripts/run_calibration.py   # one-time: download data, calibrate thresholds
python scripts/run_analysis.py      # run the forward-chaining analysis
python scripts/run_evaluation.py    # score against the historical crashes
python scripts/run_dashboard.py     # launch the live dashboard at 127.0.0.1:5000
```

## Tests

```bash
pytest tests/ -q
```

Please add or update tests for any behaviour you change — the suite runs on
every push and pull request via `.github/workflows/ci.yml`.

## Conventions

- **Interpretability first.** New rules or symbols should stay explainable —
  every signal must trace from a raw feature to a fired rule.
- Add new historical crash events to the `CRASH_EVENTS` registry in
  `src/evaluation/case_studies.py` rather than writing bespoke evaluators.
- Keep generated artifacts (`data/`, `results/`, `site/`) out of commits;
  they are already covered by `.gitignore`.
- Match the surrounding code style (docstrings on public functions, type
  hints where practical).

## Pull requests

Keep PRs focused, describe what changed and why, and make sure `pytest`
passes before opening one.
