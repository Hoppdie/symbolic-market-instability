"""Flask dashboard: view all runs and trigger new ones from the browser."""

import re
import sys
import threading
from datetime import date
from pathlib import Path
from typing import Callable, Optional

from flask import Flask, jsonify, render_template, request

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.run_manager import RunManager

TICKER_RE = re.compile(r'^[A-Za-z0-9^.\-=]{1,12}$')
DATE_RE = re.compile(r'^\d{4}-\d{2}-\d{2}$')


def _default_executor(ticker, start_date, end_date, run_manager, run_id):
    from src.pipeline.runner import execute_run
    return execute_run(ticker, start_date, end_date, run_manager, run_id=run_id)


def create_app(
    run_manager: Optional[RunManager] = None,
    executor: Optional[Callable] = None,
) -> Flask:
    """
    App factory.

    Args:
        run_manager: Injected for tests; defaults to the project's results dir.
        executor: Callable (ticker, start, end, run_manager, run_id) -> run_id.
                  Injected for tests; defaults to the real pipeline.
    """
    app = Flask(__name__)
    manager = run_manager or RunManager()
    execute = executor or _default_executor

    # One run at a time: the pipeline hits the network and writes shared
    # files, so concurrent runs would corrupt each other.
    run_lock = threading.Lock()
    in_flight = {'run_id': None}

    @app.route('/')
    def index():
        return render_template('index.html')

    @app.route('/api/runs')
    def list_runs():
        return jsonify({'runs': manager.list_runs()})

    @app.route('/api/runs/<run_id>')
    def run_detail(run_id):
        metadata = manager.get_run(run_id)
        if metadata is None:
            return jsonify({'error': f'Unknown run: {run_id}'}), 404
        return jsonify({
            **metadata,
            'metrics': manager.load_metrics(run_id),
            'timeline': manager.load_timeline(run_id),
        })

    @app.route('/api/runs', methods=['POST'])
    def trigger_run():
        payload = request.get_json(silent=True) or {}
        ticker = str(payload.get('ticker', '')).strip()
        start_date = str(payload.get('start_date', '')).strip()
        end_date = str(payload.get('end_date', '')).strip()

        errors = []
        if not TICKER_RE.match(ticker):
            errors.append('ticker must be 1-12 chars of A-Z, 0-9, ^ . - =')
        for label, value in (('start_date', start_date), ('end_date', end_date)):
            if not DATE_RE.match(value):
                errors.append(f'{label} must be YYYY-MM-DD')
        if not errors:
            if start_date >= end_date:
                errors.append('start_date must be before end_date')
            if end_date > date.today().isoformat():
                errors.append('end_date cannot be in the future')
        if errors:
            return jsonify({'errors': errors}), 400

        if not run_lock.acquire(blocking=False):
            return jsonify({
                'error': 'A run is already in progress',
                'run_id': in_flight['run_id'],
            }), 409

        # Create the run synchronously so the response carries its id, then
        # hand the heavy work to a background thread.
        run_id, _ = manager.create_run({
            'ticker': ticker,
            'start_date': start_date,
            'end_date': end_date,
            'source': 'dashboard',
        })
        in_flight['run_id'] = run_id

        def worker():
            try:
                execute(ticker, start_date, end_date, manager, run_id)
            except Exception as exc:
                manager.finalize_run(run_id, status='failed', error=str(exc))
            finally:
                in_flight['run_id'] = None
                run_lock.release()

        threading.Thread(target=worker, daemon=True).start()
        return jsonify({'run_id': run_id, 'status': 'running'}), 202

    return app


if __name__ == '__main__':
    create_app().run(host='127.0.0.1', port=5000, debug=False)
