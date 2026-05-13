from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ML_DIR = ROOT / 'ml-api'
BACKEND_DIR = ROOT / 'backend'


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


def get_json(url: str):
    with urllib.request.urlopen(url, timeout=10) as response:
        return json.loads(response.read().decode('utf-8'))


def post_json(url: str, payload: dict):
    data = json.dumps(payload).encode('utf-8')
    request = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
    # Use a generous timeout so on-chain settlements that wait for
    # Ethereum transaction confirmation have enough time to complete.
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode('utf-8'))


def wait_for_json(url: str, timeout: float = 45.0):
    deadline = time.time() + timeout
    last_error = None
    while time.time() < deadline:
        try:
            return get_json(url)
        except Exception as error:  # noqa: BLE001
            last_error = error
            time.sleep(1)
    raise RuntimeError(f'Timed out waiting for {url}: {last_error}')


def launch_process(command: list[str], cwd: Path, env: dict[str, str]):
    return subprocess.Popen(
        command,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def drain_output(process: subprocess.Popen[str]) -> str:
    if process.stdout is None:
        return ''
    try:
        return process.stdout.read()
    except Exception:  # noqa: BLE001
        return ''


def main() -> int:
    ml_port = free_port()
    backend_port = free_port()
    ml_url = f'http://127.0.0.1:{ml_port}'
    backend_url = f'http://127.0.0.1:{backend_port}'

    ml_env = os.environ.copy()
    backend_env = os.environ.copy()
    shared_secret = 'smoke-shared-secret'
    ml_env['ORACLE_SHARED_SECRET'] = shared_secret
    backend_env['ORACLE_SHARED_SECRET'] = shared_secret
    backend_env['PORT'] = str(backend_port)
    backend_env['ML_API_URL'] = ml_url

    ml_process = launch_process(
        [sys.executable, '-c', f"from app import app; app.run(host='127.0.0.1', port={ml_port}, debug=False)"],
        ML_DIR,
        ml_env,
    )
    backend_process = launch_process(['node', 'server.js'], BACKEND_DIR, backend_env)

    try:
        ml_health = wait_for_json(f'{ml_url}/health')
        backend_health = wait_for_json(f'{backend_url}/health')
        campaigns = get_json(f'{backend_url}/campaigns')
        metrics_before = get_json(f'{backend_url}/metrics')
        selected = campaigns[0]
        validations = []
        for index in range(5):
            # Use a time-based suffix so each smoke run produces unique
            # request/visitor IDs and avoids replay-protection conflicts
            # with previous executions stored in backend/data/store.json.
            uniq_suffix = f"{int(time.time() * 1000)}-{index}"
            request_id = f'full-stack-smoke-{uniq_suffix}'
            visitor_id = f'full-stack-visitor-{uniq_suffix}'
            validations.append(post_json(
                f'{backend_url}/simulate-click',
                {
                    'campaignId': selected['id'],
                    'requestId': request_id,
                    'walletAddress': '0x1111111111111111111111111111111111111111',
                    'visitorId': visitor_id,
                    'clickTime': f'2024-01-01T00:00:0{index + 3}.000Z',
                    'deviceType': 'mobile' if index % 2 == 0 else 'desktop',
                    'userAgent': 'Mozilla/5.0 (Linux; Android 14; Mobile)',
                    'platform': 'Android',
                },
            ))
        metrics_after = get_json(f'{backend_url}/metrics')
        clicks = get_json(f'{backend_url}/clicks')
        benchmarks = get_json(f'{backend_url}/benchmarks/report')
        security = get_json(f'{backend_url}/security/report')
        drift = post_json(f'{backend_url}/ml/drift/report', {'sampleSize': 3})
        adversarial = post_json(f'{backend_url}/ml/adversarial-evaluation', {'sample': clicks[0]['features'] if clicks else {}})
        result = {
            'ml_health': {
                'status': ml_health.get('status'),
                'model': ml_health.get('model'),
                'threshold': ml_health.get('threshold'),
                'dataset': ml_health.get('dataset'),
                'oracle': ml_health.get('oracle'),
            },
            'backend_health': backend_health,
            'campaign_count': len(campaigns),
            'selected_campaign': selected,
            'metrics_before': metrics_before,
            'validations': validations,
            'metrics_after': metrics_after,
            'benchmarks': benchmarks,
            'security': security,
            'drift': drift,
            'adversarial': adversarial,
            'latest_click': clicks[0] if clicks else None,
        }
        print(json.dumps(result, indent=2))
        return 0
    finally:
        for process in (backend_process, ml_process):
            process.terminate()
        for process in (backend_process, ml_process):
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
        print('\n--- ml-api log ---')
        print(drain_output(ml_process).strip())
        print('\n--- backend log ---')
        print(drain_output(backend_process).strip())


if __name__ == '__main__':
    raise SystemExit(main())
