"""Reproducible local full-suite verifier; never reports a partial run as success."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def verify(project_dir: Path, timeout_seconds: int = 180):
    env = os.environ.copy()
    for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        env[name] = '1'
    started = time.monotonic()
    command = [sys.executable, '-m', 'pytest', '-q']
    try:
        proc = subprocess.run(command, cwd=project_dir, env=env,
                              capture_output=True, text=True, timeout=timeout_seconds)
        status = 'PASS' if proc.returncode == 0 else 'FAIL'
        output = proc.stdout + proc.stderr
        return {'status': status, 'returncode': proc.returncode,
                'elapsed_seconds': round(time.monotonic()-started, 2),
                'output': output, 'timeout_seconds': timeout_seconds}
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or b''
        stderr = exc.stderr or b''
        if isinstance(stdout, bytes): stdout = stdout.decode(errors='replace')
        if isinstance(stderr, bytes): stderr = stderr.decode(errors='replace')
        return {'status': 'TIMEOUT', 'returncode': None,
                'elapsed_seconds': round(time.monotonic()-started, 2),
                'output': stdout + stderr, 'timeout_seconds': timeout_seconds}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--timeout', type=int, default=180)
    parser.add_argument('--output', default='V29_VERIFICATION.json')
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error('timeout must be positive')
    root = Path(__file__).resolve().parent
    result = verify(root, args.timeout)
    Path(root / args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(result['status'], result['elapsed_seconds'])
    print(result['output'][-1000:])
    return 0 if result['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
