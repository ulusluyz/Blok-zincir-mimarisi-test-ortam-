from pathlib import Path
import subprocess
from unittest.mock import patch

from verify_phase1 import verify


def test_success_is_only_zero_exit():
    with patch('verify_phase1.subprocess.run', return_value=subprocess.CompletedProcess([], 0, '139 passed', '')) as run:
        result = verify(Path('.'))
    assert result['status'] == 'PASS'
    assert all(run.call_args.kwargs['env'][key] == '1' for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'))


def test_nonzero_is_failure():
    with patch('verify_phase1.subprocess.run', return_value=subprocess.CompletedProcess([], 1, 'fail', '')):
        assert verify(Path('.'))['status'] == 'FAIL'


def test_timeout_is_not_success():
    with patch('verify_phase1.subprocess.run', side_effect=subprocess.TimeoutExpired('pytest', 2, output=b'... 51%')):
        result = verify(Path('.'), 2)
    assert result['status'] == 'TIMEOUT'
    assert '51%' in result['output']
