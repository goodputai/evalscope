import asyncio
import json
import os
import platform
import shutil
import subprocess
import threading
import time
from pathlib import Path

from flask import Blueprint, jsonify


bp_terminal_bench = Blueprint('terminal_bench', __name__, url_prefix='/api/v1/terminal-bench')

PROFILE_ID = os.environ.get('TERMINAL_BENCH_PROFILE_ID', 'terminal_bench_2_1_glm52_official')
DATASET_NAME = 'terminal-bench/terminal-bench-2-1'
DATASET_REF = os.environ.get('TERMINAL_BENCH_DATASET_REF', '6')
EXPECTED_DATASET_HASH = os.environ.get(
    'TERMINAL_BENCH_DATASET_HASH',
    'sha256:7d7bdc1cbedad549fc1140404bd4dc45e5fd0ea7c4186773687d177ad3a0699a',
)
EXPECTED_TASKS = int(os.environ.get('TERMINAL_BENCH_EXPECTED_TASKS', '89'))
EXPECTED_HARBOR_VERSION = os.environ.get('TERMINAL_BENCH_HARBOR_VERSION', '0.18.0')
MIN_CPUS = int(os.environ.get('TERMINAL_BENCH_MIN_HOST_CPUS', '8'))
MIN_MEMORY_BYTES = int(os.environ.get('TERMINAL_BENCH_MIN_HOST_MEMORY_BYTES', str(16 * 1024**3)))
PREFLIGHT_CACHE_SECONDS = int(os.environ.get('TERMINAL_BENCH_PREFLIGHT_CACHE_SECONDS', '60'))

_cache_lock = threading.Lock()
_cache_value: tuple[float, dict] | None = None


def _command_json(command: list[str]) -> dict:
    completed = subprocess.run(command, check=True, capture_output=True, text=True, timeout=30)
    return json.loads(completed.stdout)


def _dataset_details() -> tuple[int, str]:
    from harbor.models.job.config import DatasetConfig

    config = DatasetConfig(name=DATASET_NAME, ref=DATASET_REF)
    tasks = asyncio.run(config.get_task_configs())
    return len(tasks), config.ref or ''


def _path_check(raw_path: str) -> dict:
    path = Path(raw_path)
    return {'path': str(path), 'exists': path.is_dir(), 'writable': os.access(path, os.W_OK)}


def run_preflight() -> dict:
    checks: dict[str, object] = {}
    failures: list[str] = []

    docker_path = shutil.which('docker')
    checks['docker_cli'] = docker_path or 'missing'
    if not docker_path:
        failures.append('Docker CLI is unavailable.')
    else:
        try:
            docker_info = _command_json(['docker', 'info', '--format', '{{json .}}'])
            host_cpus = int(docker_info.get('NCPU') or 0)
            host_memory = int(docker_info.get('MemTotal') or 0)
            architecture = str(docker_info.get('Architecture') or '')
            checks['docker_host'] = {
                'architecture': architecture,
                'cpus': host_cpus,
                'memory_bytes': host_memory,
                'server_version': docker_info.get('ServerVersion'),
            }
            if architecture not in {'x86_64', 'amd64'}:
                failures.append('Docker host must be linux/amd64.')
            if host_cpus < MIN_CPUS:
                failures.append(f'Docker host requires at least {MIN_CPUS} CPUs.')
            if host_memory < MIN_MEMORY_BYTES:
                failures.append('Docker host requires at least 16 GiB memory.')
            compose = subprocess.run(
                ['docker', 'compose', 'version', '--short'],
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
            )
            checks['docker_compose'] = compose.stdout.strip()
        except Exception as exc:
            failures.append(f'Docker daemon/Compose check failed: {type(exc).__name__}.')

    try:
        import harbor

        harbor_version = getattr(harbor, '__version__', 'unknown')
        checks['harbor_version'] = harbor_version
        if harbor_version != EXPECTED_HARBOR_VERSION:
            failures.append(
                f'Harbor version must be {EXPECTED_HARBOR_VERSION}; found {harbor_version}.'
            )
        task_count, resolved_ref = _dataset_details()
        checks['dataset'] = {
            'name': DATASET_NAME,
            'requested_ref': DATASET_REF,
            'resolved_ref': resolved_ref,
            'task_count': task_count,
        }
        if task_count != EXPECTED_TASKS:
            failures.append(f'Expected {EXPECTED_TASKS} tasks, resolved {task_count}.')
        if resolved_ref != EXPECTED_DATASET_HASH:
            failures.append('Harbor dataset content hash does not match the approved revision.')
    except Exception as exc:
        failures.append(f'Harbor dataset check failed: {type(exc).__name__}.')

    output_dir = os.environ.get('EVALSCOPE_OUTPUT_DIR', 'outputs')
    cache_dir = os.environ.get('EVALSCOPE_CACHE', os.path.expanduser('~/.cache/evalscope'))
    checks['paths'] = {
        'output': _path_check(output_dir),
        'cache': _path_check(cache_dir),
    }
    for label, result in checks['paths'].items():
        if not result['exists'] or not result['writable']:
            failures.append(f'{label} directory is missing or not writable.')

    return {
        'status': 'ready' if not failures else 'blocked',
        'profile_id': PROFILE_ID,
        'environment_type': 'docker',
        'controller_architecture': platform.machine(),
        'checks': checks,
        'failures': failures,
    }


def cached_preflight() -> dict:
    global _cache_value
    now = time.monotonic()
    with _cache_lock:
        if _cache_value is not None and now - _cache_value[0] < PREFLIGHT_CACHE_SECONDS:
            return _cache_value[1]
        result = run_preflight()
        _cache_value = (now, result)
        return result


@bp_terminal_bench.get('/preflight')
def terminal_bench_preflight():
    return jsonify(cached_preflight())
