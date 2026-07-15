from types import SimpleNamespace
from unittest.mock import Mock, patch

from evalscope.benchmarks.terminal_bench.utils import HarborLLM
from evalscope.service.app import create_app
from evalscope.service.blueprints import terminal_bench as preflight


def test_harbor_llm_keeps_context_and_output_limits_separate():
    model = SimpleNamespace(config=SimpleNamespace(max_tokens=48_000))
    llm = HarborLLM(model=model, context_limit=262_144, output_limit=48_000)

    assert llm.get_model_context_limit() == 262_144
    assert llm.get_model_output_limit() == 48_000


def test_docker_preflight_reports_exact_approved_runtime(tmp_path):
    docker_info = {
        'Architecture': 'x86_64',
        'NCPU': 8,
        'MemTotal': 16 * 1024**3,
        'ServerVersion': '26.1.3',
    }
    with (
        patch.object(preflight.shutil, 'which', return_value='/usr/bin/docker'),
        patch.object(preflight, '_command_json', return_value=docker_info),
        patch.object(preflight, '_dataset_details', return_value=(89, preflight.EXPECTED_DATASET_HASH)),
        patch.object(
            preflight.subprocess,
            'run',
            return_value=Mock(stdout='2.27.0\n'),
        ),
        patch.dict(
            preflight.os.environ,
            {'EVALSCOPE_OUTPUT_DIR': str(tmp_path), 'EVALSCOPE_CACHE': str(tmp_path)},
        ),
    ):
        result = preflight.run_preflight()

    assert result['status'] == 'ready'
    assert result['profile_id'] == 'terminal_bench_2_1_glm52_official'
    assert result['checks']['dataset']['task_count'] == 89


def test_preflight_endpoint_fails_closed_without_docker():
    blocked = {'status': 'blocked', 'profile_id': preflight.PROFILE_ID, 'checks': {}, 'failures': ['no docker']}
    app = create_app()
    with patch.object(preflight, 'cached_preflight', return_value=blocked):
        response = app.test_client().get('/api/v1/terminal-bench/preflight')

    assert response.status_code == 200
    assert response.get_json()['status'] == 'blocked'


def test_docker_preflight_rejects_unapproved_harbor_version(tmp_path):
    docker_info = {
        'Architecture': 'x86_64',
        'NCPU': 8,
        'MemTotal': 16 * 1024**3,
        'ServerVersion': '26.1.3',
    }
    with (
        patch.object(preflight.shutil, 'which', return_value='/usr/bin/docker'),
        patch.object(preflight, '_command_json', return_value=docker_info),
        patch.object(preflight, '_dataset_details', return_value=(89, preflight.EXPECTED_DATASET_HASH)),
        patch.object(preflight.subprocess, 'run', return_value=Mock(stdout='2.27.0\n')),
        patch.object(preflight, 'EXPECTED_HARBOR_VERSION', 'unapproved-version'),
        patch.dict(
            preflight.os.environ,
            {'EVALSCOPE_OUTPUT_DIR': str(tmp_path), 'EVALSCOPE_CACHE': str(tmp_path)},
        ),
    ):
        result = preflight.run_preflight()

    assert result['status'] == 'blocked'
    assert any('Harbor version must be' in failure for failure in result['failures'])
