import asyncio
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from harbor.environments.ack import ACKEnvironment

from evalscope.benchmarks.terminal_bench.ack_environment import (
    IMAGE_DIGEST,
    IMAGE_PULL_SECRET,
    MANAGED_LABELS,
    NODE_LABEL_KEY,
    RELAY_IMAGE,
    RestrictedACKEnvironment,
    TASK_SERVICE_ACCOUNT,
    TMUX_APT_PACKAGES,
    validate_rendered_pod,
    tmux_bootstrap_command,
)
from evalscope.benchmarks.terminal_bench.preflight import terminal_bench_preflight
from evalscope.benchmarks.terminal_bench.terminal_bench_adapter import (
    TerminalBenchV2_1Adapter,
    _run_ack_trial,
)


def rendered_pod() -> dict:
    return {
        'metadata': {'labels': {
            'app': 'sandbox',
            'session': 'trial-env',
            'environment': 'write-compressor',
            **MANAGED_LABELS,
        }},
        'spec': {
            'restartPolicy': 'Never',
            'serviceAccountName': TASK_SERVICE_ACCOUNT,
            'automountServiceAccountToken': False,
            'imagePullSecrets': [{'name': IMAGE_PULL_SECRET}],
            'nodeSelector': {NODE_LABEL_KEY: 'true'},
            'tolerations': [{
                'key': NODE_LABEL_KEY,
                'operator': 'Equal',
                'value': 'true',
                'effect': 'NoSchedule',
            }],
            'hostNetwork': False,
            'hostPID': False,
            'hostIPC': False,
            'securityContext': {'seccompProfile': {'type': 'RuntimeDefault'}},
            'containers': [{
                'name': 'main',
                'image': RELAY_IMAGE,
                'command': ['sleep', 'infinity'],
                'securityContext': {
                    'privileged': False,
                    'allowPrivilegeEscalation': False,
                    'runAsUser': 0,
                    'runAsGroup': 0,
                    'capabilities': {'drop': ['ALL']},
                },
                'resources': {'requests': {
                    'cpu': '1',
                    'memory': '2048Mi',
                    'ephemeral-storage': '10240Mi',
                }},
            }],
        },
    }


def test_rendered_pod_contract_accepts_only_the_fixed_shape():
    validate_rendered_pod(rendered_pod())

    mutations = []
    privileged = deepcopy(rendered_pod())
    privileged['spec']['containers'][0]['securityContext']['privileged'] = True
    mutations.append(privileged)
    secret_volume = deepcopy(rendered_pod())
    secret_volume['spec']['volumes'] = [{'name': 'platform', 'secret': {'secretName': 'acr-credential'}}]
    mutations.append(secret_volume)
    env_from = deepcopy(rendered_pod())
    env_from['spec']['containers'][0]['envFrom'] = [{'secretRef': {'name': 'acr-credential'}}]
    mutations.append(env_from)
    wrong_node = deepcopy(rendered_pod())
    wrong_node['spec']['nodeSelector'] = {}
    mutations.append(wrong_node)
    wrong_resource = deepcopy(rendered_pod())
    wrong_resource['spec']['containers'][0]['resources']['requests']['ephemeral-storage'] = '1Gi'
    mutations.append(wrong_resource)

    for pod in mutations:
        with pytest.raises(ValueError):
            validate_rendered_pod(pod)


@pytest.mark.parametrize(('requested_user', 'effective_user'), [
    ('root', None),
    (0, None),
    ('task-user', 'task-user'),
])
def test_ack_exec_does_not_su_when_the_fixed_pod_is_already_root(
    requested_user,
    effective_user,
):
    environment = object.__new__(RestrictedACKEnvironment)
    with patch.object(ACKEnvironment, 'exec', new_callable=AsyncMock) as parent_exec:
        asyncio.run(environment.exec('id', user=requested_user))

    parent_exec.assert_awaited_once_with(
        command='id',
        cwd=None,
        env=None,
        timeout_sec=None,
        user=effective_user,
    )


def test_tmux_bootstrap_preserves_the_restricted_pod_security_boundary():
    command = tmux_bootstrap_command()

    assert 'APT::Sandbox::User=root' in command
    assert 'dpkg-deb --fsys-tarfile' in command
    assert 'tar -x --no-same-owner' in command
    assert 'apt-get install' not in command
    for package in TMUX_APT_PACKAGES:
        assert package in command

    environment = object.__new__(RestrictedACKEnvironment)
    with patch.object(
        RestrictedACKEnvironment,
        'exec',
        new_callable=AsyncMock,
        return_value=SimpleNamespace(return_code=0, stdout='tmux 3.4', stderr=''),
    ) as exec_command:
        asyncio.run(environment._bootstrap_tmux())

    exec_command.assert_awaited_once()
    assert exec_command.await_args.kwargs['user'] == 'root'


def test_ack_trial_infrastructure_failure_is_not_scored_as_zero():
    with pytest.raises(RuntimeError, match='HarborSetupError: tmux unavailable'):
        TerminalBenchV2_1Adapter._raise_for_ack_failure({
            'exception_info': {
                'exception_type': 'HarborSetupError',
                'exception_message': 'tmux unavailable',
            },
            'verifier_result': None,
        })

    with pytest.raises(RuntimeError, match='did not return a verifier reward'):
        TerminalBenchV2_1Adapter._raise_for_ack_failure({
            'exception_info': None,
            'verifier_result': None,
        })

    TerminalBenchV2_1Adapter._raise_for_ack_failure({
        'exception_info': None,
        'verifier_result': {'rewards': {'reward': 0}},
    })


def test_short_wall_clock_timeout_runs_trial_cleanup():
    class Environment:
        cleanup_error = None
        stopped = False

    class Trial:
        agent_environment = Environment()

        async def run(self):
            try:
                await asyncio.sleep(60)
            finally:
                self.agent_environment.stopped = True

    trial = Trial()
    with pytest.raises(asyncio.TimeoutError):
        asyncio.run(_run_ack_trial(trial, timeout_seconds=0.01))
    assert trial.agent_environment.stopped is True


def test_cleanup_residual_is_promoted_to_failure():
    class Trial:
        agent_environment = SimpleNamespace(cleanup_error='task Pod remains after cleanup')

        async def run(self):
            return object()

    with pytest.raises(RuntimeError, match='remains after cleanup'):
        asyncio.run(_run_ack_trial(Trial(), timeout_seconds=1))


class FakeClientModule:
    class V1ResourceAttributes(SimpleNamespace):
        pass

    class V1SelfSubjectAccessReviewSpec(SimpleNamespace):
        pass

    class V1SelfSubjectAccessReview(SimpleNamespace):
        pass


class FakeAuthApi:
    def create_self_subject_access_review(self, review):
        attributes = review.spec.resource_attributes
        allowed = attributes.resource == 'pods'
        return SimpleNamespace(status=SimpleNamespace(allowed=allowed))


def pull_probe():
    container = SimpleNamespace(
        image=RELAY_IMAGE,
        security_context=SimpleNamespace(
            privileged=False,
            allow_privilege_escalation=False,
            capabilities=SimpleNamespace(drop=['ALL']),
        ),
        resources=SimpleNamespace(requests={
            'cpu': '1', 'memory': '2Gi', 'ephemeral-storage': '10Gi'
        }),
    )
    return SimpleNamespace(
        metadata=SimpleNamespace(creation_timestamp=1),
        spec=SimpleNamespace(
            node_name='cn-hangzhou.10.214.101.143',
            service_account_name=TASK_SERVICE_ACCOUNT,
            automount_service_account_token=False,
            node_selector={NODE_LABEL_KEY: 'true'},
            tolerations=[SimpleNamespace(
                key=NODE_LABEL_KEY, operator='Equal', value='true', effect='NoSchedule'
            )],
            image_pull_secrets=[SimpleNamespace(name=IMAGE_PULL_SECRET)],
            containers=[container],
        ),
        status=SimpleNamespace(
            phase='Succeeded',
            container_statuses=[SimpleNamespace(image_id=f'ghcr.io/example@{IMAGE_DIGEST}')],
        ),
    )


class FakeCoreApi:
    def __init__(self, *, active=False):
        self.active = active

    def read_namespace(self, name):
        return SimpleNamespace(metadata=SimpleNamespace(name=name))

    def list_namespaced_pod(self, namespace, label_selector):
        if label_selector == 'llm-eval.goodput.ai/component=terminal-image-preflight':
            return SimpleNamespace(items=[pull_probe()])
        if self.active:
            return SimpleNamespace(items=[SimpleNamespace(status=SimpleNamespace(phase='Running'))])
        return SimpleNamespace(items=[])


def test_preflight_is_ready_only_with_pull_attestation_and_capacity(tmp_path: Path):
    factory = lambda: (FakeCoreApi(), FakeAuthApi(), FakeClientModule)
    with patch(
        'evalscope.benchmarks.terminal_bench.preflight._installed_version',
        side_effect=lambda package: {'harbor': '0.18.0', 'kubernetes': '36.0.2'}[package],
    ):
        ready = terminal_bench_preflight(str(tmp_path), client_factory=factory)
        busy = terminal_bench_preflight(
            str(tmp_path),
            client_factory=lambda: (FakeCoreApi(active=True), FakeAuthApi(), FakeClientModule),
        )

    assert ready['status'] == 'ready'
    assert all(check['status'] == 'ready' for check in ready['checks'].values())
    assert busy['status'] == 'blocked'
    assert busy['checks']['executor_capacity']['status'] == 'blocked'
