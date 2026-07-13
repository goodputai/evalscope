"""Restricted ACK environment for the fixed Terminal-Bench 2.1 smoke."""
import asyncio
from typing import Any

from harbor.environments.ack import ACKEnvironment
from kubernetes.client.rest import ApiException


TASK_NAME = 'terminal-bench/write-compressor'
TASK_PACKAGE_REF = 'sha256:d9ddd9a8e925e2c566b37b2492cbf995afecefe58874e4043ef78d7f3c892c7e'
UPSTREAM_IMAGE = 'alexgshaw/write-compressor:20251031'
IMAGE_DIGEST = 'sha256:3618e1f8a997b437c09cd3dbaac736705809c05f6f12f584b65722175970ebe1'
RELAY_IMAGE = f'ghcr.io/goodputai/llm-eval/terminal-bench-write-compressor@{IMAGE_DIGEST}'
TASK_NAMESPACE = 'evalscope'
TASK_SERVICE_ACCOUNT = 'terminal-task-no-token'
CONTROLLER_SERVICE_ACCOUNT = 'evalscope-terminal-controller'
IMAGE_PULL_SECRET = 'terminal-bench-ghcr-pull'
NODE_LABEL_KEY = 'workload.goodput.ai/terminal-bench'
MANAGED_LABELS = {
    'app.kubernetes.io/name': 'terminal-bench-task',
    'app.kubernetes.io/managed-by': 'evalscope',
    'llm-eval.goodput.ai/profile': 'terminal-bench-2-1-fixed',
}


def fixed_ack_kwargs() -> dict[str, Any]:
    return {
        'namespace': TASK_NAMESPACE,
        'image_pull_secret': IMAGE_PULL_SECRET,
        'service_account': TASK_SERVICE_ACCOUNT,
        'node_selector': {NODE_LABEL_KEY: 'true'},
        'tolerations': [{
            'key': NODE_LABEL_KEY,
            'operator': 'Equal',
            'value': 'true',
            'effect': 'NoSchedule',
        }],
        # Harbor's registry HTTP fallback cannot authenticate to private GHCR.
        # Readiness instead requires a successful digest-pinned pull probe.
        'skip_image_check': True,
        'pod_overrides': {
            'metadata': {'labels': MANAGED_LABELS},
            'spec': {
                'automountServiceAccountToken': False,
                'hostNetwork': False,
                'hostPID': False,
                'hostIPC': False,
                'securityContext': {'seccompProfile': {'type': 'RuntimeDefault'}},
                'containers': [{
                    'name': 'main',
                    'securityContext': {
                        'privileged': False,
                        'allowPrivilegeEscalation': False,
                        'runAsUser': 0,
                        'runAsGroup': 0,
                        'capabilities': {'drop': ['ALL']},
                    },
                }],
            },
        },
    }


def validate_rendered_pod(pod: dict[str, Any]) -> None:
    """Fail before creation unless Harbor rendered the exact approved Pod."""
    metadata = pod.get('metadata') or {}
    spec = pod.get('spec') or {}
    containers = spec.get('containers') or []
    if len(containers) != 1:
        raise ValueError('Terminal-Bench ACK Pod must contain exactly one container.')
    container = containers[0]
    security = container.get('securityContext') or {}
    pod_security = spec.get('securityContext') or {}
    pull_secrets = spec.get('imagePullSecrets') or []
    resources = (container.get('resources') or {}).get('requests') or {}

    if container.get('name') != 'main' or container.get('image') != RELAY_IMAGE:
        raise ValueError('Terminal-Bench ACK Pod image changed.')
    if spec.get('serviceAccountName') != TASK_SERVICE_ACCOUNT:
        raise ValueError('Terminal-Bench ACK Pod ServiceAccount changed.')
    if spec.get('automountServiceAccountToken') is not False:
        raise ValueError('Terminal-Bench ACK Pod must not mount a Kubernetes API token.')
    if pull_secrets != [{'name': IMAGE_PULL_SECRET}]:
        raise ValueError('Terminal-Bench ACK Pod pull secret changed.')
    if spec.get('nodeSelector') != {NODE_LABEL_KEY: 'true'}:
        raise ValueError('Terminal-Bench ACK Pod node selector changed.')
    if spec.get('tolerations') != fixed_ack_kwargs()['tolerations']:
        raise ValueError('Terminal-Bench ACK Pod toleration changed.')
    if not MANAGED_LABELS.items() <= (metadata.get('labels') or {}).items():
        raise ValueError('Terminal-Bench ACK Pod managed labels changed.')
    if resources != {'cpu': '1', 'memory': '2048Mi', 'ephemeral-storage': '10240Mi'}:
        raise ValueError('Terminal-Bench ACK Pod resources changed.')
    if any(spec.get(key) for key in ('hostNetwork', 'hostPID', 'hostIPC', 'volumes', 'initContainers')):
        raise ValueError('Terminal-Bench ACK Pod contains forbidden host, volume, or init-container access.')
    if container.get('env') or container.get('envFrom') or container.get('volumeMounts'):
        raise ValueError('Terminal-Bench ACK Pod contains forbidden environment or volume sources.')
    if security != fixed_ack_kwargs()['pod_overrides']['spec']['containers'][0]['securityContext']:
        raise ValueError('Terminal-Bench ACK container security context changed.')
    if pod_security != {'seccompProfile': {'type': 'RuntimeDefault'}}:
        raise ValueError('Terminal-Bench ACK Pod seccomp profile changed.')


class _ValidatingCoreApi:
    """Validate Harbor's final Pod dict immediately before the API call."""

    def __init__(self, delegate):
        self._delegate = delegate

    def __getattr__(self, name):
        return getattr(self._delegate, name)

    def create_namespaced_pod(self, namespace, body, **kwargs):
        if namespace != TASK_NAMESPACE:
            raise ValueError('Terminal-Bench ACK namespace changed.')
        validate_rendered_pod(body)
        return self._delegate.create_namespaced_pod(namespace=namespace, body=body, **kwargs)


class RestrictedACKEnvironment(ACKEnvironment):
    """Harbor ACK environment whose task Pod shape cannot be caller-controlled."""

    cleanup_error: str | None = None

    def __init__(self, *args, **kwargs):
        expected = fixed_ack_kwargs()
        try:
            profile = {key: kwargs.pop(key) for key in expected}
        except KeyError as exc:
            raise ValueError('Terminal-Bench ACK environment profile is incomplete.') from exc
        if profile != expected:
            raise ValueError('Terminal-Bench ACK environment must use the fixed platform profile.')
        task_env_config = kwargs.get('task_env_config')
        if task_env_config is None and len(args) >= 5:
            task_env_config = args[4]
        if task_env_config is None:
            raise ValueError('Terminal-Bench ACK task environment is missing.')
        if task_env_config.docker_image != UPSTREAM_IMAGE:
            raise ValueError('Terminal-Bench task image does not match the pinned task package.')
        if (task_env_config.cpus, task_env_config.memory_mb, task_env_config.storage_mb) != (1, 2048, 10240):
            raise ValueError('Terminal-Bench task resources do not match the fixed profile.')
        if task_env_config.gpus != 0 or task_env_config.env or task_env_config.mcp_servers:
            raise ValueError('Terminal-Bench fixed task may not add GPUs, environment variables, or MCP servers.')
        if kwargs.get('persistent_env') or kwargs.get('extra_docker_compose'):
            raise ValueError('Terminal-Bench ACK environment may not add environment or Compose overrides.')
        if any(kwargs.get(key) is not None for key in (
            'override_cpus', 'override_memory_mb', 'override_storage_mb', 'override_gpus', 'override_tpu'
        )):
            raise ValueError('Terminal-Bench ACK resource overrides are disabled.')
        task_env_config.docker_image = RELAY_IMAGE
        if 'task_env_config' in kwargs:
            kwargs['task_env_config'] = task_env_config
        elif len(args) >= 5:
            args = (*args[:4], task_env_config, *args[5:])
        super().__init__(*args, **profile, **kwargs)
        self._validate_profile()

    def _validate_profile(self) -> None:
        expected = fixed_ack_kwargs()
        if self.namespace != TASK_NAMESPACE or self.service_account != TASK_SERVICE_ACCOUNT:
            raise ValueError('Terminal-Bench ACK identity boundary changed.')
        if self.image_pull_secret != IMAGE_PULL_SECRET or self.skip_image_check is not True:
            raise ValueError('Terminal-Bench ACK image boundary changed.')
        if self.node_selector != expected['node_selector'] or self.tolerations != expected['tolerations']:
            raise ValueError('Terminal-Bench ACK scheduling boundary changed.')
        if self.pod_overrides != expected['pod_overrides']:
            raise ValueError('Terminal-Bench ACK Pod security boundary changed.')

    async def start(self, force_build: bool) -> None:
        if force_build:
            raise ValueError('Terminal-Bench ACK image builds are disabled.')
        self._validate_profile()
        await self._ensure_client()
        assert self._core_api is not None
        if not isinstance(self._core_api, _ValidatingCoreApi):
            self._core_api = _ValidatingCoreApi(self._core_api)
        await super().start(force_build=False)

    async def stop(self, delete: bool) -> None:
        pod_name = self.pod_name
        core_api = self._core_api
        await super().stop(delete=delete)
        if not delete or core_api is None:
            return
        try:
            await asyncio.to_thread(core_api.read_namespaced_pod, pod_name, self.namespace)
        except ApiException as exc:
            if exc.status == 404:
                return
            self.cleanup_error = f'Unable to verify deletion of task Pod {pod_name}.'
            raise RuntimeError(self.cleanup_error) from exc
        self.cleanup_error = f'Terminal-Bench task Pod {pod_name} remains after cleanup.'
        raise RuntimeError(self.cleanup_error)
