"""Machine-readable readiness checks for the fixed ACK Terminal-Bench smoke."""
import os
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Callable

import evalscope
import evalscope.benchmarks  # noqa: F401
from evalscope.api.registry import BENCHMARK_REGISTRY
from kubernetes.utils.quantity import parse_quantity

from .ack_environment import (
    CONTROLLER_SERVICE_ACCOUNT,
    IMAGE_DIGEST,
    IMAGE_PULL_SECRET,
    MANAGED_LABELS,
    NODE_LABEL_KEY,
    RELAY_IMAGE,
    TASK_NAMESPACE,
    TASK_RESOURCE_REQUESTS,
    TASK_SERVICE_ACCOUNT,
)


PULL_PROBE_LABEL = 'llm-eval.goodput.ai/component=terminal-image-preflight'


def _result(status: str, detail: str) -> dict[str, str]:
    return {'status': status, 'detail': detail}


def _installed_version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def _resource_requests_match(requests: dict[str, str]) -> bool:
    if requests.keys() != TASK_RESOURCE_REQUESTS.keys():
        return False
    try:
        return all(
            parse_quantity(requests[name]) == parse_quantity(expected)
            for name, expected in TASK_RESOURCE_REQUESTS.items()
        )
    except ValueError:
        return False


def _pod_pull_attestation(core_api) -> tuple[bool, str]:
    response = core_api.list_namespaced_pod(TASK_NAMESPACE, label_selector=PULL_PROBE_LABEL)
    probes = sorted(
        response.items,
        key=lambda pod: pod.metadata.creation_timestamp,
        reverse=True,
    )
    if not probes:
        return False, 'No authenticated digest-pull attestation Pod exists.'
    pod = probes[0]
    spec = pod.spec
    statuses = pod.status.container_statuses or []
    pull_secrets = [item.name for item in (spec.image_pull_secrets or [])]
    tolerations = spec.tolerations or []
    matching_toleration = any(
        item.key == NODE_LABEL_KEY
        and item.operator == 'Equal'
        and item.value == 'true'
        and item.effect == 'NoSchedule'
        for item in tolerations
    )
    container = spec.containers[0] if len(spec.containers) == 1 else None
    security = container.security_context if container else None
    requests = container.resources.requests if container and container.resources else {}
    valid = all((
        pod.status.phase == 'Succeeded',
        bool(spec.node_name),
        spec.service_account_name == TASK_SERVICE_ACCOUNT,
        spec.automount_service_account_token is False,
        spec.node_selector == {NODE_LABEL_KEY: 'true'},
        matching_toleration,
        pull_secrets == [IMAGE_PULL_SECRET],
        container is not None and container.image == RELAY_IMAGE,
        _resource_requests_match(requests),
        security is not None and security.privileged is False,
        security is not None and security.allow_privilege_escalation is False,
        security is not None and list(security.capabilities.drop or []) == ['ALL'],
        len(statuses) == 1 and (statuses[0].image_id or '').endswith(IMAGE_DIGEST),
    ))
    if not valid:
        return False, 'Latest pull attestation does not match the fixed image, identity, scheduling, or security contract.'
    return True, f'Authenticated digest pull succeeded on node {spec.node_name}.'


def _controller_rbac_ready(auth_api, client_module) -> tuple[bool, str]:
    allowed = (
        ('create', 'pods', None),
        ('get', 'pods', None),
        ('list', 'pods', None),
        ('watch', 'pods', None),
        ('delete', 'pods', None),
        ('create', 'pods', 'exec'),
        ('get', 'pods', 'log'),
    )
    denied = (
        ('get', 'secrets'),
        ('get', 'persistentvolumeclaims'),
        ('create', 'jobs'),
        ('update', 'deployments'),
        ('update', 'roles'),
    )

    def can_i(verb: str, resource: str, subresource: str | None = None) -> bool:
        attributes = client_module.V1ResourceAttributes(
            namespace=TASK_NAMESPACE,
            verb=verb,
            group='' if resource not in {'jobs', 'deployments', 'roles'} else {
                'jobs': 'batch', 'deployments': 'apps', 'roles': 'rbac.authorization.k8s.io'
            }[resource],
            resource=resource,
            subresource=subresource,
        )
        review = client_module.V1SelfSubjectAccessReview(
            spec=client_module.V1SelfSubjectAccessReviewSpec(resource_attributes=attributes)
        )
        return bool(auth_api.create_self_subject_access_review(review).status.allowed)

    if not all(can_i(*item) for item in allowed):
        return False, f'{CONTROLLER_SERVICE_ACCOUNT} is missing a required Pod operation.'
    if any(can_i(verb, resource) for verb, resource in denied):
        return False, f'{CONTROLLER_SERVICE_ACCOUNT} has a forbidden non-Pod permission.'
    return True, 'Controller has only the required Pod lifecycle, exec, and log operations.'


def terminal_bench_preflight(
    outputs_root: str | None,
    *,
    client_factory: Callable[[], tuple[Any, Any, Any]] | None = None,
) -> dict[str, Any]:
    checks: dict[str, dict[str, str]] = {}

    runtime_ready = (
        sys.version_info >= (3, 12)
        and evalscope.__version__ == '1.9.0'
        and _installed_version('harbor') == '0.18.0'
        and _installed_version('kubernetes') == '36.0.2'
    )
    checks['runtime'] = _result(
        'ready' if runtime_ready else 'blocked',
        'Python 3.12+, EvalScope 1.9.0, Harbor 0.18.0, and Kubernetes client 36.0.2 are pinned.'
        if runtime_ready else 'Runtime package versions do not match the fixed release contract.',
    )
    benchmark_ready = 'terminal_bench_v2_1' in BENCHMARK_REGISTRY
    checks['benchmark'] = _result(
        'ready' if benchmark_ready else 'blocked',
        'terminal_bench_v2_1 is registered.' if benchmark_ready else 'terminal_bench_v2_1 is not registered.',
    )

    output_path = Path(outputs_root or os.environ.get('EVALSCOPE_OUTPUTS', '/app/outputs'))
    output_ready = output_path.is_dir() and os.access(output_path, os.R_OK | os.W_OK)
    checks['output_path'] = _result(
        'ready' if output_ready else 'blocked',
        'EvalScope output path is readable and writable.' if output_ready else 'EvalScope output path is unavailable.',
    )

    try:
        if client_factory is None:
            from kubernetes import client, config

            config.load_incluster_config()
            core_api = client.CoreV1Api()
            auth_api = client.AuthorizationV1Api()
            client_module = client
        else:
            core_api, auth_api, client_module = client_factory()
        managed_pods = core_api.list_namespaced_pod(
            TASK_NAMESPACE,
            label_selector=','.join(f'{key}={value}' for key, value in MANAGED_LABELS.items()),
        ).items
        checks['kubernetes_api'] = _result('ready', f'Namespace {TASK_NAMESPACE} is reachable.')

        rbac_ready, rbac_detail = _controller_rbac_ready(auth_api, client_module)
        checks['controller_rbac'] = _result('ready' if rbac_ready else 'blocked', rbac_detail)

        pull_ready, pull_detail = _pod_pull_attestation(core_api)
        checks['image_pull_attestation'] = _result('ready' if pull_ready else 'blocked', pull_detail)

        active = [pod for pod in managed_pods if pod.status.phase not in {'Succeeded', 'Failed'}]
        checks['executor_capacity'] = _result(
            'ready' if not active else 'blocked',
            'No active or residual Terminal-Bench task Pod exists.'
            if not active else 'A Terminal-Bench task Pod is active or residual.',
        )
    except Exception as exc:
        checks.setdefault('kubernetes_api', _result('blocked', f'Kubernetes readiness check failed: {type(exc).__name__}.'))
        checks.setdefault('controller_rbac', _result('blocked', 'Controller RBAC could not be verified.'))
        checks.setdefault('image_pull_attestation', _result('blocked', 'Authenticated digest pull could not be verified.'))
        checks.setdefault('executor_capacity', _result('blocked', 'Executor capacity could not be verified.'))

    status = 'ready' if all(item['status'] == 'ready' for item in checks.values()) else 'blocked'
    return {'status': status, 'checks': checks}
