import json
import os
from pathlib import Path
import threading
from datetime import datetime, timezone
from typing import Any


_lock = threading.RLock()
_WORK_DIR_ENV = 'EVALSCOPE_RUNTIME_WORK_DIR'
_ATTEMPT_ID_ENV = 'EVALSCOPE_RUNTIME_ATTEMPT_ID'


def configure_runtime_liveness(work_dir: str, attempt_id: str | None) -> None:
    os.environ[_WORK_DIR_ENV] = work_dir
    if attempt_id:
        os.environ[_ATTEMPT_ID_ENV] = attempt_id
    else:
        os.environ.pop(_ATTEMPT_ID_ENV, None)
    # RESET (not merge) the request/sample phase fields so a stale liveness.json
    # left by a SIGKILL'd subprocess cannot pretend a generation is still open.
    # design.md §6.4 (R2 init-merge bug).
    update_runtime_liveness(
        attempt_id=attempt_id,
        current_sample_uid=None,
        last_request_started_at=None,
        last_chunk_at=None,
        bytes_received=0,
        provider_request_id=None,
        generation_request_open=False,
    )


def update_runtime_liveness(**values: Any) -> None:
    work_dir = os.environ.get(_WORK_DIR_ENV)
    if not work_dir:
        return
    path = Path(work_dir) / 'liveness.json'
    with _lock:
        payload: dict[str, Any] = {}
        try:
            existing = json.loads(path.read_text(encoding='utf-8'))
            if isinstance(existing, dict):
                payload = existing
        except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError):
            pass
        payload.update(values)
        payload['attempt_id'] = os.environ.get(_ATTEMPT_ID_ENV)
        payload['updated_at'] = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(payload, ensure_ascii=True, sort_keys=True), encoding='utf-8')
        os.replace(temporary, path)


def record_request_started() -> None:
    update_runtime_liveness(
        last_request_started_at=datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
        last_chunk_at=None,
        bytes_received=0,
        provider_request_id=None,
        generation_request_open=True,
    )


def record_request_closed() -> None:
    """Mark the generation request as no longer open.

    Called from a ``finally`` around both sync/async generation entrypoints
    (``anthropic_compatible.py``). This is the P1 phase signal (design §6.4):
    when False, the heartbeat verdict must NOT treat a stall as a hung
    generation — there is no in-flight request to be hung.
    """
    update_runtime_liveness(generation_request_open=False)


def record_sample_started(sample_uid: str) -> None:
    update_runtime_liveness(current_sample_uid=sample_uid)


def record_sample_completed() -> None:
    """Clear per-sample liveness fields at the sample-end boundary.

    So a subsequent judging/finalization window on a new idiom is not seen as a
    stalled generation on the now-finished sample (design §6.4).
    """
    update_runtime_liveness(
        current_sample_uid=None,
        last_chunk_at=None,
        bytes_received=0,
    )


def record_stream_event(event: Any) -> None:
    try:
        if hasattr(event, 'model_dump_json'):
            size = len(event.model_dump_json().encode('utf-8'))
        else:
            size = len(str(event).encode('utf-8'))
    except Exception:
        size = 0
    provider_request_id = None
    message = getattr(event, 'message', None)
    if message is not None:
        provider_request_id = getattr(message, 'id', None)
    with _lock:
        path = os.environ.get(_WORK_DIR_ENV)
        previous_bytes = 0
        if path:
            try:
                payload = json.loads((Path(path) / 'liveness.json').read_text(encoding='utf-8'))
                previous_bytes = int(payload.get('bytes_received') or 0)
            except (FileNotFoundError, ValueError, TypeError, json.JSONDecodeError):
                pass
        values = {
            'last_chunk_at': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
            'bytes_received': previous_bytes + size,
        }
        if provider_request_id:
            values['provider_request_id'] = provider_request_id
        update_runtime_liveness(**values)


def read_runtime_liveness(work_dir: str) -> dict:
    try:
        payload = json.loads((Path(work_dir) / 'liveness.json').read_text(encoding='utf-8'))
        return payload if isinstance(payload, dict) else {}
    except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError):
        return {}
