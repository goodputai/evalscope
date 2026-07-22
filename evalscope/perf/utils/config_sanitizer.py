"""Secret-safe serialization helpers for performance benchmark configuration."""

import re
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from typing import Any

_REDACTED = '<redacted>'
_SENSITIVE_KEYS = {
    'api_key',
    'authorization',
    'proxy_authorization',
    'security_token',
    'wandb_api_key',
    'swanlab_api_key',
    'x_api_key',
    'x_amz_credential',
    'x_amz_security_token',
    'x_amz_signature',
    'x_oss_credential',
    'x_oss_security_token',
    'x_oss_signature',
}
_SENSITIVE_SUFFIXES = ('_api_key', '_password', '_secret', '_security_token')
_BEARER_RE = re.compile(r'(?i)\bBearer\s+[^\s,;]+')
_API_KEY_ASSIGNMENT_RE = re.compile(
    r'(?i)(api[_-]?key|authorization|proxy[_-]?authorization)(\s*[=:]\s*)([^\s,;]+)'
)
_URL_RE = re.compile(r'https?://[^\s]+', re.IGNORECASE)


def _normalize_key(value: object) -> str:
    return re.sub(r'[^a-z0-9]+', '_', str(value).strip().lower()).strip('_')


def _is_sensitive_key(value: object) -> bool:
    normalized = _normalize_key(value)
    return normalized in _SENSITIVE_KEYS or normalized.endswith(_SENSITIVE_SUFFIXES)


def sanitize_text(value: str) -> str:
    """Redact common credential forms from an arbitrary diagnostic string."""
    sanitized = _BEARER_RE.sub(f'Bearer {_REDACTED}', value)
    sanitized = _API_KEY_ASSIGNMENT_RE.sub(lambda match: f'{match.group(1)}{match.group(2)}{_REDACTED}', sanitized)
    return _URL_RE.sub(lambda match: _sanitize_url_query(match.group(0)), sanitized)


def _sanitize_url_query(value: str) -> str:
    try:
        parts = urlsplit(value)
    except ValueError:
        return value
    if parts.scheme not in {'http', 'https'} or not parts.netloc:
        return value

    netloc = parts.netloc
    if '@' in netloc:
        netloc = f'{_REDACTED}@{netloc.rsplit("@", 1)[1]}'

    query = parse_qsl(parts.query, keep_blank_values=True)
    if not query and netloc == parts.netloc:
        return value
    safe_query = [(key, _REDACTED if _is_sensitive_key(key) else item) for key, item in query]
    return urlunsplit((parts.scheme, netloc, parts.path, urlencode(safe_query), parts.fragment))


def sanitize_config(value: Any, *, key: object = None) -> Any:
    """Return a recursively sanitized copy of a benchmark configuration."""
    if key is not None and _is_sensitive_key(key):
        return _REDACTED
    if isinstance(value, dict):
        return {item_key: sanitize_config(item, key=item_key) for item_key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_config(item) for item in value]
    if isinstance(value, tuple):
        return tuple(sanitize_config(item) for item in value)
    if isinstance(value, str):
        return sanitize_text(value)
    return value
