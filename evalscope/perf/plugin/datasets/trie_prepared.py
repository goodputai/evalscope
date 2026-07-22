"""Versioned prepared-conversation format for TRIE performance workloads."""

import gzip
import io
import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Sequence, Tuple

from evalscope.perf.plugin.datasets.base import Conversation, Turn

FORMAT_VERSION = 'evalscope-trie-prepared-v1'


def _open_text(path: str):
    if path.endswith('.gz'):
        return gzip.open(path, mode='rt', encoding='utf-8')
    return open(path, mode='r', encoding='utf-8')


def read_prepared_header(path: str) -> Dict[str, Any]:
    """Read and validate the first-line prepared artifact header."""
    with _open_text(path) as handle:
        line = handle.readline()
    if not line:
        raise ValueError(f'Prepared TRIE artifact is empty: {path}')
    try:
        header = json.loads(line)
    except json.JSONDecodeError as error:
        raise ValueError(f'Prepared TRIE header is not valid JSON: {path}') from error
    if not isinstance(header, dict) or header.get('format') != FORMAT_VERSION:
        raise ValueError(f'Unsupported prepared TRIE format in {path}')
    metadata = header.get('metadata')
    if not isinstance(metadata, dict):
        raise ValueError(f'Prepared TRIE metadata must be an object: {path}')
    count = metadata.get('conversation_count')
    if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
        raise ValueError(f'Prepared TRIE conversation_count must be a positive integer: {path}')
    return header


def is_prepared_trie_path(path: str) -> bool:
    """Return whether a local file carries the prepared TRIE header."""
    if not path or not Path(path).is_file():
        return False
    try:
        read_prepared_header(path)
    except (OSError, ValueError):
        return False
    return True


def conversation_to_record(conversation_id: str, conversation: Conversation) -> Dict[str, Any]:
    if not conversation_id:
        raise ValueError('Prepared TRIE conversation_id must not be empty')
    if not conversation:
        raise ValueError(f'Prepared TRIE conversation must not be empty: {conversation_id}')
    return {
        'conversation_id': conversation_id,
        'turns': [
            {
                'messages': turn.messages,
                'max_tokens': turn.max_tokens,
                'tool_call_latency': turn.tool_call_latency,
                'is_final': turn.is_final,
            } for turn in conversation
        ],
    }


def record_to_conversation(record: object) -> Tuple[str, Conversation]:
    """Validate one prepared JSONL record and restore EvalScope Turn objects."""
    if not isinstance(record, dict):
        raise ValueError('Prepared TRIE conversation record must be an object')
    conversation_id = record.get('conversation_id')
    turns = record.get('turns')
    if not isinstance(conversation_id, str) or not conversation_id:
        raise ValueError('Prepared TRIE conversation_id must be a non-empty string')
    if not isinstance(turns, list) or not turns:
        raise ValueError(f'Prepared TRIE turns must be a non-empty list: {conversation_id}')

    conversation: Conversation = []
    for index, item in enumerate(turns):
        if not isinstance(item, dict):
            raise ValueError(f'Prepared TRIE turn {index} must be an object: {conversation_id}')
        messages = item.get('messages')
        if not isinstance(messages, list) or not messages:
            raise ValueError(f'Prepared TRIE turn {index} messages must be non-empty: {conversation_id}')
        for message in messages:
            if not isinstance(message, dict) or not isinstance(message.get('role'), str) or 'content' not in message:
                raise ValueError(f'Prepared TRIE turn {index} has an invalid message: {conversation_id}')

        max_tokens = item.get('max_tokens')
        if (
            not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens <= 0
        ):
            raise ValueError(f'Prepared TRIE turn {index} max_tokens must be positive: {conversation_id}')

        latency = item.get('tool_call_latency')
        if latency is not None and (
            not isinstance(latency, (int, float)) or isinstance(latency, bool) or latency < 0
        ):
            raise ValueError(f'Prepared TRIE turn {index} tool_call_latency is invalid: {conversation_id}')

        is_final = item.get('is_final')
        if not isinstance(is_final, bool):
            raise ValueError(f'Prepared TRIE turn {index} is_final must be boolean: {conversation_id}')
        if is_final != (index == len(turns) - 1):
            raise ValueError(f'Prepared TRIE final marker is inconsistent: {conversation_id}')

        conversation.append(
            Turn(
                messages=messages,
                max_tokens=max_tokens,
                tool_call_latency=None if latency is None else float(latency),
                is_final=is_final,
            )
        )
    return conversation_id, conversation


def _iter_records(path: str) -> Iterator[Tuple[int, Dict[str, Any]]]:
    with _open_text(path) as handle:
        next(handle, None)
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f'Prepared TRIE row {index} is not valid JSON: {path}') from error
            if not isinstance(record, dict):
                raise ValueError(f'Prepared TRIE row {index} must be an object: {path}')
            yield index, record


def iter_prepared_conversations(path: str, dataset_offset: int = 0) -> Iterator[Conversation]:
    """Yield prepared conversations with deterministic rotation and streaming reads."""
    header = read_prepared_header(path)
    count = header['metadata']['conversation_count']
    offset = dataset_offset % count
    yielded = 0

    for index, record in _iter_records(path):
        if index < offset:
            continue
        _, conversation = record_to_conversation(record)
        yielded += 1
        yield conversation

    if offset:
        for index, record in _iter_records(path):
            if index >= offset:
                break
            _, conversation = record_to_conversation(record)
            yielded += 1
            yield conversation

    if yielded != count:
        raise ValueError(
            f'Prepared TRIE conversation_count mismatch: header={count}, rows={yielded}, path={path}'
        )


def write_prepared_artifact(
    path: str,
    metadata: Dict[str, Any],
    records: Sequence[Tuple[str, Conversation]],
) -> None:
    """Write deterministic plain or gzip JSONL prepared artifact bytes."""
    if not records:
        raise ValueError('Prepared TRIE artifact requires at least one conversation')
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    header = {
        'format': FORMAT_VERSION,
        'metadata': {
            **metadata,
            'conversation_count': len(records),
        },
    }

    def _write_lines(handle) -> None:
        handle.write(json.dumps(header, ensure_ascii=True, sort_keys=True, separators=(',', ':')) + '\n')
        for conversation_id, conversation in records:
            record = conversation_to_record(conversation_id, conversation)
            handle.write(json.dumps(record, ensure_ascii=True, sort_keys=True, separators=(',', ':')) + '\n')

    if str(target).endswith('.gz'):
        with target.open('wb') as raw:
            with gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as compressed:
                with io.TextIOWrapper(compressed, encoding='utf-8', newline='\n') as text:
                    _write_lines(text)
    else:
        with target.open('w', encoding='utf-8', newline='\n') as handle:
            _write_lines(handle)
