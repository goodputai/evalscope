import hashlib
import tempfile
import unittest
from pathlib import Path

from evalscope.perf.arguments import Arguments
from evalscope.perf.plugin.datasets.base import Turn
from evalscope.perf.plugin.datasets.trie import TrieAgenticCodingPlugin
from evalscope.perf.plugin.datasets.trie_prepared import (
    iter_prepared_conversations,
    record_to_conversation,
    write_prepared_artifact,
)


def _conversation(label: str):
    return [
        Turn(
            messages=[{'role': 'user', 'content': f'{label}-first'}],
            max_tokens=10,
            tool_call_latency=None,
            is_final=False,
        ),
        Turn(
            messages=[{'role': 'user', 'content': f'{label}-second'}],
            max_tokens=20,
            tool_call_latency=0.25,
            is_final=True,
        ),
    ]


class TestPreparedTrieArtifact(unittest.TestCase):

    def test_gzip_bytes_are_deterministic(self) -> None:
        records = [('a', _conversation('a')), ('b', _conversation('b'))]
        metadata = {'dataset': 'trie_agentic_coding', 'seed': 42}
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / 'first.jsonl.gz'
            second = Path(directory) / 'second.jsonl.gz'
            write_prepared_artifact(str(first), metadata, records)
            write_prepared_artifact(str(second), metadata, records)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            self.assertEqual(
                hashlib.sha256(first.read_bytes()).hexdigest(),
                hashlib.sha256(second.read_bytes()).hexdigest(),
            )

    def test_streaming_loader_rotates_and_preserves_turn_contract(self) -> None:
        records = [('a', _conversation('a')), ('b', _conversation('b')), ('c', _conversation('c'))]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'prepared.jsonl.gz'
            write_prepared_artifact(str(path), {'dataset': 'trie_agentic_coding'}, records)
            loaded = list(iter_prepared_conversations(str(path), dataset_offset=1))

        self.assertEqual(
            [conversation[0].messages[0]['content'] for conversation in loaded],
            ['b-first', 'c-first', 'a-first'],
        )
        self.assertEqual([turn.max_tokens for turn in loaded[0]], [10, 20])
        self.assertEqual([turn.tool_call_latency for turn in loaded[0]], [None, 0.25])
        self.assertEqual([turn.is_final for turn in loaded[0]], [False, True])

    def test_trie_plugin_loads_prepared_artifact_without_tokenizer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'prepared.jsonl.gz'
            write_prepared_artifact(
                str(path),
                {'dataset': 'trie_agentic_coding'},
                [('a', _conversation('a'))],
            )
            args = Arguments(
                model='model',
                url='https://example.com/v1/chat/completions',
                dataset='trie_agentic_coding',
                dataset_path=str(path),
                multi_turn=True,
                number=1,
            )
            plugin = TrieAgenticCodingPlugin(args)
            loaded = list(plugin.build_messages())

        self.assertIsNone(plugin.tokenizer)
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0][1].max_tokens, 20)

    def test_invalid_final_marker_is_rejected(self) -> None:
        record = {
            'conversation_id': 'bad',
            'turns': [{
                'messages': [{'role': 'user', 'content': 'x'}],
                'max_tokens': 1,
                'tool_call_latency': None,
                'is_final': False,
            }],
        }
        with self.assertRaisesRegex(ValueError, 'final marker'):
            record_to_conversation(record)


if __name__ == '__main__':
    unittest.main()
