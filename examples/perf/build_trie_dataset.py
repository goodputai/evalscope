"""Build an immutable prepared TRIE conversation artifact.

The output preserves exact synthetic prompt bytes plus every trace-owned output
cap and tool-call wait, allowing repeated perf runs without prompt synthesis.
"""

import argparse
import hashlib
import random
from pathlib import Path

import numpy as np

from evalscope.perf.arguments import Arguments
from evalscope.perf.plugin.datasets.trie import TrieAgenticCodingPlugin, TrieCodeQaPlugin, TrieOfficeWorkPlugin
from evalscope.perf.plugin.datasets.trie_prepared import FORMAT_VERSION, write_prepared_artifact

_PLUGINS = {
    'trie_agentic_coding': TrieAgenticCodingPlugin,
    'trie_code_qa': TrieCodeQaPlugin,
    'trie_office_work': TrieOfficeWorkPlugin,
}


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=sorted(_PLUGINS), required=True)
    parser.add_argument('--dataset-path', help='Optional frozen raw TRIE JSONL path')
    parser.add_argument('--tokenizer-path', required=True)
    parser.add_argument('--number', type=int, default=1024)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-path', required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.number <= 0:
        raise SystemExit('--number must be positive')

    random.seed(args.seed)
    np.random.seed(args.seed)
    query = Arguments(
        model='prepared-trie-builder',
        url='http://127.0.0.1/v1/chat/completions',
        api='openai',
        dataset=args.dataset,
        dataset_path=args.dataset_path,
        tokenizer_path=args.tokenizer_path,
        multi_turn=True,
        number=args.number,
        seed=args.seed,
    )
    plugin = _PLUGINS[args.dataset](query)
    source_path = plugin._resolve_dataset_path()

    records = []
    for index, conversation in enumerate(plugin.build_messages()):
        records.append((f'{args.dataset}-{index:06d}', conversation))
        if len(records) >= args.number:
            break
    if len(records) < args.number:
        raise SystemExit(f'Only {len(records)} conversations were available; requested {args.number}')

    metadata = {
        'dataset': args.dataset,
        'format_version': FORMAT_VERSION,
        'seed': args.seed,
        'source_file': Path(source_path).name,
        'source_sha256': _sha256(source_path),
        'tokenizer_path': args.tokenizer_path,
    }
    write_prepared_artifact(args.output_path, metadata, records)
    output = Path(args.output_path)
    print(f'Wrote {len(records)} conversations to {output}')
    print(f'SHA-256: {_sha256(str(output))}')


if __name__ == '__main__':
    main()
