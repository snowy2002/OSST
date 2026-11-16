"""Write or verify a SHA256 manifest for the complete experiment artifact."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from c1025.data import ROOT, save, sha


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT/'results_c7016')
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args(argv)
    target = args.out/'manifest.json'
    if args.verify:
        manifest = json.loads(target.read_text())
        for item in manifest['files']:
            path = args.out/item['path']
            if path.stat().st_size != item['bytes'] or sha(path) != item['sha256']:
                raise RuntimeError(f'Artifact checksum failed: {path}')
        print(f'PASS: {len(manifest["files"])} artifact files')
        return
    if not json.loads((args.out/'validation.json').read_text())['passed']:
        raise RuntimeError('Result validation has not passed')
    for suite in ['synthetic', 'natural', 'lifted', 'amplification', 'field_stress', 'diagnostics']:
        if not (args.out/f'{suite}_complete.json').exists():
            raise RuntimeError(f'Incomplete experiment phase: {suite}')
    files = []
    for path in sorted(args.out.rglob('*')):
        if (not path.is_file() or path == target or '__pycache__' in path.parts
                or path.suffix in ['.log', '.aux', '.out', '.pyc']):
            continue
        files.append(dict(path=str(path.relative_to(args.out)), bytes=path.stat().st_size, sha256=sha(path)))
    save(target, dict(algorithm='C0=70,L0=16', files=files, total_bytes=sum(f['bytes'] for f in files),
          exclusions='Self, transient logs, TeX intermediates and Python bytecode',
          verification='python -m c7016.package --out results_c7016 --verify'))
    print(f'Manifest: {len(files)} files, {sum(f["bytes"] for f in files):,} bytes')


if __name__ == '__main__':
    main()
