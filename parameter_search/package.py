"""Hash and verify the complete parameter-search evidence directory."""
from pathlib import Path
import argparse
import hashlib
import json


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=Path('results_parameter_search'))
    parser.add_argument('--verify',action='store_true')
    args=parser.parse_args();target=args.out/'manifest.json'
    if args.verify:
        data=json.loads(target.read_text())
        for row in data['files']:
            p=args.out/row['path']
            if p.stat().st_size!=row['bytes'] or digest(p)!=row['sha256']:
                raise ArithmeticError(f'Artifact changed: {p}')
        print('PASS:',len(data['files']),'files');return
    for name in ['certificate.json','lower_bound_certificate.json','audit.json','document_validation.json']:
        if not json.loads((args.out/name).read_text())['passed']:
            raise ArithmeticError(f'Check failed: {name}')
    if not json.loads((args.out/'experiment_summary.json').read_text())['complete']:
        raise ArithmeticError('Experiments incomplete')
    files=[]
    for p in sorted(args.out.rglob('*')):
        if not p.is_file() or p==target or '__pycache__' in p.parts or p.suffix in ['.aux','.out','.pyc']:
            continue
        files.append(dict(path=str(p.relative_to(args.out)),bytes=p.stat().st_size,sha256=digest(p)))
    target.write_text(json.dumps(dict(files=files,total_bytes=sum(r['bytes'] for r in files),
               verification='python -m parameter_search.package --verify',
               exclusions='Manifest itself, Python bytecode, transient TeX aux/out files'),indent=2)+'\n')
    print('MANIFEST:',len(files),'files;',sum(r['bytes'] for r in files),'bytes')


if __name__=='__main__':main()
