"""Build and test a portable artifact without including unrelated data or caches."""
import argparse,json,shutil,subprocess,sys,tempfile,zipfile
from pathlib import Path
from c1025.data import ROOT,DATA,save,sha


def package(out):
    for p in (ROOT/'c1025').glob('*'):
        if p.suffix in ['.py','.txt']:
            target=out/'source_snapshot/c1025'/p.name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    latex=out/'latex_source.zip'
    with zipfile.ZipFile(latex,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in [*sorted(out.glob('*.tex')),*sorted((out/'figures').glob('*.pdf')),out/'REPRODUCE.txt']:
            z.write(p,p.relative_to(out))
    files=set(ROOT.glob('*.py'))
    files.update(ROOT/p for p in ['README.md','pytest.ini','requirements.txt','requirements-lock.txt','2610.18723v1.pdf'])
    files.update(p for p in (ROOT/'publication').glob('*.py'))
    files.update(p for p in (ROOT/'c1025').glob('*') if p.suffix in ['.py','.txt'])
    files.update(p for p in DATA.rglob('*') if p.is_file())
    files.update(ROOT/'data'/p for p in ['InchLake1.csv','inch_lake_fish.csv','provenance.json','strata_summary.csv'])
    files.update(ROOT/'data/large/prepared'/p for p in ['bci_counts.npz','bci_groups.npz','bci_summary.json'])
    files.add(ROOT/'data/large/raw/BCI.rda')
    files.update(ROOT/'artifacts'/p for p in ['paper.txt','audit_c0_1025.py','c0_reduction_proof.tex',
        'c0_1025_paper/c0_1025_note.tex','c0_1025_paper/c0_1025_note.pdf',
        'c0_1025_audit/verify_fractional_output.py','c0_1025_audit/exact_endpoint_certificates.py'])
    files.update(p for p in out.rglob('*') if p.is_file() and 'preview' not in p.relative_to(out).parts
                 and p.name not in ['artifact_manifest.json','portable_verification.json'])
    assert all(p.exists() for p in files)
    # A clean directory must run the existing tests and public entry point
    # using only the selected portable source/data files.
    with tempfile.TemporaryDirectory(prefix='osst_c1025_portable_') as temp:
        target=Path(temp)
        for p in files:
            relative=p.relative_to(ROOT)
            if relative.parts[0]==out.name:continue
            q=target/relative;q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
        tests=subprocess.run([sys.executable,'-m','pytest','-q'],cwd=target,capture_output=True,text=True)
        if tests.returncode:raise RuntimeError(tests.stdout+tests.stderr)
        demo=subprocess.run([sys.executable,'paper_reference.py','demo','--n','1000','--epsilon','0.1','--c0','1025'],
                            cwd=target,capture_output=True,text=True,check=True)
        result=json.loads(demo.stdout)
        assert result['plan']['C0']==1025 and [r['decision'] for r in result['runs']]==['Yes','No']
        save(out/'portable_verification.json',dict(passed=True,tests=tests.stdout.strip(),public_demo_passed=True,
              scope='Fresh temporary directory, only bundled source/data; existing installed Python dependencies; no project-relative imports from original workspace'))
    files.add(out/'portable_verification.json')
    manifest={str(p.relative_to(ROOT)):sha(p) for p in sorted(files)}
    save(out/'artifact_manifest.json',dict(files=manifest,scope='SHA256 of each bundled file, excluding this manifest itself'))
    files.add(out/'artifact_manifest.json')
    destination=ROOT/'artifacts/osst_c1025_experiments.zip'
    with zipfile.ZipFile(destination,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(files):z.write(p,Path('osst_c1025_artifact')/p.relative_to(ROOT))
    with zipfile.ZipFile(destination) as z:
        assert z.testzip() is None
        import hashlib
        for rel,digest in manifest.items():
            assert hashlib.sha256(z.read('osst_c1025_artifact/'+rel)).hexdigest()==digest
    destination.with_suffix('.zip.sha256').write_text(sha(destination)+'  '+destination.name+'\n')
    print(json.dumps(dict(artifact=str(destination),bytes=destination.stat().st_size,files=len(files),sha256=sha(destination),portable_test_passed=True),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=ROOT/'results_c1025')
    package(parser.parse_args().out.resolve())
