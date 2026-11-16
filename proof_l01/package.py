"""Assemble proof appendices and a portable certificate replay archive."""
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import zipfile
from pypdf import PdfReader,PdfWriter

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results_proof_l01'


def main():
    summary=json.loads((OUT/'summary.json').read_text())
    if summary['status']!='PASS':
        raise ValueError('Proof certificates must pass before packaging')
    pdfs=[(OUT/'C70_L1_proof.pdf','New C70 L1 sufficient-constant proof'),
          (ROOT/'results_parameter_search/proof.pdf','Appendix A: refined large-range concentration'),
          (ROOT/'artifacts/joint_constants_submission/submission_proof.pdf','Appendix B: exact coefficient and interval lemmas')]
    writer=PdfWriter()
    pages={}
    for path,title in pdfs:
        pages[str(path.relative_to(ROOT))]=len(PdfReader(path).pages)
        writer.append(path,outline_item=title)
    writer.add_metadata({'/Title':'C70 L1: complete sufficient-constant proof and supporting appendices'})
    combined=OUT/'C70_L1_complete_proof.pdf'
    writer.write(combined)
    paths=set()
    for folder in ['proof_l01','parameter_search','c7016','c1025']:
        paths.update((ROOT/folder).glob('*.py'))
    paths.update((ROOT/'proof_l01').glob('*.txt'))
    for name in ['support_size.py','paper_reference.py','guaranteed.py','benchmark_core.py',
                 'real_l01/__init__.py','real_l01/theory_review.py',
                 'results_parameter_search/certificate.json','results_parameter_search/proof.tex',
                 'results_parameter_search/proof.pdf','results_parameter_search/lower_bound_certificate.json']:
        paths.add(ROOT/name)
    folder=ROOT/'artifacts/joint_constants_submission'
    for ext in ['*.py','*.tex','*.pdf','*.json']:
        paths.update(folder.glob(ext))
    for name in ['summary.json','basic_certificate.json','large_range.json','large_range_independent_audit.json',
                 'small_certificates.json','small_audit.json','n74_certificate.json','n74_audit.json',
                 'mgf_certificates.json','finite_audit.json','midrange_certificate.json',
                 'midrange_independent_audit.json','C70_L1_proof.tex','certified_bounds.tex',
                 'C70_L1_proof.pdf','C70_L1_complete_proof.pdf','manuscript_review.txt',
                 'reliability_report.txt']:
        paths.add(OUT/name)
    environment=dict(python=platform.python_version(),packages={p:version(p) for p in ['python-flint','numpy','scipy','pypdf']})
    requirements='\n'.join(f'{k}=={v}' for k,v in environment['packages'].items())+'\n'
    readme='''C0=70, L0=1 sufficient-constant proof

Unzip and run from the extracted directory:
  python -m pip install -r proof_requirements.txt
  OPENBLAS_NUM_THREADS=1 python -m proof_l01.verify_all

The verifier replays rational proof parameters and global interval bounds.
It does not run the exploratory numerical optimizers or require biological data.
Python 3.14 was used for the provided run; the exact tested library versions
are recorded in proof_environment.json. Runtime is ordinarily a few minutes.

Read results_proof_l01/C70_L1_complete_proof.pdf for the proof and appendices.
The theorem is conditional on iid Poisson sampling and the exact stated
statistic. It is a sufficiency proof for 70, not an optimal-real-constant claim.
This archive is not a Lean/Coq formalization.

Source and artifact SHA256 values are listed in proof_manifest.json.
Exploration outputs are omitted because their selected rational parameters
are already contained in the certificates. No network access is needed by
the certificate verifier after installing dependencies.
'''
    manifest={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}
    archive=OUT/'C70_L1_certificate_bundle.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for path in sorted(paths):
            z.write(path,str(path.relative_to(ROOT)))
        z.writestr('REPRODUCE.txt',readme)
        z.writestr('proof_requirements.txt',requirements)
        z.writestr('proof_environment.json',json.dumps(environment,indent=2)+'\n')
        z.writestr('proof_manifest.json',json.dumps(manifest,indent=2)+'\n')
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for path,digest in manifest.items():
            assert hashlib.sha256(z.read(path)).hexdigest()==digest,path
    result=dict(status='PASS',pdf_page_counts=pages,complete_pdf_pages=sum(pages.values()),
                files=len(manifest),archive_bytes=archive.stat().st_size,
                archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                environment=environment,archive_crc_and_all_hashes_verified=True)
    (OUT/'package_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
