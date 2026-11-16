"""Reproduce the noninteger full-degree output and rational endpoint checks."""
import argparse,importlib.util,json,shutil
from pathlib import Path
from c1025.data import ROOT,sha


def module_at(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def run(out):
    target=out/'algebra_audit';target.mkdir(parents=True,exist_ok=True)
    source=ROOT/'artifacts/c0_1025_audit/verify_fractional_output.py'
    prior=target/'fractional_output.json'
    if prior.exists():
        record=json.loads(prior.read_text())
        assert record['passed'] and record['script_sha256']==sha(source)
        print('Preserving completed noninteger-output verification')
    else:
        module=module_at('fractional_output_followup',source);module.HERE=target;module.main()
    source=ROOT/'artifacts/c0_1025_audit/exact_endpoint_certificates.py'
    local=target/source.name;shutil.copy2(source,local)
    module_at('exact_endpoints_followup',local).main()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=ROOT/'results_c1025')
    run(parser.parse_args().out.resolve())
