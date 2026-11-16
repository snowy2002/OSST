"""Pinned natural abundance data; empirical populations are not ecosystem truth."""
from __future__ import annotations
from pathlib import Path
from collections import Counter
import csv,hashlib,json,urllib.request
import numpy as np
import rdata
from scipy import sparse

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/c1025'
SPECS={
 'mite':('vegandevs/vegan','19643ae1a547c1789e54a993f42cdcd03b212990','072aabb3f03ee7a364ce8ac4b9e3a022cbac087dedb48b785ffd4ec36433e361','https://vegandevs.github.io/vegan/reference/mite.html','GPL-2','soil mites; 70 spatial soil cores'),
 'bird':('annechao/iNEXT','41a87952f6858f61f5681793e8eac6aa0cf5af75','eb4183b080af2d7ce2f2b39f3714f4ad01b50971b5dd94ddfdaf9612575af733','https://annechao.r-universe.dev/iNEXT/doc/manual.html','GPL >=3','birds; Barrington Tops North and South sites kept separate'),
 'spider':('annechao/iNEXT','41a87952f6858f61f5681793e8eac6aa0cf5af75','aba4a081cfcd8ecbd1d1b2ba78eb7fddb10f48d8b929b28fd7e6b9f2f384543e','https://annechao.r-universe.dev/iNEXT/doc/manual.html','GPL >=3','spiders; Girdled and Logged treatments kept separate'),
 'moths':('pilaboratory/sads','462e9d4e2febfeb679e240ea9a61adc34624f314','df6af55671612c6b2e51583ec1d3748b3479cc2de099986e3e05f5ba1aa5226c','https://pilaboratory.r-universe.dev/sads/doc/manual.html','GPL-2','moths; Rothamsted light-trap aggregate 1933--1936; no temporal rows available'),
 'okland':('pilaboratory/sads','462e9d4e2febfeb679e240ea9a61adc34624f314','6e06344d126e4a81dd0f0e19251a90d1c9e4d19e4a6a7c47900adec250e5cbba','https://pilaboratory.r-universe.dev/sads/doc/manual.html','GPL-2','land snails; Norway historical abundance vector'),
}

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def prepare():
    raw=DATA/'raw';prepared=DATA/'prepared';raw.mkdir(parents=True,exist_ok=True);prepared.mkdir(exist_ok=True)
    datasets=[];sources=[]
    def add(name,counts,labels=None,matrix=None,source=None):
        counts=np.asarray(counts,dtype=float)
        assert counts.ndim==1 and np.all(np.isfinite(counts)) and np.all(counts>=0) and np.all(counts==np.floor(counts))
        keep=counts>0; counts=counts[keep].astype(np.int64)
        labels=np.asarray(labels if labels is not None else [f'category_{i+1}' for i in range(len(keep))],dtype=str)[keep]
        path=prepared/f'{name}.npz';np.savez_compressed(path,counts=counts,labels=labels)
        groups=0
        if matrix is not None:
            matrix=sparse.csr_matrix(np.asarray(matrix,dtype=np.int64)[:,keep]);matrix=matrix[np.asarray(matrix.sum(axis=1)).ravel()>0]
            assert np.array_equal(np.asarray(matrix.sum(axis=0)).ravel(),counts)
            sparse.save_npz(prepared/f'{name}_groups.npz',matrix);groups=matrix.shape[0]
        datasets.append(dict(dataset=name,N=int(counts.sum()),S=len(counts),groups=groups,
                             singleton_species=int(np.sum(counts==1)),sha256=sha(path),source=source,
                             scope='Finite published empirical population only; no claim of full ecosystem support or iid field sampling'))
    for name,(repo,rev,digest,page,license,note) in SPECS.items():
        url=f'https://raw.githubusercontent.com/{repo}/{rev}/data/{name}.rda';p=raw/f'{name}.rda'
        if not p.exists():p.write_bytes(urllib.request.urlopen(url,timeout=45).read())
        assert sha(p)==digest,(name,'raw hash mismatch')
        sources.append(dict(name=name,url=url,commit=rev,sha256=digest,page=page,license=license,note=note))
        obj=rdata.read_rda(p)[name]
        if name=='mite':add(name,np.asarray(obj).sum(axis=0),obj.columns,np.asarray(obj),page)
        elif name=='bird':
            for col in obj.columns:add('bird_'+str(col).split('.')[0].lower(),obj[col].to_numpy(),obj.index,source=page)
        elif name=='spider':
            for key,values in obj.items():add('spider_'+str(key).lower(),values,source=page)
        else:add(name,obj,source=page)
    from benchmark_data import download,prepare_bci
    bci_path=ROOT/'data/large/prepared/bci_counts.npz'
    if not bci_path.exists():download('bci');prepare_bci()
    bci=np.load(bci_path); matrix=sparse.load_npz(ROOT/'data/large/prepared/bci_groups.npz').toarray()
    add('bci',bci['counts'],bci['labels'],matrix,'https://vegandevs.github.io/vegan/reference/BCI.html')
    sources.append(dict(name='bci',**json.loads((ROOT/'data/large/prepared/bci_summary.json').read_text())))
    from fish_data import prepare_fish_data
    records=prepare_fish_data()
    for year in [2007,2008]:
        selected=[r for r in records if r['year']==year and r['gear']=='seine'];labels=sorted({r['species'] for r in selected})
        events=sorted({r['event_id'] for r in selected});lookup={s:i for i,s in enumerate(labels)};elook={s:i for i,s in enumerate(events)}
        matrix=np.zeros((len(events),len(labels)),dtype=np.int64)
        for r in selected:matrix[elook[r['event_id']],lookup[r['species']]]+=1
        add(f'fish_{year}_seine',matrix.sum(axis=0),labels,matrix,'https://fishr-core-team.github.io/FSAdata/reference/InchLake1.html')
    sources.append(json.loads((ROOT/'data/provenance.json').read_text()))
    save(DATA/'sources.json',sources);save(DATA/'datasets.json',datasets)
    return datasets


def load(name):return np.load(DATA/'prepared'/f'{name}.npz')['counts']

if __name__=='__main__': print(json.dumps(prepare(),indent=2))
