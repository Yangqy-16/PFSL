"""Add a reference ESS panel and human CORUM view; never read SL labels."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse,json,itertools,pickle
import pandas as pd
import torch
from pfsl.next.data import csr,rewire
from script.prepare_ppi import sha256

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',default=str(ROOT/'data/ppi/physical_v1.pt'))
    p.add_argument('--output',default=str(ROOT/'data/ppi/next_v1.pt'))
    p.add_argument('--exclude-cells',nargs='+',default=['A549','PK1','PATU8988S','IPC298','JURKAT','22RV1'])
    args=p.parse_args();output=Path(args.output)
    if output.exists():p.error('Refusing to overwrite existing bank')
    bank=torch.load(args.input,map_location='cpu',weights_only=True)
    ids=bank['gene_ids'].tolist();lookup={g:i for i,g in enumerate(ids)}
    names=pickle.load(open(ROOT/'data/mapping/name2id.pkl','rb'));names={k:int(v) for k,v in names.items()}
    paths=sorted(f for f in (ROOT.parent/'MVGCNiSL/data/cellline_feats').glob('*_ess.txt') if f.stem[:-4] not in args.exclude_cells)
    if not paths:raise ValueError('No reference cells remain')
    raw=torch.full((len(ids),len(paths)),float('nan'))
    for col,path in enumerate(paths):
        for line in path.read_text().splitlines():
            gene,value=line.split()
            if gene in names and names[gene] in lookup:raw[lookup[names[gene]],col]=float(value)
    bank['profile_raw']=raw
    bank['profile_cells']=[f.stem[:-4] for f in paths]
    corum=ROOT/'data/sae_analysis/protrek_final_1024/topk_w8_k32_seed42/feature_atlas_v1/external/raw/corum/allComplexes.txt'
    table=pd.read_csv(corum,sep='\t')
    uid=pd.read_csv(ROOT/'data/mapping/gid2protein.csv',usecols=['From','Entry'])
    mapping={}
    for gid,entry in uid.itertuples(index=False,name=None):
        if int(gid) in lookup:mapping.setdefault(str(entry),set()).add(lookup[int(gid)])
    edges=set();complexes=0
    for row in table.loc[table.organism.str.lower().eq('human')].itertuples(index=False):
        members=set()
        for entry in str(row.subunits_uniprot_id).split(';'):members.update(mapping.get(entry.strip(),()))
        if len(members)>1:
            complexes+=1;edges.update(itertools.combinations(sorted(members),2))
    ptr,idx=bank['indptr'].tolist(),bank['indices'].tolist()
    physical=[(i,j) for i in range(len(ids)) for j in idx[ptr[i]:ptr[i+1]] if i<j]
    print('Rewiring physical edges:',len(physical),flush=True)
    shuffled=rewire(len(ids),physical,42)
    bank['views']={'complex':csr(len(ids),edges),'ppi_rewired':csr(len(ids),shuffled)}
    metadata={'parent_sha256':sha256(args.input),'corum_sha256':sha256(corum),
              'protein_mapping_sha256':sha256(ROOT/'data/mapping/gid2protein.csv'),
              'reference_files':{str(f):sha256(f) for f in paths},
              'excluded_cells':args.exclude_cells,'reference_cells':bank['profile_cells'],
              'profile_gene_coverage':int(torch.isfinite(raw).any(1).sum()),
              'complexes_mapped':complexes,'complex_edges':len(edges),'rewire_seed':42,
              'rewired_edges_changed':len(set(physical)^set(shuffled))//2,
              'profile_preprocessing':'raw cache; imputation/scaling fit on source training genes per fold'}
    bank['next_metadata']=metadata
    output.parent.mkdir(parents=True,exist_ok=True);torch.save(bank,output)
    output.with_suffix('.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps(metadata,indent=2),flush=True)

if __name__=='__main__':main()
