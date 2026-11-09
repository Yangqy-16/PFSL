"""Shared CLI, split auditing and outputs for opt-in author-method adapters."""
from pathlib import Path
import argparse
import hashlib
import itertools
import json
import os
import random
import subprocess
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
COLUMNS=['Accuracy','Precision','Recall','AUC','AUPR','F1','BACC']


def parser_for(method,mode,project):
    p=argparse.ArgumentParser(description=f'{method}: existing SLKB {mode} splits; author architecture/loss')
    p.add_argument('--cell-lines',nargs='+',default=['A549','PK1','PATU8988S'])
    p.add_argument('--scenes',nargs='+',choices=['C1','C2','C3','Tail'],default=['C1','C3'])
    p.add_argument('--folds',nargs='+',type=int,choices=range(5),default=list(range(5)))
    p.add_argument('--ratio',choices=['1:1','original'],default='1:1')
    p.add_argument('--data-root',type=Path,default=ROOT/'PFSL/data/SLKB')
    p.add_argument('--output-root',type=Path)
    p.add_argument('--gpu',default='2',help="Physical GPU id or 'cpu'")
    p.add_argument('--seed',type=int,default=42 if method=='PromptCASL' else 1024)
    p.add_argument('--epochs',type=int,help='Optional explicit override; default uses author settings')
    p.add_argument('--dry-run',action='store_true',help='Print planned jobs without loading models or writing outputs')
    p.add_argument('--check',action='store_true',help='Check splits and required assets; no training')
    p.add_argument('--resume',action='store_true',help='Skip completed folds only; never overwrite partial runs')
    return p


def make_jobs(mode,cells,scenes,folds,ratio,data_root,output_root):
    if len(set(cells))!=len(cells):raise ValueError('Duplicate cell names')
    if mode=='onetoone' and len(cells)<2:raise ValueError('One-to-one needs at least two cells')
    suffix='_1:1' if ratio=='1:1' else ''
    if mode=='multitoone':
        panel=['22RV1','A375','A549','JURKAT','MEWO','PK1']
        if set(cells)-set(panel):raise ValueError('Unsupported multi-to-one target')
        return [dict(source=[c for c in panel if c!=target],target=target,scene=None,fold=fold,
                     data=Path(data_root)/(mode+suffix)/f'Multi_5_to_{target}',
                     output=Path(output_root)/f'Multi_5_to_{target}'/f'fold_{fold}')
                for target in cells for fold in folds]
    groups=itertools.permutations(cells,2) if mode=='onetoone' else itertools.product(cells,scenes)
    jobs=[]
    for a,b in groups:
        target=b if mode=='onetoone' else a
        name=Path(a+'_'+b) if mode=='onetoone' else Path(a)/b
        for fold in folds:
            jobs.append(dict(source=a,target=target,scene=None if mode=='onetoone' else b,fold=fold,
                             data=Path(data_root)/(mode+suffix)/name,output=Path(output_root)/name/f'fold_{fold}'))
    return jobs


def configure(args,mode,project):
    args.output_root=(args.output_root or project/f'results_{mode}_{args.ratio}').resolve()
    args.data_root=args.data_root.resolve()
    if args.epochs is not None and args.epochs<1:raise ValueError('--epochs must be positive')
    os.environ['CUDA_VISIBLE_DEVICES']='' if args.gpu=='cpu' else args.gpu
    os.environ.setdefault('WANDB_MODE','disabled')
    os.environ.setdefault('MPLCONFIGDIR','/tmp/slkb-adapters-mpl')
    return make_jobs(mode,args.cell_lines,args.scenes,args.folds,args.ratio,args.data_root,args.output_root)


def read_frame(path,context):
    frame=pd.read_csv(path)
    needed=['context','gene1','gene2','label']
    if not set(needed)<=set(frame):raise ValueError(f'Missing columns in {path}')
    frame=frame[needed].copy()
    if frame.empty:raise ValueError(f'Empty split: {path}')
    allowed=[context] if isinstance(context,str) else list(context)
    if not frame.context.isin(allowed).all():raise ValueError(f'Unexpected context in {path}: expected {context}')
    for col in ['gene1','gene2','label']:
        values=pd.to_numeric(frame[col],errors='raise')
        if not np.isfinite(values).all() or not values.eq(np.floor(values)).all():raise ValueError(f'{col} must contain finite integer values')
        frame[col]=values.astype('int64')
    if not set(frame.label)<=set([0,1]):raise ValueError('Labels must be 0/1')
    if frame.gene1.eq(frame.gene2).any():raise ValueError('Self pair in split')
    if len(pair_keys(frame))!=len(frame):raise ValueError(f'Duplicate undirected pair in {path}')
    return frame


def pair_keys(frame):
    return {(c,min(a,b),max(a,b)) for c,a,b in frame[['context','gene1','gene2']].itertuples(index=False,name=None)}


def load_splits(folder,fold,source,target):
    data={s:read_frame(Path(folder)/f'sl_{s}_{fold}.csv',source if s!='test' else target) for s in ['train','val','test']}
    for a,b in itertools.combinations(data,2):
        if pair_keys(data[a])&pair_keys(data[b]):raise ValueError(f'{a}/{b} overlap in the same cell context')
    return data


def seed_all(seed):
    import torch
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)


def start_fold(job,args,project,extra=None):
    out=job['output']
    if args.resume and (out/'completed.json').is_file():return False
    if out.exists() and any(out.iterdir()):raise FileExistsError(f'Refusing to overwrite {out}')
    out.mkdir(parents=True,exist_ok=True)
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    config['job']={k:str(v) if isinstance(v,Path) else v for k,v in job.items()}
    config['upstream_commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=project,text=True).strip()
    config['split_sha256']={s:hashlib.sha256((job['data']/f'sl_{s}_{job["fold"]}.csv').read_bytes()).hexdigest() for s in ['train','val','test']}
    config['adaptation_notes']=extra or {}
    (out/'run_config.json').write_text(json.dumps(config,indent=2)+'\n')
    return True


def export_predictions(frame,logits,out):
    from sklearn.metrics import accuracy_score,precision_score,recall_score,roc_auc_score,average_precision_score,f1_score,balanced_accuracy_score
    logits=np.asarray(logits)
    if logits.shape!=(len(frame),2) or not np.isfinite(logits).all():raise ValueError('Invalid two-class logits or prediction row count')
    z=logits-logits.max(1,keepdims=True);prob=np.exp(z);prob=prob[:,1]/prob.sum(1)
    pred=prob>0.5;y=frame.label.to_numpy()
    metrics=dict(zip(COLUMNS,[accuracy_score(y,pred),precision_score(y,pred,zero_division=0),recall_score(y,pred,zero_division=0),
        roc_auc_score(y,prob) if len(np.unique(y))==2 else float('nan'),average_precision_score(y,prob),
        f1_score(y,pred,zero_division=0),balanced_accuracy_score(y,pred)]))
    result=frame.copy();result['logit_0']=logits[:,0];result['logit_1']=logits[:,1];result['logit']=logits[:,1]-logits[:,0];result['prob']=prob
    result.to_csv(out/'test_predictions.csv',index=False)
    pd.DataFrame([metrics]).to_csv(out/'metrics.csv',index=False)
    (out/'completed.json').write_text(json.dumps({'metrics':metrics,'test_count':len(frame)},indent=2)+'\n')
    return metrics


def summarize(jobs):
    for folder in set(j['output'].parent for j in jobs):
        rows=[]
        for fold in range(5):
            p=folder/f'fold_{fold}/metrics.csv'
            if p.is_file():rows.append(pd.read_csv(p).iloc[0].to_dict()|{'fold':fold})
        if not rows:continue
        detailed=pd.DataFrame(rows);detailed.to_csv(folder/'fold_metrics.csv',index=False)
        frame=detailed[COLUMNS]
        if len(frame)==5:
            mean,std=frame.mean(),frame.std(ddof=1);frame=frame.astype(object)
            frame.loc[5]=[f'{mean[k]:.4f} ({std[k]:.4f})' for k in COLUMNS]
        frame.to_csv(folder/'cv_results.csv',index=False)


def print_jobs(jobs):
    print(json.dumps([{k:str(v) if isinstance(v,Path) else v for k,v in j.items()} for j in jobs],indent=2))


def check_jobs(jobs,asset_errors):
    errors=list(asset_errors)
    for job in jobs:
        try:load_splits(job['data'],job['fold'],job['source'],job['target'])
        except (ValueError,FileNotFoundError) as exc:errors.append(str(exc))
    report={'jobs':len(jobs),'ready':not errors,'errors':list(dict.fromkeys(errors))}
    print(json.dumps(report,indent=2));return report
