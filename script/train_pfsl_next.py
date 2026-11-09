"""Isolated staged PFSL runner, with source-validation selection and optional UDA."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse,json,pickle,random,math
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from pfsl.next.data import fit_profile,NextBatchBuilder,load_omics,PairDataset,MultiContextPairDataset
from pfsl.next.model import NextGraph
from pfsl.next.fusion import NoGraphFusion
from pfsl.next.da import CDANE
from pfsl.model.ppi import PPIResidual
from pfsl.utils.metric import compute_metrics

ROOT=Path(__file__).resolve().parents[1]
STAGES=['legacy','self','rewired','gated','profile','multiview','context','auxiliary','cdan','cdane','gated_corum',
        'no_func','no_seq','no_omics','no_ppi','no_graph_concat','no_graph_gate']


def ppi_options(stage):
    return dict(use_func=stage!='no_func',use_seq=stage!='no_seq',use_omics=stage!='no_omics')


def settings(stage):
    if stage not in STAGES:raise ValueError(stage)
    return dict(profile=stage in ['profile','multiview','context','auxiliary','cdan','cdane'],
                views=['ppi','complex'] if stage in ['multiview','context','auxiliary','cdan','cdane','gated_corum'] else ['ppi'],
                split=stage in ['context','auxiliary','cdan','cdane'],
                auxiliary=stage in ['auxiliary','cdan','cdane'],da=stage in ['cdan','cdane'])


def to_device(data,device):return {k:v.to(device) for k,v in data.items()}


def forward(model,data):
    value=model(data)
    return value if isinstance(value,dict) else {'logit':value,'auxiliary':{}}


@torch.no_grad()
def evaluate(model,loader,device):
    model.eval();logits=[];labels=[];pairs=[]
    for batch in loader:
        data=to_device(batch,device);logits.append(forward(model,data)['logit'].cpu());labels.append(batch['label'])
        pairs.append(torch.stack([batch['g1_idx'],batch['g2_idx']],1))
    logits=torch.cat(logits);labels=torch.cat(labels)
    metrics={k:float(v) for k,v in compute_metrics(labels,logits.sigmoid()).items()}
    metrics['mean_loss']=float(F.binary_cross_entropy_with_logits(logits,labels))
    return metrics,torch.cat(pairs),labels,logits


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=STAGES,required=True)
    p.add_argument('--source',required=True);p.add_argument('--target')
    p.add_argument('--scene',choices=['C1','C3','primary_gene_holdout'],default='C3');p.add_argument('--fold',type=int,choices=range(5),default=0)
    p.add_argument('--bank',default=str(ROOT/'data/ppi/next_v1.pt'));p.add_argument('--epochs',type=int,default=100)
    p.add_argument('--fusion-ablation',choices=['full','no_text','no_seq','no_omics','omics_only','no_ess','seq_only','text_only'],default='full')
    p.add_argument('--ablation-mode',choices=['zero','remove'],default='zero',help='Legacy zero-input or actual branch removal; new corrected runs use remove')
    p.add_argument('--omics-normalization',choices=['global','train_genes'],default='global',help='Legacy global statistics or strict per-fold training-gene statistics')
    p.add_argument('--text-bank',type=Path,help='Audited frozen text-aspect control bank; graph-free concat only')
    p.add_argument('--seed',type=int,default=42);p.add_argument('--device',default='cuda')
    p.add_argument('--output',required=True);p.add_argument('--da-protocol',choices=['transductive','inductive'])
    p.add_argument('--data-root',type=Path,help='Alternative root containing source_target or cell/scene folders')
    p.add_argument('--multitoone',action='store_true',help='Five source contexts with per-row omics; graph-free full model only')
    p.add_argument('--random-missing-ess',action='store_true',help='Learn a shared missing ESS token initialized N(0,0.02^2); retain missing bits')
    p.add_argument('--unlabeled-target');p.add_argument('--warmup',type=int,default=10)
    p.add_argument('--da-lr',type=float,default=5e-5)
    p.add_argument('--da-weight',type=float,default=.05);p.add_argument('--aux-weight',type=float,default=.1)
    args=p.parse_args();opt=settings(args.stage);out=Path(args.output)
    if args.omics_normalization=='train_genes' and (args.target or args.multitoone):
        p.error('Strict normalization currently supports specific C1/C3; transfer needs an explicit target-domain policy')
    if args.ablation_mode=='remove' and args.stage!='no_graph_concat':
        p.error('Structural removal is supported for no_graph_concat')
    if args.multitoone and (args.stage!='no_graph_concat' or not args.target):
        p.error('--multitoone requires no_graph_concat and a target cell')
    if args.random_missing_ess and args.stage!='no_graph_concat':p.error('Random missing ESS requires no_graph_concat')
    if (args.fusion_ablation!='full' or args.text_bank) and args.stage!='no_graph_concat':
        p.error('Information ablations require --stage no_graph_concat')
    if args.epochs<1:p.error('--epochs must be positive')
    if out.exists() and any(out.iterdir()):p.error('Output is not empty')
    if opt['da'] and (not args.target or not args.da_protocol):p.error('CDAN requires --target and an explicit --da-protocol')
    if opt['da'] and args.da_protocol=='inductive' and not args.unlabeled_target:p.error('Inductive DA requires a separate unlabeled target file')
    torch.set_num_threads(2);torch.manual_seed(args.seed);np.random.seed(args.seed);random.seed(args.seed)
    device=torch.device(args.device)
    if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('Requested GPU is unavailable')
    bank=torch.load(args.bank,map_location='cpu',weights_only=True)
    text_control=None
    if args.text_bank:
        replacement=torch.load(args.text_bank,map_location='cpu',weights_only=True)
        if not torch.equal(replacement['gene_ids'],bank['gene_ids']):raise ValueError('Text control gene order mismatch')
        if replacement['text'].ndim!=2 or replacement['text'].shape[0]!=bank['text'].shape[0]:raise ValueError('Text control gene-count mismatch')
        if not torch.isfinite(replacement['text']).all():raise ValueError('Nonfinite replacement text')
        bank['text']=replacement['text']
        if 'available' in replacement:
            bank['available']=bank['available'].clone()
            bank['available'][:,1]=replacement['available'].to(bank['available'].dtype)
        text_control=replacement['metadata']
    if opt['profile'] and {args.source,args.target}&set(bank['profile_cells']):raise ValueError('Evaluation cell appears in reference profile panel')
    folder=args.data_root or ROOT/'data/SLKB'/('onetoone_1:1' if args.target else 'specific_1:1')
    folder=folder/(args.source+'_'+args.target) if args.target else folder/args.source/args.scene
    if args.multitoone:
        folder=(args.data_root or ROOT/'data/SLKB/multitoone_1:1')/f'Multi_5_to_{args.target}'
    paths={s:folder/f'sl_{s}_{args.fold}.csv' for s in ['train','val','test']}
    train_genes=set(pd.read_csv(paths['train'],usecols=['gene1','gene2']).to_numpy().flatten())
    lookup={g:i for i,g in enumerate(bank['gene_ids'].tolist())}
    if train_genes-lookup.keys():raise ValueError('Training genes missing from bank')
    if args.stage in ['no_graph_concat','no_graph_gate']:
        required=set()
        for path in paths.values():
            required.update(pd.read_csv(path,usecols=['gene1','gene2']).to_numpy().flatten())
        absent=required-lookup.keys()
        if absent:raise ValueError(f'Genes missing from bank: {sorted(absent)[:10]}')
        indices=[lookup[g] for g in sorted(required)]
        if not bank['available'][indices].bool().all():raise ValueError('No-availability fusion requires observed sequence and text for every used gene')
        if not all(torch.isfinite(bank[k][indices]).all() for k in ['seq','text']):raise ValueError('Nonfinite input features')
    if opt['profile']:
        profile,mask,fit=fit_profile(bank['profile_raw'],[lookup[g] for g in train_genes])
    else:
        profile=torch.empty(len(lookup),0);mask=profile.bool();fit={'enabled':False}
    names=pickle.load(open(ROOT/'data/mapping/name2id.pkl','rb'));omics_root=ROOT.parent/'MVGCNiSL/data/cellline_feats'
    builder_cache={};omics_fits={}
    def builder(cell):
        if cell in builder_cache:return builder_cache[cell]
        omics=load_omics(omics_root,cell,names,allow_missing_ess=args.random_missing_ess)
        fit=None
        if args.omics_normalization=='train_genes':
            from pfsl.omics_scaling import fit_omics
            fit=fit_omics(omics,train_genes);omics_fits[cell]=fit
        value=NextBatchBuilder(bank,omics,profile,mask,views=opt['views'],
                              mode='none' if args.stage in ['no_ppi','no_graph_concat','no_graph_gate'] else args.stage if args.stage in ['self','rewired'] else 'real',seed=42,allow_empty_ess=args.random_missing_ess,omics_statistics=fit)
        builder_cache[cell]=value
        return value
    if args.multitoone:
        from slkb_baseline_common import load_splits
        sources=[c for c in ['22RV1','A375','A549','JURKAT','MEWO','PK1'] if c!=args.target]
        load_splits(folder,args.fold,sources,args.target)
        builders={cell:builder(cell) for cell in sources}
        train=MultiContextPairDataset(paths['train'],builders)
        val=MultiContextPairDataset(paths['val'],builders)
    else:
        source_builder=builder(args.source)
        train=PairDataset(paths['train'],source_builder,args.source)
        val=PairDataset(paths['val'],source_builder,args.source)
    def loader(ds,shuffle=False,batch=16):
        return torch.utils.data.DataLoader(ds,batch_size=batch,shuffle=shuffle,collate_fn=ds.collate,num_workers=0,
                                          generator=torch.Generator().manual_seed(args.seed))
    train_loader=loader(train,True);val_loader=loader(val,batch=64)
    target_loader=None
    if opt['da']:
        target_path=Path(args.unlabeled_target) if args.unlabeled_target else paths['test']
        unlabeled=PairDataset(target_path,builder(args.target),args.target,labeled=False)
        if not len(unlabeled):raise ValueError('Empty adaptation pool')
        if args.da_protocol=='inductive':
            # Compare pair identities only; target labels remain unopened.
            test_pairs=pd.read_csv(paths['test'],usecols=['gene1','gene2']).to_numpy()
            held={tuple(sorted(x)) for x in test_pairs}
            if any(tuple(sorted(x)) in held for x in unlabeled.pairs):raise ValueError('Inductive adaptation pool overlaps evaluation pairs')
        target_loader=loader(unlabeled,True)
    if args.stage in ['no_graph_concat','no_graph_gate']:
        model=NoGraphFusion(bank['seq'].shape[1],bank['text'].shape[1],gated=args.stage=='no_graph_gate',ablation=args.fusion_ablation,random_missing_ess=args.random_missing_ess,ablation_mode=args.ablation_mode)
    elif args.stage in ['legacy','self','rewired','no_func','no_seq','no_omics','no_ppi']:
        model=PPIResidual(bank['seq'].shape[1],bank['text'].shape[1],standalone=True,scale=1.,**ppi_options(args.stage))
    else:
        model=NextGraph(bank['seq'].shape[1],bank['text'].shape[1],profile.shape[1] if opt['profile'] else 0,
                        views=opt['views'],split_context=opt['split'],auxiliary=opt['auxiliary'])
    model=model.to(device);da=CDANE(128,entropy=args.stage=='cdane').to(device) if opt['da'] else None
    parameters=[{'params':model.parameters(),'lr':1e-4}]
    if da is not None:parameters.append({'params':da.parameters(),'lr':args.da_lr})
    optimizer=torch.optim.Adam(parameters,lr=1e-4,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min')
    out.mkdir(parents=True,exist_ok=True)
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    config.update(settings=opt,reference_cells=bank['profile_cells'] if opt['profile'] else [],
                source_training_gene_ids=sorted(map(int,train_genes)),bank_metadata=bank['next_metadata'])
    if text_control is not None:config['text_control']=text_control
    if args.stage in ['no_graph_concat','no_graph_gate']:
        config['input_audit']={'n_genes':len(required),'sequence_text_complete':True,'availability_features':False,'external_neighbors':False,'gate':args.stage=='no_graph_gate'}
    config['parameter_count']=sum(p.numel() for p in model.parameters())
    if args.stage in ['no_graph_concat','no_graph_gate']:
        config['active_modalities']=model.active_modalities
        config['fusion_input_dim']=model.input[0].in_features
    if omics_fits:
        config['omics_fit_file']='omics_fit.json'
        (out/'omics_fit.json').write_text(json.dumps(omics_fits,indent=2)+'\n')
    if args.random_missing_ess and getattr(model,'missing_ess',None) is not None:
        config['missing_ess_policy']={'type':'shared_trainable_scalar','initialization':'normal(0,0.02^2)',
            'initial_value':float(model.missing_ess.detach().cpu()[0]),'missing_indicator_retained':True,
            'missing_cell_files':[c for c in (sources+[args.target] if args.multitoone else [args.source,args.target])
                                  if c and not (omics_root/f'{c}_ess.txt').is_file()]}
    if args.stage in ['legacy','self','rewired','no_func','no_seq','no_omics','no_ppi']:
        config['ppi_inputs']=ppi_options(args.stage)
        config['external_neighbors']=args.stage not in ['self','no_ppi']
        config['neighbor_mode']='none' if args.stage=='no_ppi' else args.stage if args.stage in ['self','rewired'] else 'real'
    if (folder/'protocol.json').is_file():config['data_protocol']=json.loads((folder/'protocol.json').read_text())
    (out/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    torch.save(fit,out/'profile_fit.pt')
    best=float('inf');patience=0;history=[]
    for epoch in range(args.epochs):
        model.train();total=0.;count=0;target_iterator=iter(target_loader) if target_loader is not None else None
        source_iterator=iter(train_loader)
        steps=max(len(train_loader),len(target_loader)) if da is not None and epoch>=args.warmup else len(train_loader)
        for _ in range(steps):
            try:batch=next(source_iterator)
            except StopIteration:source_iterator=iter(train_loader);batch=next(source_iterator)
            data=to_device(batch,device);result=forward(model,data)
            supervised=F.binary_cross_entropy_with_logits(result['logit'],data['label'])
            loss=supervised
            if result['auxiliary']:
                loss=loss+args.aux_weight*torch.stack([F.binary_cross_entropy_with_logits(v,data['label']) for v in result['auxiliary'].values()]).mean()
            if da is not None and epoch>=args.warmup:
                try:target_batch=next(target_iterator)
                except StopIteration:target_iterator=iter(target_loader);target_batch=next(target_iterator)
                assert 'label' not in target_batch
                target_result=forward(model,to_device(target_batch,device))
                progress=(epoch-args.warmup+1)/max(1,args.epochs-args.warmup)
                coefficient=2/(1+math.exp(-10*progress))-1
                loss=loss+args.da_weight*da(result['shared_pair'],result['shared_logit'],
                                           target_result['shared_pair'],target_result['shared_logit'],coefficient)
            if not torch.isfinite(loss):raise FloatingPointError('Non-finite training loss')
            optimizer.zero_grad();loss.backward();optimizer.step()
            total+=float(supervised.detach())*len(data['label']);count+=len(data['label'])
        metrics,_,_,_=evaluate(model,val_loader,device);scheduler.step(metrics['mean_loss'])
        history.append(dict(epoch=epoch,train_loss=total/count,**{'val/'+k:v for k,v in metrics.items()}))
        pd.DataFrame(history).to_csv(out/'history.csv',index=False)
        print(f"epoch={epoch} train={total/count:.5f} val={metrics['mean_loss']:.5f}",flush=True)
        if metrics['mean_loss']<best:
            best=metrics['mean_loss'];patience=0
            torch.save({'model':model.state_dict(),'epoch':epoch,'val_loss':best,
                        'da':da.state_dict() if da is not None else None},out/'best.pt')
        else:
            patience+=1
            if patience>=15:break
    model.load_state_dict(torch.load(out/'best.pt',map_location=device,weights_only=True)['model'])
    # Target labels are opened only after training and source-val selection.
    test=PairDataset(paths['test'],builder(args.target or args.source),args.target or args.source)
    metrics,pairs,labels,logits=evaluate(model,loader(test,batch=64),device)
    pd.DataFrame({'gene1':pairs[:,0],'gene2':pairs[:,1],'label':labels,'prob':logits.sigmoid()}).to_csv(out/'test_predictions.csv',index=False)
    pd.DataFrame([{f'test/{k}':v for k,v in metrics.items()}]).to_csv(out/'metrics.csv',index=False)
    (out/'completed.json').write_text(json.dumps({'test_metrics':metrics,'best_val_loss':best,'epochs_run':len(history)},indent=2)+'\n')
    print('Completed',out,metrics,flush=True)

if __name__=='__main__':main()
