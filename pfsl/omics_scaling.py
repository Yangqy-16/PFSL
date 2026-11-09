"""Fit gene-level omics statistics on unique training genes only."""
import math
import torch


def fit_omics(omics, training_gene_ids):
    ids=sorted(set(map(int,training_gene_ids)))
    if not ids:
        raise ValueError('Cannot fit omics without training genes')
    if len(omics)!=4:
        raise ValueError('Expected CNV, ESS, EXP, MUT')
    means=[];scales=[];counts=[];fallback=[]
    for mapping in omics:
        values=[float(mapping[g]) for g in ids if g in mapping and math.isfinite(float(mapping[g]))]
        count=len(values)
        if count:
            x=torch.tensor(values,dtype=torch.float64)
            mean=float(x.mean());std=float(x.std(unbiased=False))
        else:mean=0.;std=0.
        means.append(mean);scales.append(std if std>1e-6 else 1.);counts.append(count)
        fallback.append('no_training_observations' if count==0 else 'constant_training_values' if std<=1e-6 else None)
    return dict(protocol='unique_training_genes',fit_gene_ids=ids,assays=['CNV','ESS','EXP','MUT'],
                mean=means,scale=scales,count=counts,fallback=fallback,
                missing_value=0.,constant_or_empty_scale=1.)


def transform_omics(omics,gene_ids,fit):
    if len(omics)!=4 or len(fit['mean'])!=4 or len(fit['scale'])!=4:
        raise ValueError('Expected four assay statistics')
    out=torch.zeros(len(gene_ids),4,dtype=torch.float32)
    missing=torch.ones(len(gene_ids),4,dtype=torch.bool)
    for col,mapping in enumerate(omics):
        mean,scale=fit['mean'][col],fit['scale'][col]
        if not math.isfinite(mean) or not math.isfinite(scale) or scale<=0:
            raise ValueError('Invalid fixed omics statistics')
        for i,g in enumerate(gene_ids):
            value=float(mapping.get(int(g),float('nan')))
            if math.isfinite(value):
                out[i,col]=(value-mean)/scale;missing[i,col]=False
    if not torch.isfinite(out).all():
        raise ValueError('Nonfinite transformed omics')
    return out,missing
