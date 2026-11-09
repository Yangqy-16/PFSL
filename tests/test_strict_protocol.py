from pathlib import Path
import sys
import torch
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pfsl.next.fusion import NoGraphFusion


def test_holdout_values_do_not_change_training_normalization():
    from pfsl.omics_scaling import fit_omics, transform_omics
    a=[{1:2.,2:4.,3:100.} for _ in range(4)]
    b=[{1:2.,2:4.,3:-10000.,99:1e10} for _ in range(4)]
    fa,fb=fit_omics(a,[1,2,1]),fit_omics(b,[1,2])
    assert fa==fb
    assert fa['mean']==[3.]*4 and fa['scale']==[1.]*4
    x,_=transform_omics(a,[1,2,3],fa);y,_=transform_omics(b,[1,2,3],fa)
    assert torch.equal(x[:2],y[:2])
    assert x[:,0].tolist()==[-1.,1.,97.]
    assert not torch.equal(x[2],y[2])


def test_missing_and_constant_assays_never_use_holdout_statistics():
    from pfsl.omics_scaling import fit_omics,transform_omics
    raw=[{1:7.,2:7.,3:8.},{3:123.},{1:float('nan'),2:5.},{}]
    fit=fit_omics(raw,[1,2]);x,mask=transform_omics(raw,[1,2,3],fit)
    assert fit['count']==[2,0,1,0]
    assert fit['scale']==[1.,1.,1.,1.]
    assert fit['mean']==[7.,0.,5.,0.]
    assert x[0,2]==0 and mask[0,2]
    assert x[2,1]==123 and not mask[2,1]
    assert mask[:,3].all() and torch.isfinite(x).all()


@pytest.mark.parametrize('variant,active',[
 ('full',['seq','text','omics']),('no_text',['seq','omics']),('no_seq',['text','omics']),
 ('no_omics',['seq','text']),('seq_only',['seq']),('text_only',['text']),('omics_only',['omics'])])
def test_structural_ablation_removes_parameters_and_ignores_absent_inputs(variant,active):
    model=NoGraphFusion(7,9,ablation=variant,ablation_mode='remove').eval()
    assert model.input[0].in_features==64*len(active)
    for name in ['seq','text','omics']:
        params=[n for n,_ in model.named_parameters() if n.startswith(name+'_proj.')]
        assert bool(params)==(name in active)
    batch={'ppi_pair':torch.tensor([[0,1],[2,3]])}
    if 'seq' in active:batch['ppi_seq']=torch.randn(4,7)
    if 'text' in active:batch['ppi_text']=torch.randn(4,9)
    if 'omics' in active:batch.update(ppi_omics=torch.randn(4,4),ppi_omics_mask=torch.zeros(4,4,dtype=torch.bool))
    y=model(batch)['logit'];assert y.shape==(2,) and torch.isfinite(y).all()
    batch['ppi_pair']=batch['ppi_pair'].flip(1)
    assert torch.equal(y,model(batch)['logit'])


def test_full_model_is_identical_across_ablation_modes():
    torch.manual_seed(42);legacy=NoGraphFusion(7,9)
    torch.manual_seed(42);strict=NoGraphFusion(7,9,ablation_mode='remove')
    assert legacy.state_dict().keys()==strict.state_dict().keys()
    assert all(torch.equal(v,strict.state_dict()[k]) for k,v in legacy.state_dict().items())
