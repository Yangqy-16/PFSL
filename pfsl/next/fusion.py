"""Graph-free three-modality fusion; no text/sequence availability features.

Inputs must have observed text and sequence, audited by the experiment runner.
Group order is sequence, functional text, current-cell omics.
"""
import torch
from torch import nn
from .model import symmetric


class NoGraphFusion(nn.Module):
    def __init__(self, seq_dim, text_dim, hidden=64, dropout=0.1, gated=False, ablation='full', random_missing_ess=False, ablation_mode='zero'):
        super().__init__()
        if ablation not in ['full', 'no_text', 'no_seq', 'no_omics', 'omics_only', 'no_ess', 'seq_only', 'text_only']:
            raise ValueError(f'Unknown fusion ablation: {ablation}')
        self.ablation = ablation
        if ablation_mode not in ['zero','remove']:
            raise ValueError(ablation_mode)
        self.ablation_mode = ablation_mode
        active = {'full':['seq','text','omics'], 'no_text':['seq','omics'], 'no_seq':['text','omics'],
                  'no_omics':['seq','text'], 'seq_only':['seq'], 'text_only':['text'],
                  'omics_only':['omics'], 'no_ess':['seq','text','omics']}
        self.active_modalities = active[ablation] if ablation_mode == 'remove' else active['full']
        self.seq_proj = nn.Linear(seq_dim, hidden) if 'seq' in self.active_modalities else None
        self.text_proj = nn.Linear(text_dim, hidden) if 'text' in self.active_modalities else None
        omics_dim = 6 if ablation_mode == 'remove' and ablation == 'no_ess' else 8
        self.omics_proj = nn.Linear(omics_dim, hidden) if 'omics' in self.active_modalities else None
        n = len(self.active_modalities)
        self.input = nn.Sequential(nn.Linear(n*hidden,hidden),nn.LayerNorm(hidden),nn.GELU())
        self.self_node = nn.Linear(hidden,hidden)
        self.norm = nn.LayerNorm(hidden)
        self.head = nn.Sequential(nn.Linear(2*hidden,hidden),nn.GELU(),nn.Dropout(dropout),nn.Linear(hidden,1))
        # Construct after shared layers to preserve their seeded initialization.
        self.gate = nn.Linear(n*hidden,n) if gated else None
        if self.gate is not None:
            nn.init.zeros_(self.gate.weight)
            nn.init.zeros_(self.gate.bias)
        use_missing_ess = random_missing_ess and (ablation_mode == 'zero' or ('omics' in self.active_modalities and ablation != 'no_ess'))
        self.missing_ess = nn.Parameter(torch.empty(1)) if use_missing_ess else None
        if self.missing_ess is not None:
            nn.init.normal_(self.missing_ess, mean=0., std=0.02)

    def forward(self,data):
        if self.missing_ess is not None:
            data = dict(data)
            values = data['ppi_omics'].clone()
            values[:, 1] = torch.where(data['ppi_omics_mask'][:, 1], self.missing_ess[0], values[:, 1])
            data['ppi_omics'] = values
        if self.ablation_mode == 'zero' and self.ablation != 'full':
            # Fixed dimensions/initialization: remove information, not architecture.
            data = dict(data)
            if self.ablation in ['no_text', 'omics_only', 'seq_only']:
                data['ppi_text'] = torch.zeros_like(data['ppi_text'])
            if self.ablation in ['no_seq', 'omics_only', 'text_only']:
                data['ppi_seq'] = torch.zeros_like(data['ppi_seq'])
            if self.ablation in ['no_omics', 'seq_only', 'text_only']:
                for key in ['ppi_omics', 'ppi_omics_mask']:
                    data[key] = torch.zeros_like(data[key])
            if self.ablation == 'no_ess':
                for key in ['ppi_omics', 'ppi_omics_mask']:
                    data[key] = data[key].clone()
                    data[key][:, 1] = 0  # CNV, ESS, EXP, MUT; remove ESS missing bit too.
        parts=[]
        if self.seq_proj is not None: parts.append(self.seq_proj(data['ppi_seq']))
        if self.text_proj is not None: parts.append(self.text_proj(data['ppi_text']))
        if self.omics_proj is not None:
            omics=torch.cat([data['ppi_omics'],data['ppi_omics_mask'].float()],-1)
            if self.ablation_mode == 'remove' and self.ablation == 'no_ess':
                omics=omics[:,[0,2,3,4,6,7]]
            parts.append(self.omics_proj(omics))
        modalities=torch.stack(parts,1)
        n=len(parts)
        joined=modalities.flatten(1)
        weights=(torch.softmax(self.gate(joined),-1) if self.gate is not None
                 else joined.new_full((len(joined),n),1/n))
        # 3*alpha starts at 1: no initial shrinkage relative to plain concatenation.
        if self.gate is not None: joined=(modalities*(n*weights).unsqueeze(-1)).flatten(1)
        x=self.input(joined)
        h=x+torch.nn.functional.gelu(self.norm(self.self_node(x)))
        pair=symmetric(h,data['ppi_pair'])
        return {'logit':self.head(pair).squeeze(-1),'modality_weights':weights,'auxiliary':{}}
