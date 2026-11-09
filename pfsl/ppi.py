"""Label-free PPI feature bank and deterministic one-hop mini-batches."""
import random

import pandas as pd
import torch


def physical_edges(path, gene_ids):
    """Read BioGRID Entrez IDs; exclude genetic interactions and self edges."""
    columns = ['Entrez Gene Interactor A', 'Entrez Gene Interactor B',
               'Experimental System Type']
    df = pd.read_csv(path, usecols=columns)
    rows = df.loc[df[columns[2]].str.lower().eq('physical'), columns[:2]]
    lookup = {int(g): i for i, g in enumerate(gene_ids)}
    pairs = set()
    for a, b in rows.itertuples(index=False, name=None):
        a, b = int(a), int(b)
        if a != b and a in lookup and b in lookup:
            pairs.add(tuple(sorted((lookup[a], lookup[b]))))
    return sorted(pairs), {'input_rows': len(df), 'physical_rows': len(rows),
                           'physical_edges': len(pairs), 'nodes': len(lookup)}


class PPIBatchBuilder:
    """One-hop sampling depends on gene ID/seed, never batch composition or labels.

    All background nodes may be present: this is SL-label cold start, not a
    promise that held-out nodes were absent from the background graph.
    """
    def __init__(self, bank, omics, fanout=16, seed=42, allow_empty_ess=False, omics_statistics=None):
        if bank.get('version') != 1 or fanout < 1:
            raise ValueError('Unsupported PPI bank version or non-positive fanout')
        self.bank = bank
        self.ids = bank['gene_ids'].tolist()
        self.lookup = {g: i for i, g in enumerate(self.ids)}
        if len(self.lookup) != len(self.ids):
            raise ValueError('Duplicate PPI gene IDs')
        self.neighbors = []
        ptr, indices = bank['indptr'].tolist(), bank['indices'].tolist()
        for i, gid in enumerate(self.ids):
            neighbors = indices[ptr[i]:ptr[i + 1]]
            if len(neighbors) > fanout:
                neighbors = sorted(random.Random(seed + gid).sample(neighbors, fanout))
            self.neighbors.append(neighbors)
        self.omics = torch.zeros(len(self.ids), 4)
        self.omics_mask = torch.ones(len(self.ids), 4, dtype=torch.bool)
        if omics_statistics is not None:
            from pfsl.omics_scaling import transform_omics
            self.omics, self.omics_mask = transform_omics(omics, self.ids, omics_statistics)
            self.omics_statistics = omics_statistics
            return
        # Standardize each assay within its cell, using observed values only.
        for col, mapping in enumerate(omics):
            values = torch.tensor(list(mapping.values()), dtype=torch.float32)
            values = values[torch.isfinite(values)]
            if not len(values):
                if allow_empty_ess and col == 1:
                    continue  # zero placeholder and missing bit; model supplies learned token
                raise ValueError('PPI omics assay contains no finite observations')
            mean, std = values.mean(), values.std(unbiased=False).clamp_min(1e-6)
            for gid, value in mapping.items():
                if gid in self.lookup and torch.isfinite(torch.tensor(value)):
                    idx = self.lookup[gid]
                    self.omics[idx, col] = (value - mean) / std
                    self.omics_mask[idx, col] = False

    @classmethod
    def from_config(cls, cfg, omics):
        path = cfg.MODEL.PPI.BANK_PATH
        if not path:
            raise ValueError('USE_PPI requires MODEL.PPI.BANK_PATH; run script/prepare_ppi.py')
        bank = torch.load(path, map_location='cpu', weights_only=True)
        if bank['seq'].shape[1] != cfg.MODEL.SEQ_IN_CHANNELS or bank['text'].shape[1] != cfg.MODEL.FUNC_IN_CHANNELS:
            raise ValueError('PPI bank feature dimensions do not match the model')
        expected_aspects = (list(range(cfg.DATASET.ASPECT_NUM)) if cfg.MODEL.FUNC_USE_ASPECT is None
                            else [int(i) for i in str(cfg.MODEL.FUNC_USE_ASPECT).split(',')])
        if bank['metadata']['aspects'] != expected_aspects:
            raise ValueError('PPI bank text aspects differ from MODEL.FUNC_USE_ASPECT')
        return cls(bank, omics, cfg.MODEL.PPI.FANOUT, cfg.MODEL.PPI.SAMPLE_SEED)

    def __call__(self, first, second):
        gids = [int(g) for g in first] + [int(g) for g in second]
        absent = sorted(set(gids) - self.lookup.keys())
        if absent:
            raise ValueError(f'Pair genes absent from PPI bank: {absent[:10]}')
        roots = sorted({self.lookup[g] for g in gids})
        nodes = sorted(set(roots).union(*(self.neighbors[i] for i in roots)))
        local = {idx: i for i, idx in enumerate(nodes)}
        edges = [(local[j], local[i]) for i in roots for j in self.neighbors[i]]
        edge_index = torch.tensor(edges, dtype=torch.long).reshape(-1, 2).T.contiguous()
        degree = torch.zeros(len(nodes))
        for i in roots:
            degree[local[i]] = len(self.neighbors[i])
        pairs = [[local[self.lookup[int(a)]], local[self.lookup[int(b)]]]
                 for a, b in zip(first, second)]
        return {'ppi_seq': self.bank['seq'][nodes], 'ppi_text': self.bank['text'][nodes],
                'ppi_available': self.bank['available'][nodes],
                'ppi_omics': self.omics[nodes], 'ppi_omics_mask': self.omics_mask[nodes],
                'ppi_edges': edge_index, 'ppi_degree': degree,
                'ppi_pair': torch.tensor(pairs, dtype=torch.long),
                'ppi_node_ids': self.bank['gene_ids'][nodes]}
