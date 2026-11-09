"""Fold-fitted profile preprocessing and label-free multi-view batches."""

import random
import torch
from pfsl.ppi import PPIBatchBuilder


def fit_profile(raw, train_indices):
    raw = raw.float()
    fit = raw[sorted(set(train_indices))]
    if not len(fit):
        raise ValueError("Profile preprocessing needs source training genes")
    observed = torch.isfinite(fit)
    count = observed.sum(0).clamp_min(1)
    mean = torch.where(observed, fit, 0.0).sum(0) / count
    var = torch.where(observed, (fit - mean).square(), 0.0).sum(0) / count
    std = var.sqrt().clamp_min(1.0)  # avoid amplifying nearly constant reference assays
    mask = ~torch.isfinite(raw)
    values = torch.where(mask, torch.zeros_like(raw), (raw - mean) / std)
    return (
        values,
        mask,
        {
            "mean": mean,
            "std": std,
            "fit_indices": torch.tensor(sorted(set(train_indices))),
        },
    )


def rewire(n, pairs, seed=42):
    import networkx as nx

    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    graph.add_edges_from(pairs)
    if len(graph.edges) < 4:
        raise ValueError("Graph too small for a rewiring control")
    nx.double_edge_swap(
        graph, nswap=2 * len(graph.edges), max_tries=100 * len(graph.edges), seed=seed
    )
    return sorted(tuple(sorted(e)) for e in graph.edges)


def csr(n, pairs):
    rows = [set() for _ in range(n)]
    for a, b in pairs:
        if a != b:
            rows[a].add(b)
            rows[b].add(a)
    ptr = [0]
    idx = []
    for row in rows:
        idx.extend(sorted(row))
        ptr.append(len(idx))
    return {"indptr": torch.tensor(ptr), "indices": torch.tensor(idx, dtype=torch.long)}


class NextBatchBuilder(PPIBatchBuilder):
    def __init__(
        self,
        bank,
        omics,
        profile,
        profile_mask,
        views=("ppi",),
        mode="real",
        fanout=16,
        seed=42,
        allow_empty_ess=False,
        omics_statistics=None,
    ):
        super().__init__(bank, omics, fanout, seed, allow_empty_ess=allow_empty_ess, omics_statistics=omics_statistics)
        self.profile, self.profile_mask = profile, profile_mask
        self.views = list(views)
        self.view_neighbors = {}
        for view in views:
            if view == "ppi" and mode == "real":
                neighbors = self.neighbors
            elif mode == "self":
                neighbors = [[i] for i in range(len(self.ids))]
            elif mode == "none":
                neighbors = [[] for _ in self.ids]
            else:
                graph = bank["views"][
                    "ppi_rewired" if view == "ppi" and mode == "rewired" else view
                ]
                ptr, idx = graph["indptr"].tolist(), graph["indices"].tolist()
                neighbors = []
                for i, gid in enumerate(self.ids):
                    row = idx[ptr[i] : ptr[i + 1]]
                    if len(row) > fanout:
                        row = sorted(random.Random(seed + gid).sample(row, fanout))
                    neighbors.append(row)
            self.view_neighbors[view] = neighbors

    def __call__(self, first, second):
        if len(first) != len(second):
            raise ValueError("Mismatched pair lengths")
        absent = (set(map(int, first)) | set(map(int, second))) - self.lookup.keys()
        if absent:
            raise ValueError("Genes absent from bank: " + str(sorted(absent)[:10]))
        roots = sorted({self.lookup[int(g)] for g in list(first) + list(second)})
        nodes = set(roots)
        for neighbors in self.view_neighbors.values():
            for i in roots:
                nodes.update(neighbors[i])
        nodes = sorted(nodes)
        local = {g: i for i, g in enumerate(nodes)}
        pairs = [
            [local[self.lookup[int(a)]], local[self.lookup[int(b)]]]
            for a, b in zip(first, second)
        ]
        result = {
            "ppi_seq": self.bank["seq"][nodes],
            "ppi_text": self.bank["text"][nodes],
            "ppi_available": self.bank["available"][nodes],
            "ppi_omics": self.omics[nodes],
            "ppi_omics_mask": self.omics_mask[nodes],
            "ppi_node_ids": self.bank["gene_ids"][nodes],
            "ppi_profile": self.profile[nodes],
            "ppi_profile_mask": self.profile_mask[nodes],
            "ppi_pair": torch.tensor(pairs, dtype=torch.long),
        }
        for view, neighbors in self.view_neighbors.items():
            edges = [(local[j], local[i]) for i in roots for j in neighbors[i]]
            degree = torch.zeros(len(nodes))
            for i in roots:
                degree[local[i]] = len(neighbors[i])
            result[view + "_edges"] = (
                torch.tensor(edges, dtype=torch.long).reshape(-1, 2).T.contiguous()
            )
            result[view + "_degree"] = degree
        return result


def load_omics(root, cell, name_to_id, allow_missing_ess=False):
    from pathlib import Path

    result = []
    for assay in ["cnv", "ess", "exp", "mut"]:
        mapping = {}
        path = Path(root) / f"{cell}_{assay}.txt"
        if assay == 'ess' and allow_missing_ess and not path.is_file():
            result.append(mapping)
            continue
        for line in path.read_text().splitlines():
            name, value = line.split()
            if name in name_to_id:
                mapping[int(name_to_id[name])] = float(value)
        result.append(mapping)
    return result


class PairDataset(torch.utils.data.Dataset):
    """Unlabeled mode never loads the label column, even if present in the file."""

    def __init__(self, path, builder, context, labeled=True):
        import pandas as pd

        columns = {"gene1", "gene2", "context"} | ({"label"} if labeled else set())
        self.frame = pd.read_csv(path, usecols=lambda name: name in columns)
        if "context" in self.frame and not self.frame.context.eq(context).all():
            raise ValueError("Unexpected context in " + str(path))
        if labeled and "label" not in self.frame:
            raise ValueError("Missing supervised labels")
        self.builder = builder
        self.labeled = labeled
        self.pairs = self.frame[["gene1", "gene2"]].to_numpy(dtype="int64")
        self.labels = (
            torch.tensor(self.frame.label.to_numpy(), dtype=torch.float32)
            if labeled
            else None
        )

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        return i

    def collate(self, indices):
        pairs = self.pairs[indices]
        data = self.builder(pairs[:, 0].tolist(), pairs[:, 1].tolist())
        data["g1_idx"] = torch.tensor(pairs[:, 0])
        data["g2_idx"] = torch.tensor(pairs[:, 1])
        if self.labeled:
            data["label"] = self.labels[indices]
        return data


class MultiContextPairDataset(PairDataset):
    """Graph-free batches with distinct nodes for each (cell, gene)."""

    def __init__(self, path, builders):
        import pandas as pd

        self.frame = pd.read_csv(path)
        if not set(self.frame.context) <= set(builders):
            raise ValueError('Unexpected context in ' + str(path))
        self.builders = builders
        self.labeled = True
        self.pairs = self.frame[['gene1', 'gene2']].to_numpy(dtype='int64')
        self.labels = torch.tensor(self.frame.label.to_numpy(), dtype=torch.float32)

    def collate(self, indices):
        contexts = self.frame.context.iloc[indices].to_numpy()
        pairs = self.pairs[indices]
        pieces = []; pair_parts = []; order = []; offset = 0
        for cell in dict.fromkeys(contexts):
            positions = [i for i, value in enumerate(contexts) if value == cell]
            chosen = pairs[positions]
            part = self.builders[cell](chosen[:, 0].tolist(), chosen[:, 1].tolist())
            if part['ppi_edges'].numel():
                raise ValueError('Multi-context batches require the graph-free model')
            pair_parts.append(part.pop('ppi_pair') + offset)
            offset += len(part['ppi_seq']); pieces.append(part); order.extend(positions)
        data = {key: torch.cat([part[key] for part in pieces], dim=1 if key.endswith('_edges') else 0)
                for key in pieces[0]}
        data['ppi_pair'] = torch.cat(pair_parts)[torch.argsort(torch.tensor(order))]
        data['g1_idx'] = torch.tensor(pairs[:, 0]); data['g2_idx'] = torch.tensor(pairs[:, 1])
        data['label'] = self.labels[indices]
        return data
