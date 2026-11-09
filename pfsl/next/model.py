"""Attribute gates, view gates, and optional private cell-context residual."""

import torch
from torch import nn


def symmetric(h, pair):
    a, b = pair.unbind(1)
    return torch.cat([h[a] * h[b], (h[a] - h[b]).abs()], -1)


def head(dim, hidden, dropout):
    return nn.Sequential(
        nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 1)
    )


class NextGraph(nn.Module):
    def __init__(
        self,
        seq_dim,
        text_dim,
        profile_dim=0,
        views=("ppi",),
        hidden=64,
        split_context=False,
        auxiliary=False,
        dropout=0.1,
    ):
        super().__init__()
        self.views = list(views)
        self.split_context = split_context
        self.profile_dim = profile_dim
        dims = {"seq": seq_dim, "text": text_dim}
        if profile_dim:
            dims["profile"] = profile_dim * 2
        if not split_context:
            dims["omics"] = 8
        self.projections = nn.ModuleDict(
            {k: nn.Linear(d, hidden) for k, d in dims.items()}
        )
        self.modality_gate = nn.Linear(hidden, 1)
        self.empty = nn.Parameter(torch.zeros(hidden))
        self.input_norm = nn.LayerNorm(hidden)
        self.self_node = nn.Linear(hidden, hidden)
        self.neighbors = nn.ModuleDict(
            {v: nn.Linear(hidden, hidden, bias=False) for v in views}
        )
        self.norms = nn.ModuleDict({v: nn.LayerNorm(hidden) for v in views})
        self.view_gate = nn.Linear(hidden, 1)
        self.base_head = head(hidden * 2, hidden, dropout)
        if split_context:
            # The shared branch has limited context interaction, so alignment
            # need not operate on identical sequence/text representations.
            self.shared_context = nn.Linear(8, hidden)
            self.private_film = nn.Sequential(
                nn.Linear(8, hidden), nn.GELU(), nn.Linear(hidden, hidden * 2)
            )
            self.private_head = head(hidden * 4, hidden, dropout)
            nn.init.zeros_(self.private_head[-1].weight)
            nn.init.zeros_(self.private_head[-1].bias)
        self.aux_heads = (
            nn.ModuleDict({v: head(hidden * 2, hidden, dropout) for v in views})
            if auxiliary
            else nn.ModuleDict()
        )

    def forward(self, data):
        omics = torch.cat([data["ppi_omics"], data["ppi_omics_mask"].float()], -1)
        values = {"seq": data["ppi_seq"], "text": data["ppi_text"], "omics": omics}
        available = {
            "seq": data["ppi_available"][:, 0].bool(),
            "text": data["ppi_available"][:, 1].bool(),
            "omics": (~data["ppi_omics_mask"]).any(1),
        }
        if self.profile_dim:
            values["profile"] = torch.cat(
                [data["ppi_profile"], data["ppi_profile_mask"].float()], -1
            )
            available["profile"] = (~data["ppi_profile_mask"]).any(1)
        tokens = [
            torch.nn.functional.gelu(layer(values[k]))
            for k, layer in self.projections.items()
        ]
        valid = [available[k] for k in self.projections]
        # A fallback token is available only where every modality is missing.
        missing = ~torch.stack(valid, 1).any(1)
        tokens.append(self.empty.expand(len(omics), -1))
        valid.append(missing)
        tokens = torch.stack(tokens, 1)
        valid = torch.stack(valid, 1)
        weights = torch.softmax(
            self.modality_gate(tokens).squeeze(-1).masked_fill(~valid, -1e9), 1
        )
        x = self.input_norm((tokens * weights.unsqueeze(-1)).sum(1))
        embeddings = {"self": x}
        for view in self.views:
            source, target = data[view + "_edges"]
            agg = torch.zeros_like(x)
            agg.index_add_(0, target, x[source])
            agg = agg / data[view + "_degree"].clamp_min(1).unsqueeze(-1)
            embeddings[view] = x + torch.nn.functional.gelu(
                self.norms[view](self.self_node(x) + self.neighbors[view](agg))
            )
        stacked = torch.stack(list(embeddings.values()), 1)
        support = torch.stack(
            [torch.ones(len(x), dtype=torch.bool, device=x.device)]
            + [data[v + "_degree"] > 0 for v in self.views],
            1,
        )
        view_weights = torch.softmax(
            self.view_gate(stacked).squeeze(-1).masked_fill(~support, -1e9), 1
        )
        h = (stacked * view_weights.unsqueeze(-1)).sum(1)
        prior_pair = symmetric(h, data["ppi_pair"])
        shared = (
            h + 0.1 * torch.tanh(self.shared_context(omics))
            if self.split_context
            else h
        )
        pair = symmetric(shared, data["ppi_pair"])
        base = self.base_head(pair).squeeze(-1)
        logit = base
        if self.split_context:
            gamma, beta = self.private_film(omics).chunk(2, -1)
            private = h * torch.tanh(gamma) + beta
            residual = self.private_head(
                torch.cat([prior_pair, symmetric(private, data["ppi_pair"])], -1)
            ).squeeze(-1)
            logit = base + residual
        auxiliary = {
            v: layer(symmetric(embeddings[v], data["ppi_pair"])).squeeze(-1)
            for v, layer in self.aux_heads.items()
        }
        return {
            "logit": logit,
            "shared_logit": base,
            "shared_pair": pair,
            "auxiliary": auxiliary,
            "modality_weights": weights,
            "view_weights": view_weights,
        }
