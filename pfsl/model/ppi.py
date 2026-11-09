"""A shallow, attribute-based PPI residual; no node-ID parameters or PyG dependency."""
import torch
from torch import nn


class PPIResidual(nn.Module):
    def __init__(self, seq_dim, text_dim, hidden=64, dropout=0.1, scale=0.1,
                 use_seq=True, use_func=True, use_omics=True, standalone=False):
        super().__init__()
        self.standalone = standalone
        self.use_seq, self.use_func, self.use_omics = use_seq, use_func, use_omics
        parts = 1  # sequence/text availability indicators
        if use_seq:
            self.seq_proj = nn.Linear(seq_dim, hidden)
            parts += 1
        if use_func:
            self.text_proj = nn.Linear(text_dim, hidden)
            parts += 1
        if use_omics:
            self.omics_proj = nn.Linear(8, hidden)
            parts += 1
        self.availability = nn.Linear(2, hidden)
        self.input = nn.Sequential(nn.Linear(parts * hidden, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.self_node = nn.Linear(hidden, hidden)
        self.neighbor = nn.Linear(hidden, hidden, bias=False)
        self.norm = nn.LayerNorm(hidden)
        self.head = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(hidden, 1))
        self.register_buffer('scale', torch.tensor(float(scale)))

    def forward(self, data):
        available = data['ppi_available'].clone()
        available[:, 0] *= self.use_seq
        available[:, 1] *= self.use_func
        parts = [self.availability(available)]
        if self.use_seq:
            parts.append(self.seq_proj(data['ppi_seq']) * available[:, :1])
        if self.use_func:
            parts.append(self.text_proj(data['ppi_text']) * available[:, 1:])
        if self.use_omics:
            parts.append(self.omics_proj(torch.cat([data['ppi_omics'], data['ppi_omics_mask'].float()], dim=-1)))
        x = self.input(torch.cat(parts, dim=-1))
        source, target = data['ppi_edges']
        neighborhood = torch.zeros_like(x)
        neighborhood.index_add_(0, target, x[source])
        neighborhood = neighborhood / data['ppi_degree'].clamp_min(1).unsqueeze(-1)
        h = x + torch.nn.functional.gelu(self.norm(self.self_node(x) + self.neighbor(neighborhood)))
        a, b = data['ppi_pair'].unbind(dim=1)
        pair = torch.cat([h[a] * h[b], (h[a] - h[b]).abs()], dim=-1)
        supported = (data['ppi_degree'][a] + data['ppi_degree'][b] > 0).to(x.dtype)
        # Two isolated genes fall back exactly to the existing PFSL prediction.
        prediction = self.head(pair).squeeze(-1)
        if self.standalone:
            return prediction
        return self.scale * supported * prediction
