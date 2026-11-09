from torch import Tensor
from .attn_wrap import *
from .protfunc import *
from .omics import CellLineFiLM
from .calibrator import FeatureCalibrator
from .ppi import PPIResidual


def padded_to_variadic(rep: Tensor, mask: Tensor) -> list[Tensor]:
    num_samples = mask.size(0)
    var_reps = []

    for i in range(num_samples):
        valid_num = (~mask[i]).sum().item()
        real_rep = rep[i, :valid_num]
        var_reps.append(real_rep)

    return var_reps


class Gating(nn.Module):
    """
    Gating network to generate weights for each expert.
    """
    def __init__(self, num_experts=3, embed_dim=256):
        super().__init__()
        self.num_experts = num_experts
        self.embed_dim = embed_dim
        
        self.embedding_fusion = nn.Sequential(
            nn.Linear(num_experts * embed_dim, embed_dim),
            nn.ReLU(),
            nn.LayerNorm(embed_dim)
        )
        
        self.weight_generator = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * num_experts),
            nn.Unflatten(1, (embed_dim, num_experts))
        )
    
    def forward(self, expert_embeddings: list[Tensor]):
        """
        Args:
            expert_embeddings: list, each element of size [batch_size, embed_dim]
        Returns:
            weights: [batch_size, embed_dim, num_experts]
        """
        fused_emb = torch.cat(expert_embeddings, dim=1)  # [B, 3*E]
        gate_feat = self.embedding_fusion(fused_emb)  # [B, E]
        weights = self.weight_generator(gate_feat)  # [B, E, 3]
        return torch.softmax(weights, dim=2)  # sum of weights = 1


@MODEL_REGISTRY.register()
class ProSL(nn.Module):

    @configurable
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.ppi_only = cfg.MODEL.get('PPI_ONLY', False)
        if self.ppi_only:
            if not cfg.MODEL.get('USE_PPI', False):
                raise ValueError('PPI_ONLY requires USE_PPI=True')
            ppi = cfg.MODEL.PPI
            self.ppi_branch = PPIResidual(
                cfg.MODEL.SEQ_IN_CHANNELS, cfg.MODEL.FUNC_IN_CHANNELS,
                ppi.HIDDEN_DIM, ppi.DROPOUT, 1.0,
                cfg.MODEL.USE_SEQ, cfg.MODEL.USE_FUNC, cfg.MODEL.USE_OMICS,
                standalone=True,
            )
            return
        
        self.modal_num = cfg.MODEL.USE_SEQ + cfg.MODEL.USE_FUNC
        assert self.modal_num >= 1

        if cfg.MODEL.USE_SEQ:
            if cfg.MODEL.SEQ_USE_ATT:
                self.seq_encoder = AttnWrap(cfg)
            #     self.seq_downsample = MLP(in_features=cfg.MODEL.EMBED_DIM * 2, out_features=cfg.MODEL.EMBED_DIM, drop=cfg.MODEL.DROPOUT)
            # else:
            #     self.seq_downsample = MLP(in_features=cfg.MODEL.SEQ_IN_CHANNELS * 2, out_features=cfg.MODEL.EMBED_DIM, drop=cfg.MODEL.DROPOUT)
            self.seq_downsample = MLP(in_features=cfg.MODEL.SEQ_IN_CHANNELS, out_features=cfg.MODEL.EMBED_DIM, drop=cfg.MODEL.DROPOUT)

        if cfg.MODEL.USE_FUNC:
            if cfg.MODEL.FUNC_USE_ATT:
                self.func_encoder = PF_Att(cfg)
            
            self.pf_pre = PF_pre(cfg)
            self.func_downsample = MLP(in_features=cfg.MODEL.FUNC_IN_CHANNELS, out_features=cfg.MODEL.EMBED_DIM, drop=cfg.MODEL.DROPOUT) # * 2

        if cfg.MODEL.USE_CL:
            self.cl_downsample = MLP(in_features=cfg.MODEL.CL_IN_CHANNELS, out_features=cfg.MODEL.EMBED_DIM, drop=cfg.MODEL.DROPOUT)

        if cfg.MODEL.USE_OMICS:
            self.omics_film = CellLineFiLM(
                hidden_dim=cfg.MODEL.EMBED_DIM * self.modal_num
            )
        
        if cfg.MODEL.CALIBRATE:
            self.calibrator = FeatureCalibrator(cfg.MODEL.FUNC_IN_CHANNELS)

        # if self.modal_num > 1:
        #     self.gating = Gating(num_experts=self.modal_num, embed_dim=cfg.MODEL.EMBED_DIM)
        #     self.out_classifier = MLP(in_features=cfg.MODEL.EMBED_DIM, out_features=1)
        # else:
        self.out_classifier = nn.Linear(in_features=cfg.MODEL.EMBED_DIM * self.modal_num * 2, out_features=1)
        if cfg.MODEL.get('USE_PPI', False):
            ppi = cfg.MODEL.PPI
            self.ppi_branch = PPIResidual(
                cfg.MODEL.SEQ_IN_CHANNELS, cfg.MODEL.FUNC_IN_CHANNELS,
                ppi.HIDDEN_DIM, ppi.DROPOUT, ppi.RESIDUAL_SCALE,
                cfg.MODEL.USE_SEQ, cfg.MODEL.USE_FUNC, cfg.MODEL.USE_OMICS,
            )

    @classmethod
    def from_config(cls, cfg):
        return {
            "cfg": cfg,
        }

    def forward(self, data: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        if self.ppi_only:
            return self.ppi_branch(data), []
        fused_emb = []
        att = []
        g1_parts = []
        g2_parts = []

        if self.cfg.MODEL.USE_SEQ:
            if self.cfg.MODEL.SEQ_USE_ATT:
                g1_seq_emb, g2_seq_emb, seq_att = self.seq_encoder(data['g1_seq_emb'], data['g1_seq_mask'], data['g2_seq_emb'], data['g2_seq_mask'], predict=False)
                att.append(seq_att)
            else:
                g1_seq_emb, g2_seq_emb = data['g1_seq_emb'], data['g2_seq_emb']
            #     seq_emb = torch.cat([g1_seq_emb * g2_seq_emb, torch.abs(g1_seq_emb - g2_seq_emb)], dim=-1)
            #     seq_att = None
            
            g1_seq_emb = self.seq_downsample(g1_seq_emb)
            g2_seq_emb = self.seq_downsample(g2_seq_emb)
            g1_parts.append(g1_seq_emb)
            g2_parts.append(g2_seq_emb)
            # print(g1_seq_emb.shape, g2_seq_emb.shape)
            # fused_emb.append(self.seq_downsample(seq_emb))
        
        if self.cfg.MODEL.USE_FUNC:
            if self.cfg.MODEL.FUNC_USE_ATT:
                g1_func_emb, g2_func_emb = self.pf_pre(data['g1_func_emb'], data['g1_func_mask'], data['g2_func_emb'], data['g2_func_mask'])
            else:
                # ProtDataset already averages valid functional-text aspects
                # when attention is disabled; there are no aspect masks here.
                g1_func_emb, g2_func_emb = data['g1_func_emb'], data['g2_func_emb']
            
            # if self.cfg.MODEL.CALIBRATE:
            #     g1_func_emb = self.calibrator(g1_func_emb)
            #     g2_func_emb = self.calibrator(g2_func_emb)
            
            if self.cfg.MODEL.FUNC_USE_ATT:
                g1_func_emb, g2_func_emb, func_att = self.func_encoder(g1_func_emb, g2_func_emb) #func_emb
                att.append(func_att)
            # else:
            #     func_emb = torch.cat([g1_func_emb * g2_func_emb, torch.abs(g1_func_emb - g2_func_emb)], dim=-1)
            #     func_att = None

            g1_func_emb = self.func_downsample(g1_func_emb)
            g2_func_emb = self.func_downsample(g2_func_emb)
            g1_parts.append(g1_func_emb)
            g2_parts.append(g2_func_emb)
            # print(g1_func_emb.shape, g2_func_emb.shape)
            # fused_emb.append(self.func_downsample(func_emb))

        if self.cfg.MODEL.USE_CL:
            fused_emb.append(self.cl_downsample(data['cl_emb']))

        g1_embed = (
            torch.cat(g1_parts, dim=1) if len(g1_parts) > 1 else g1_parts[0]
        )
        g2_embed = (
            torch.cat(g2_parts, dim=1) if len(g2_parts) > 1 else g2_parts[0]
        )

        if self.cfg.MODEL.USE_OMICS:
            g1_omics, g2_omics = data['g1_omics'], data['g2_omics']

            g1_embed = self.omics_film(g1_omics, g1_embed)
            g2_embed = self.omics_film(g2_omics, g2_embed)

        # if self.modal_num > 1:
        #     W = self.gating(fused_emb)  # [B, E, 3]
        #     H = torch.stack(fused_emb, dim=1)  # [B, 3, E]
        #     fused_emb = torch.einsum('bme, bem -> be', H, W)  # [B, E]
        #     output = self.out_classifier(fused_emb)
        # else:
        combined = torch.cat([g1_embed * g2_embed, torch.abs(g1_embed - g2_embed)], dim=-1)
        output = self.out_classifier(combined) #torch.cat([g1_embed, g2_embed], dim=1)  fused_emb
        
        output = output.squeeze(-1)
        if hasattr(self, 'ppi_branch'):
            output = output + self.ppi_branch(data)
        return output, att


# @MODEL_REGISTRY.register()
# class PF_MLP(nn.Module):

#     @configurable
#     def __init__(self, in_channels: int = 1280, kernel_size: int = 3,
#                  embed_dim: int = 256, num_heads: int = 4, num_layers: int = 1, 
#                  linear_depth: int = 2, dropout: float = 0.1, out_channels: int = 1,
#                  pooling: str = 'mean', use_cell: bool | None = None) -> None:
#         super().__init__()
        
#         self.func_mlp = nn.Sequential(
#             nn.Linear(in_features=1024, out_features=1024),
#             nn.ReLU(),
#             nn.Dropout(p=dropout),
#             nn.Linear(in_features=1024, out_features=embed_dim),
#         )

#         self.head = MLP(in_features=embed_dim*2, out_features=out_channels)#, norm_layer=nn.LayerNorm

#     @classmethod
#     def from_config(cls, cfg):
#         return {
#             "in_channels": cfg.MODEL.IN_CHANNELS,
#             "kernel_size": cfg.MODEL.KERNEL_SIZE,
#             "embed_dim": cfg.MODEL.EMBED_DIM,
#             "num_heads": cfg.MODEL.NUM_HEADS,
#             "num_layers": cfg.MODEL.NUM_LAYERS,
#             "linear_depth": cfg.MODEL.LINEAR_DEPTH,
#             "dropout": cfg.MODEL.DROPOUT,
#             "out_channels": cfg.MODEL.OUT_CHANNELS,
#             "use_cell": cfg.DATASET.CELL_LINE,
#         }

#     def forward(self, g1_func: Tensor, g2_func: Tensor, predict: bool = True) -> tuple[Tensor, Tensor]:
#         g1_func = self.func_mlp(g1_func)  # [B, Eo]
#         g2_func = self.func_mlp(g2_func)
#         gene_embed = torch.cat([g1_func, g2_func], dim=1)

#         if predict:
#             output = self.head(gene_embed)  # [B, 1]
#             return output
#         else:
#             return gene_embed
