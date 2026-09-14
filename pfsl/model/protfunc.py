import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from omegaconf import DictConfig

from coach_pl.configuration import configurable
from coach_pl.model import MODEL_REGISTRY


class PF_pre(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()

        embed_dim = cfg.MODEL.FUNC_IN_CHANNELS
        
        num_aspects = cfg.DATASET.ASPECT_NUM
        if cfg.MODEL.FUNC_USE_ASPECT is not None:
            num_aspects = len(str(cfg.MODEL.FUNC_UNUSE_ASPECT).split(','))

        # --- 核心改进 1: Aspect Embeddings ---
        # 类似于Positional Encoding，但这里代表功能的类别（定位、通路等）
        # 让模型知道当前处理的是哪个维度的信息
        self.aspect_embedding = nn.Parameter(torch.randn(1, num_aspects, embed_dim))
        
        # --- 核心改进 2: Learnable Missing Tokens ---
        # 专门用于填补缺失值的向量。
        self.missing_tokens = nn.Parameter(torch.randn(1, num_aspects, embed_dim))

        # 初始化参数
        self._init_weights()
    
    def _init_weights(self):
        # 良好的初始化对收敛很重要
        nn.init.xavier_uniform_(self.aspect_embedding)
        nn.init.normal_(self.missing_tokens, std=0.02)

    def prepare_input(self, x: Tensor, missing_mask: Tensor) -> Tensor:
        """
        x: [Batch, n, 1024] - 原始输入, 缺失位置可以是0或任意值
        missing_mask: [Batch, n] - Boolean, True表示该位置缺失
        """
        batch_size = x.shape[0]
        
        # 1. 填补缺失值
        # 扩展 missing_tokens 到 [Batch, n, 1024]
        expanded_missing = self.missing_tokens.expand(batch_size, -1, -1)
        
        # 使用 where 操作：如果 mask 为 True，使用 learnable missing token，否则使用原始 x
        # unsqueeze(-1) 是为了对齐维度
        mask_expanded = missing_mask.unsqueeze(-1)
        # print('In prepare_input', mask_expanded.shape, expanded_missing.shape, x.shape)
        x_filled = torch.where(mask_expanded, expanded_missing, x)
        
        # 2. 加上 Aspect Embedding (注入位置/类别信息)
        x_final = x_filled + self.aspect_embedding
        
        return x_final

    def forward(self,
        gene_a_feats: Tensor, mask_a: Tensor, 
        gene_b_feats: Tensor, mask_b: Tensor,
    ) -> Tensor:
        """
        gene_a_feats: [Batch, n, 1024]
        mask_a: [Batch, n] (True where aspect is missing)
        """
        h_a = self.prepare_input(gene_a_feats, mask_a)
        h_b = self.prepare_input(gene_b_feats, mask_b)
        
        return h_a, h_b


@MODEL_REGISTRY.register()
class PF_Att(nn.Module):
    @configurable
    def __init__(self, cfg: DictConfig):
        super().__init__()

        embed_dim = cfg.MODEL.FUNC_IN_CHANNELS
        n_heads = cfg.MODEL.NUM_HEADS
        dropout = cfg.MODEL.DROPOUT
        self.output_dim = embed_dim * 2

        self.preprocess = PF_pre(cfg)
        
        # 保持之前的架构
        self.self_attn = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True, dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        
        self.cross_attn = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True, dropout=dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # hidden_dim = min(embed_dim, project_dim * 2)
        # self.classifier = nn.Sequential(
        #     nn.Linear(embed_dim * 2, embed_dim), # 4
        #     nn.ReLU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(embed_dim, embed_dim)
        # )

    @classmethod
    def from_config(cls, cfg):
        return {
            "cfg": cfg
        }

    def forward(self, h_a: Tensor, h_b: Tensor) -> Tensor:
        """
        gene_a_feats: [Batch, n, 1024]
        mask_a: [Batch, n] (True where aspect is missing)
        """
        # --- 下面与之前的逻辑一致，但不需要再传 key_padding_mask 了 ---
        # 因为我们已经用有意义的向量填补了空缺，不再是 Padding，而是特征的一部分
        
        # Step 1: Self-Attention
        h_a = self.norm1(h_a + self.self_attn(h_a, h_a, h_a)[0])
        h_b = self.norm1(h_b + self.self_attn(h_b, h_b, h_b)[0])
        
        # Step 2: Cross-Attention
        # 注意：这里不再需要 key_padding_mask，因为所有的位置现在都有了合理的Embedding
        # 模型会学习到：当它看到 "Missing Token 0" 时，应该如何去注意对方基因
        a_attends_b, att_a = self.cross_attn(h_a, h_b, h_b)
        h_a_context = self.norm2(h_a + a_attends_b)
        
        b_attends_a, att_b = self.cross_attn(h_b, h_a, h_a)
        h_b_context = self.norm2(h_b + b_attends_a)
        
        # Step 3: Pooling
        v_a = h_a_context.max(dim=1)[0]
        v_b = h_b_context.max(dim=1)[0]
        
        # Step 4: Prediction
        # combined = torch.cat([v_a * v_b, torch.abs(v_a - v_b)], dim=-1) #v_a, v_b, 
        # logits = self.classifier(combined)
        
        # if predict:
        #     return logits.squeeze(-1), [att_a, att_b]
        # else:
        return v_a, v_b, [att_a, att_b] #combined
