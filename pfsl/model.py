import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import GCNConv


class GCN_pool(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCN_pool, self).__init__()

        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv2 = GCNConv(2 * out_channels, out_channels)
        self.fc1 = torch.nn.Linear(2 * out_channels, out_channels)
        self.fc2 = torch.nn.Linear(out_channels, int(out_channels / 2))
        self.fc3 = torch.nn.Linear(int(out_channels / 2), int(out_channels / 4))
        self.fc4 = torch.nn.Linear(int(out_channels / 4), 1)
        self.dropout = torch.nn.Dropout(0.5)

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

    def decode(self, z, edge_index):
        x = torch.cat((z[edge_index[0]], z[edge_index[1]]), dim=1)
        x = self.fc1(x).relu()
        x = self.dropout(x)
        x = self.fc2(x).relu()
        x = self.dropout(x)
        x = self.fc3(x).relu()
        x = self.dropout(x)
        x = self.fc4(x)
        return torch.squeeze(x)


class GateModule(nn.Module):
    def __init__(self, gnn_dim):
        super().__init__()
        # 接收两个基因的图表征，输出 24 维的门控权重
        self.mlp = nn.Sequential(
            nn.Linear(gnn_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 24),
            # nn.Sigmoid() # 将权重压缩到 0~1 之间
        )
    def forward(self, z_gene1, z_gene2):
        # 拼接两个基因的图表征
        z_concat = torch.cat([z_gene1, z_gene2], dim=-1)
        return self.mlp(z_concat)


class PF_Att(nn.Module):
    def __init__(self, num_aspects: int = 24, embed_dim: int = 1024, n_heads: int = 4, dropout: float = 0.3):
        super().__init__()

        self.output_dim = embed_dim * 2

        # --- 核心改进 1: Aspect Embeddings ---
        # 类似于Positional Encoding，但这里代表功能的类别（定位、通路等）
        # 让模型知道当前处理的是哪个维度的信息
        self.aspect_embedding = nn.Parameter(torch.randn(1, num_aspects, embed_dim))
        
        # --- 核心改进 2: Learnable Missing Tokens ---
        # 专门用于填补缺失值的向量。
        self.missing_tokens = nn.Parameter(torch.randn(1, num_aspects, embed_dim))
        
        # 保持之前的架构
        self.self_attn = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True, dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        
        self.cross_attn = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True, dropout=dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # hidden_dim = min(embed_dim, project_dim * 2)
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim * 2),
            nn.Linear(embed_dim * 2, embed_dim), # 4
            nn.GELU(),
            nn.Dropout(0.1),
            # nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        )

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
        # predict: bool = True
    ) -> Tensor:
        """
        gene_a_feats: [Batch, n, 1024]
        mask_a: [Batch, n] (True where aspect is missing)
        """
        # print('In PF_Att', gene_a_feats.shape, mask_a.shape)
        # --- 数据预处理阶段 ---
        h_a = self.prepare_input(gene_a_feats, mask_a)
        h_b = self.prepare_input(gene_b_feats, mask_b)
        
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
        combined = torch.cat([v_a * v_b, torch.abs(v_a - v_b)], dim=-1) #v_a, v_b, 
        logits = self.classifier(combined)
        
        # if predict:
        #     return logits.squeeze(-1), [att_a, att_b]
        # else:
        return logits.squeeze(-1), [att_a, att_b]
