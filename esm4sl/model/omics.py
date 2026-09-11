import torch
import torch.nn as nn

class CellLineFiLM(nn.Module):
    def __init__(self, scalar_dim=4, hidden_dim=256):
        """
        scalar_dim: 细胞系标量特征的维度 (例如：CCLE表达量、突变数等4个特征)
        hidden_dim: 基因/蛋白质高维表征的维度 (主干网络的隐藏层维度)
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # 用一个 MLP 将 4 维的标量映射为缩放(gamma)和平移(beta)参数
        # 考虑到标量维度很低，中间层可以适当展宽，输出维度为 2 * hidden_dim
        self.mlp = nn.Sequential(
            nn.Linear(scalar_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim * 2)
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, cell_line_features, gene_embeddings):
        """
        cell_line_features: [batch_size, scalar_dim]
        gene_embeddings: [batch_size, seq_len, hidden_dim] 或 [batch_size, hidden_dim]
        """
        # 1. 计算调制参数
        # params shape: [batch_size, 2 * hidden_dim]
        params = self.mlp(cell_line_features)

        # 2. 分割出 gamma 和 beta
        # shape 均为: [batch_size, hidden_dim]
        gamma, beta = torch.chunk(params, 2, dim=-1)
        
        # 让网络学习增量，默认 gamma 为 1
        gamma = gamma + 1.0
        
        # 3. 维度对齐 (Broadcasting)
        # 如果你的基因表征包含序列长度维度 (如 Language Model 提取的特征序列)
        # 需要扩展 gamma 和 beta 的维度以便进行广播
        if gene_embeddings.dim() == 3:
            gamma = gamma.unsqueeze(1)  # [batch_size, 1, hidden_dim]
            beta = beta.unsqueeze(1)    # [batch_size, 1, hidden_dim]
            
        # 4. 施加线性调制
        # 建议在使用前，确保 gene_embeddings 已经经过了 LayerNorm 等标准化处理
        modulated_embeddings = gamma * gene_embeddings + beta
        
        return modulated_embeddings

# --- 测试代码 ---
if __name__ == "__main__":
    batch_size = 32
    hidden_dim = 256
    
    # 模拟输入数据
    cell_features = torch.randn(batch_size, 4) # 4个 scalar
    gene_features = torch.randn(batch_size, 1024, hidden_dim) # 假设序列长度为 1024
    
    # 实例化并前向传播
    film_layer = CellLineFiLM(scalar_dim=4, hidden_dim=hidden_dim)
    output = film_layer(cell_features, gene_features)
    
    print(f"输入基因特征 shape: {gene_features.shape}")
    print(f"FiLM调制后特征 shape: {output.shape}")
