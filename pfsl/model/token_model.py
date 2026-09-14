import torch
import torch.nn as nn
import math

from coach_pl.configuration import configurable
from coach_pl.model import MODEL_REGISTRY


class FeatureTokenizer(nn.Module):
    """
    将低维标量（如4维组学数据）转化为独立的 Token 向量
    """
    def __init__(self, num_features=4, d_model=512):
        super().__init__()
        # 为每个标量特征分配独立的权重和偏置
        self.weight = nn.Parameter(torch.randn(num_features, d_model))
        self.bias = nn.Parameter(torch.zeros(num_features, d_model))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x):
        # x: [batch_size, num_features]
        # 返回: [batch_size, num_features, d_model]
        return x.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)


class SequenceCompressor(nn.Module):
    """
    使用一维卷积对超长蛋白质序列进行局部降采样，缓解 Transformer 的 OOM 风险
    """
    def __init__(self, in_dim=1280, d_model=512, compress_factor=4):
        super().__init__()
        # stride = compress_factor 实现序列长度的等比例缩短
        self.conv = nn.Conv1d(
            in_channels=in_dim, 
            out_channels=d_model,
            kernel_size=compress_factor, 
            stride=compress_factor
        )

    def forward(self, x):
        # x: [batch_size, seq_len, in_dim]
        x = x.transpose(1, 2)            # [batch_size, in_dim, seq_len]
        x = self.conv(x)                 # [batch_size, d_model, seq_len // factor]
        return x.transpose(1, 2)         # [batch_size, seq_len // factor, d_model]


class UnifiedGeneEncoder(nn.Module):
    """
    单基因的统一深层全自注意力编码器
    """
    def __init__(self, d_model=512, nhead=8, num_layers=4, max_seq_len=1000, compress_factor=4): #
        super().__init__()
        self.compress_factor = compress_factor # 需要保存这个属性用于 mask 压缩

        self.missing_tokens = nn.Parameter(torch.randn(1, 24, 1024))
        
        # 1. 各模态的投影层
        self.seq_proj = SequenceCompressor(1280, d_model, compress_factor)
        self.text_proj = nn.Linear(1024, d_model)
        self.omics_proj = FeatureTokenizer(4, d_model)
        self.global_proj = nn.Linear(768, d_model)
        
        # 2. [CLS] Token 声明
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # 3. 模态类型嵌入 (Modality Type Embeddings)
        # 告诉 Transformer 不同的 Token 来源于哪种物理意义的数据
        self.modality_emb = nn.ParameterDict({
            'cls': nn.Parameter(torch.randn(1, 1, d_model)),
            'seq': nn.Parameter(torch.randn(1, 1, d_model)),
            'text': nn.Parameter(torch.randn(1, 1, d_model)),
            'omics': nn.Parameter(torch.randn(1, 1, d_model)),
            'global': nn.Parameter(torch.randn(1, 1, d_model))
        })
        
        # 4. 序列位置嵌入 (Positional Embedding for Sequence)
        # 只有序列需要位置信息，其他模态（如文本方面、组学特征）本质上是无序集合(Set)
        compressed_max_len = max_seq_len // compress_factor
        self.seq_pos_emb = nn.Parameter(torch.randn(1, compressed_max_len, d_model))
        
        # 5. 主干 Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            batch_first=True, 
            activation='gelu',
            norm_first=True # 深层 Transformer 推荐 Pre-Norm 结构
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def prepare_text_input(self, x, missing_mask):
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
        # x_final = x_filled + self.aspect_embedding
        
        return x_filled

    def forward(self, seq, seq_padding_mask, text, text_mask, omics, global_ctx):
        B = seq.size(0)
        
        # --- 步骤 1: 特征投影与模态信息注入 ---
        # 序列 Token (附加位置编码和模态编码)
        t_seq = self.seq_proj(seq) # [B, L_comp, d_model]
        L_comp = t_seq.size(1)
        t_seq = t_seq + self.seq_pos_emb[:, :L_comp, :] + self.modality_emb['seq']
        
        # 文本 Token
        text = self.prepare_text_input(text, text_mask)
        t_text = self.text_proj(text) + self.modality_emb['text'] # [B, 24, d_model]
        
        # 组学 Token
        t_omics = self.omics_proj(omics) + self.modality_emb['omics'] # [B, 4, d_model]
        
        # 全局背景 Token
        t_global = self.global_proj(global_ctx).unsqueeze(1) + self.modality_emb['global'] # [B, 1, d_model]
        
        # CLS Token
        t_cls = self.cls_token.expand(B, -1, -1) + self.modality_emb['cls'] # [B, 1, d_model]
        
        # --- 步骤 2: 全序列拼接 ---
        # 拼接顺序无关紧要，因为自注意力是置换不变的 (除了加上了位置编码的 t_seq)
        # Shape: [B, 1 + L_comp + 24 + 4 + 1, d_model]
        tokens = torch.cat([t_cls, t_seq, t_text, t_omics, t_global], dim=1)

        # --- 步骤 3: 处理 Padding Mask ---
        
        # 1. 同步降采样序列 Mask
        # 逻辑：对于被 Conv1D 压缩的一个窗口(大小为compress_factor)，
        # 只有当窗口内**所有**元素都是 padding(True) 时，压缩后的 Token 才被视为 padding。
        # 这里利用 .view() 和 .all() 实现高效池化。
        L = seq_padding_mask.size(1)
        compressed_mask = seq_padding_mask.view(B, L // self.compress_factor, self.compress_factor).all(dim=-1)
        
        # 2. 为其他必定有效的模态生成全 False 的 Mask
        # False 代表有效，无需 ignore
        cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=compressed_mask.device) #, device=device
        text_mask = torch.zeros(B, 24, dtype=torch.bool, device=compressed_mask.device) #, device=device
        omics_mask = torch.zeros(B, 4, dtype=torch.bool, device=compressed_mask.device) #, device=device
        global_mask = torch.zeros(B, 1, dtype=torch.bool, device=compressed_mask.device) #, device=device
        
        # 3. 按步骤 2 中 tokens 相同的顺序拼接 Mask
        full_padding_mask = torch.cat([cls_mask, compressed_mask, text_mask, omics_mask, global_mask], dim=1)

        # --- 步骤 3: 深层交叉注意力融合 ---
        encoded_tokens = self.transformer(tokens, src_key_padding_mask=full_padding_mask)
        
        # --- 步骤 4: 路由输出 ---
        # 只取第 0 个位置的 [CLS] Token 作为该基因在当前细胞系下的最终表征
        cls_out = encoded_tokens[:, 0, :] # [B, d_model]
        
        return cls_out


@MODEL_REGISTRY.register()
class UnifiedTokenPredictor(nn.Module):
    """
    完整的双塔合成致死预测模型
    """
    @configurable
    def __init__(self, cfg, nhead=8, num_layers=2, compress_factor=4): #d_model=512, 
        super().__init__()
        d_model = cfg.MODEL.EMBED_DIM
        
        # 实例化 Siamese 编码器 (Gene A 和 Gene B 共享权重)
        self.gene_encoder = UnifiedGeneEncoder(
            d_model=d_model, 
            nhead=nhead, 
            num_layers=num_layers, 
            max_seq_len=cfg.DATASET.MAX_LEN,
            compress_factor=compress_factor
        )
        
        # 灵活预测头 (多视角特征拼接后为 4 * d_model)
        self.head = nn.Sequential(
            nn.Linear(4 * d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1) # 输出未归一化的连续 Score
        )

    @classmethod
    def from_config(cls, cfg):
        return {
            "cfg": cfg,
        }

    def forward(self, data): #gene_a_dict, gene_b_dict, global_ctx
        """
        gene_a_dict / gene_b_dict 包含键: 'seq', 'text', 'omics'
        global_ctx: 细胞系的全局文本特征
        """
        # 1. 独立编码两个基因
        h_a = self.gene_encoder(
            data['g1_seq_emb'], data['g1_seq_mask'], 
            data['g1_func_emb'], data['g1_func_mask'], 
            data['g1_omics'], data['cl_emb']
        )
        h_b = self.gene_encoder(
            data['g2_seq_emb'], data['g2_seq_mask'], 
            data['g2_func_emb'], data['g2_func_mask'], 
            data['g2_omics'], data['cl_emb']
        )
        
        # 2. 多视角基因对互作
        h_diff = torch.abs(h_a - h_b)
        h_mul = h_a * h_b
        # [B, 4 * d_model]
        h_pair = torch.cat([h_a, h_b, h_diff, h_mul], dim=-1)
        
        # 3. 预测得分
        score = self.head(h_pair) # [B, 1]
        
        return score.squeeze(-1)


if __name__ == "__main__":
    # --- 超参数设置 ---
    batch_size = 8
    seq_len = 1000  # 假设蛋白质序列长度为 1000
    d_model = 512
    
    # --- 模拟输入数据 ---
    # 全局细胞系背景 (两个基因共享)
    global_ctx = torch.randn(batch_size, 768)
    
    # 构造 Gene A 的字典
    gene_a = {
        'seq': torch.randn(batch_size, seq_len, 1280),
        'text': torch.randn(batch_size, 24, 1024),
        'omics': torch.randn(batch_size, 4)
    }
    
    # 构造 Gene B 的字典
    gene_b = {
        'seq': torch.randn(batch_size, seq_len, 1280),
        'text': torch.randn(batch_size, 24, 1024),
        'omics': torch.randn(batch_size, 4)
    }
    
    # --- 实例化模型并前向传播 ---
    model = UnifiedTokenPredictor(d_model=d_model, nhead=8, num_layers=4, compress_factor=4)
    
    # 将模型设为 eval 模式测试输出稳定性
    model.eval()
    with torch.no_grad():
        score = model(gene_a, gene_b, global_ctx)
        
    print(f"输出 Score 形状: {score.shape} (预期为 [{batch_size}, 1])")
    print(f"统一交互后的 Transformer 总长度: "
          f"1(CLS) + {seq_len//4}(压缩序列) + 24(文本) + 4(组学) + 1(全局) = "
          f"{1 + seq_len//4 + 24 + 4 + 1}")