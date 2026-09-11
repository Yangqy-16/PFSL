import torch
import torch.nn as nn

class FeatureCalibrator(nn.Module):
    def __init__(self, num_features: int, momentum: float = 0.1):
        """
        特征校准层。
        :param num_features: 输入特征的维度 (例如 768)
        :param momentum: 训练期统计量移动平均的动量
        """
        super().__init__()
        self.num_features = num_features
        self.momentum = momentum
        
        # 使用 register_buffer 注册不参与梯度更新的全局统计量
        # 它们会被保存在 state_dict 中，随模型一起保存和加载
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: [batch_size, ..., num_features]
        """
        # 确定需要聚合的维度（除了最后一维 feature 维度以外的所有维度）
        dims_to_reduce = tuple(range(x.ndim - 1))
        
        if self.training:
            # --- 训练阶段 ---
            # 1. 计算当前 Batch 的统计量
            # unbiased=False 保证方差计算与 BN 层逻辑一致
            batch_mean = x.mean(dim=dims_to_reduce)
            batch_var = x.var(dim=dims_to_reduce, unbiased=False)
            
            # 2. 更新全局平滑统计量 (Moving Average)
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var
                
            # 训练期不对特征做任何改变，直接原样输出
            return x
            
        else:
            # --- 测试/推理阶段 (TTA 发生在这里) ---
            # 1. 计算当前 测试 Batch 的统计量
            batch_mean = x.mean(dim=dims_to_reduce, keepdim=True)
            batch_std = torch.sqrt(x.var(dim=dims_to_reduce, unbiased=False, keepdim=True) + 1e-5)
            
            # 获取训练集的全局统计量并调整形状以支持广播
            train_mean = self.running_mean.view(*([1] * len(dims_to_reduce)), -1)
            train_std = torch.sqrt(self.running_var.view(*([1] * len(dims_to_reduce)), -1) + 1e-5)
            
            # 2. 特征校准：先去除测试集的偏差，再注入训练集的分布特征
            x_calibrated = (x - batch_mean) / batch_std
            x_calibrated = x_calibrated * train_std + train_mean
            
            return x_calibrated