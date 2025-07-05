"""
基础Transition类
定义所有transition函数的通用接口和基础功能
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class BaseTransition(nn.Module, ABC):
    """
    基础Transition类
    所有transition函数的抽象基类
    """
    
    def __init__(self, 
                 input_dim: int,
                 output_dim: int,
                 hidden_dim: int = 256,
                 model_type: str = 'mlp',
                 **kwargs):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.model_type = model_type
        
        # 存储其他参数
        self.kwargs = kwargs
        
        # 子类需要实现的核心网络
        self.core_network = None
        
    @abstractmethod
    def forward(self, x: torch.Tensor, training: bool = True) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入特征
            training: 是否为训练模式
            
        Returns:
            output: 输出特征（通常是delta）
        """
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'hidden_dim': self.hidden_dim,
            'model_type': self.model_type,
            'total_params': sum(p.numel() for p in self.parameters() if p.requires_grad),
            'kwargs': self.kwargs
        }


class MLPTransition(BaseTransition):
    """
    简单的MLP-based transition
    用作基线和快速实现
    """
    
    def __init__(self, 
                 input_dim: int,
                 output_dim: int,
                 hidden_dim: int = 256,
                 num_layers: int = 3,
                 activation: str = 'relu',
                 dropout: float = 0.1,
                 **kwargs):
        super().__init__(input_dim, output_dim, hidden_dim, 'mlp', **kwargs)
        
        self.num_layers = num_layers
        self.activation = activation
        self.dropout = dropout
        
        # 构建网络
        layers = []
        
        # 输入层
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(self._get_activation(activation))
        layers.append(nn.Dropout(dropout))
        
        # 隐藏层
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(self._get_activation(activation))
            layers.append(nn.Dropout(dropout))
        
        # 输出层
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.core_network = nn.Sequential(*layers)
    
    def _get_activation(self, activation: str) -> nn.Module:
        """获取激活函数"""
        if activation == 'relu':
            return nn.ReLU()
        elif activation == 'gelu':
            return nn.GELU()
        elif activation == 'leaky_relu':
            return nn.LeakyReLU(0.2)
        elif activation == 'swish':
            return nn.SiLU()
        else:
            return nn.ReLU()
    
    def forward(self, x: torch.Tensor, training: bool = True) -> torch.Tensor:
        """前向传播"""
        return self.core_network(x)


class ResidualBlock(nn.Module):
    """残差块"""
    
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class ResMLPTransition(BaseTransition):
    """
    基于残差连接的MLP transition
    更深的网络，更好的梯度流
    """
    
    def __init__(self, 
                 input_dim: int,
                 output_dim: int,
                 hidden_dim: int = 256,
                 num_blocks: int = 4,
                 dropout: float = 0.1,
                 **kwargs):
        super().__init__(input_dim, output_dim, hidden_dim, 'resmlp', **kwargs)
        
        self.num_blocks = num_blocks
        self.dropout = dropout
        
        # 输入投影
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # 残差块
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout) 
            for _ in range(num_blocks)
        ])
        
        # 输出投影
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        
        # Layer Norm
        self.ln = nn.LayerNorm(hidden_dim)
    
    def forward(self, x: torch.Tensor, training: bool = True) -> torch.Tensor:
        """前向传播"""
        # 输入投影
        x = self.input_proj(x)
        
        # 残差块
        for block in self.res_blocks:
            x = block(x)
        
        # Layer Norm
        x = self.ln(x)
        
        # 输出投影
        return self.output_proj(x)


def create_transition(transition_type: str, **kwargs) -> BaseTransition:
    """
    创建transition的工厂函数
    
    Args:
        transition_type: transition类型
        **kwargs: 其他参数
        
    Returns:
        transition实例
    """
    if transition_type == 'mlp':
        return MLPTransition(**kwargs)
    elif transition_type == 'resmlp':
        return ResMLPTransition(**kwargs)
    else:
        raise ValueError(f"Unknown transition type: {transition_type}")


# 示例使用
if __name__ == "__main__":
    # 测试MLP transition
    mlp_transition = MLPTransition(
        input_dim=192,  # 64 + 128
        output_dim=64,
        hidden_dim=256,
        num_layers=3
    )
    
    # 测试输入
    x = torch.randn(4, 192)
    output = mlp_transition(x)
    
    print(f"MLP Transition:")
    print(f"  输入形状: {x.shape}")
    print(f"  输出形状: {output.shape}")
    print(f"  参数数量: {sum(p.numel() for p in mlp_transition.parameters())}")
    
    # 测试ResMLP transition
    resmlp_transition = ResMLPTransition(
        input_dim=192,
        output_dim=64,
        hidden_dim=256,
        num_blocks=4
    )
    
    output2 = resmlp_transition(x)
    print(f"\nResMLP Transition:")
    print(f"  输入形状: {x.shape}")
    print(f"  输出形状: {output2.shape}")
    print(f"  参数数量: {sum(p.numel() for p in resmlp_transition.parameters())}") 