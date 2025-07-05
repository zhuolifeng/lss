"""
基于ControlNet的Transition2
处理多个条件（Xt-1, Yt-1, At-1）并预测delta_Y

支持的架构：
1. ControlNet: 专用于多条件控制
2. CrossAttention: 基于交叉注意力的多模态融合
3. FiLM: Feature-wise Linear Modulation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple

from .base_transition import BaseTransition


class ControlNetTransition2(BaseTransition):
    """
    基于ControlNet的Transition2
    
    处理多个条件输入：
    1. Xt-1: 上一时刻的lift特征
    2. Yt-1: 上一时刻的splat特征
    3. At-1: 编码后的动作信息
    
    预测delta_Y用于更新splat特征
    """
    
    def __init__(self,
                 lift_dim: int,
                 splat_dim: int,
                 action_dim: int,
                 output_dim: int,
                 hidden_dim: int = 256,
                 model_type: str = 'controlnet',
                 num_layers: int = 4,
                 num_heads: int = 8,
                 cross_attn_dropout: float = 0.1,
                 **kwargs):
        # 计算总输入维度
        total_input_dim = lift_dim + splat_dim + action_dim
        super().__init__(total_input_dim, output_dim, hidden_dim, model_type, **kwargs)
        
        self.lift_dim = lift_dim
        self.splat_dim = splat_dim
        self.action_dim = action_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.cross_attn_dropout = cross_attn_dropout
        
        # 创建核心网络
        self.core_network = self._create_controlnet_model()
        
    def _create_controlnet_model(self) -> nn.Module:
        """创建ControlNet模型"""
        if self.model_type == 'controlnet':
            return ControlNetCore(
                lift_dim=self.lift_dim,
                splat_dim=self.splat_dim,
                action_dim=self.action_dim,
                output_dim=self.output_dim,
                hidden_dim=self.hidden_dim,
                num_layers=self.num_layers
            )
        elif self.model_type == 'cross_attention':
            return CrossAttentionCore(
                lift_dim=self.lift_dim,
                splat_dim=self.splat_dim,
                action_dim=self.action_dim,
                output_dim=self.output_dim,
                hidden_dim=self.hidden_dim,
                num_heads=self.num_heads,
                num_layers=self.num_layers,
                dropout=self.cross_attn_dropout
            )
        elif self.model_type == 'film':
            return FiLMCore(
                lift_dim=self.lift_dim,
                splat_dim=self.splat_dim,
                action_dim=self.action_dim,
                output_dim=self.output_dim,
                hidden_dim=self.hidden_dim,
                num_layers=self.num_layers
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def forward(self, 
                lift_features: torch.Tensor,
                splat_features: torch.Tensor,
                action_features: torch.Tensor,
                training: bool = True) -> torch.Tensor:
        """
        前向传播
        
        Args:
            lift_features: Xt-1 [B, lift_dim]
            splat_features: Yt-1 [B, splat_dim]
            action_features: encoded At-1 [B, action_dim]
            training: 是否为训练模式
            
        Returns:
            delta_Y: 预测的delta_Y [B, output_dim]
        """
        return self.core_network(lift_features, splat_features, action_features)


class ControlNetCore(nn.Module):
    """ControlNet核心模块"""
    
    def __init__(self, 
                 lift_dim: int,
                 splat_dim: int,
                 action_dim: int,
                 output_dim: int,
                 hidden_dim: int,
                 num_layers: int):
        super().__init__()
        
        # 条件编码器
        self.lift_encoder = nn.Sequential(
            nn.Linear(lift_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.splat_encoder = nn.Sequential(
            nn.Linear(splat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 控制模块
        self.control_layers = nn.ModuleList([
            ControlLayer(hidden_dim, hidden_dim, num_conditions=3)
            for _ in range(num_layers)
        ])
        
        # 输出层
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
    def forward(self, 
                lift_features: torch.Tensor,
                splat_features: torch.Tensor,
                action_features: torch.Tensor) -> torch.Tensor:
        
        # 编码条件
        lift_emb = self.lift_encoder(lift_features)
        splat_emb = self.splat_encoder(splat_features)
        action_emb = self.action_encoder(action_features)
        
        # 初始特征
        x = splat_emb
        
        # 逐层应用控制
        conditions = [lift_emb, splat_emb, action_emb]
        for control_layer in self.control_layers:
            x = control_layer(x, conditions)
        
        # 输出delta_Y
        return self.output_proj(x)


class ControlLayer(nn.Module):
    """单个控制层"""
    
    def __init__(self, input_dim: int, hidden_dim: int, num_conditions: int):
        super().__init__()
        
        # 主干网络
        self.main_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 控制网络
        self.condition_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim * 2)  # scale和shift
            )
            for _ in range(num_conditions)
        ])
        
        # 权重网络
        self.weight_net = nn.Sequential(
            nn.Linear(input_dim * num_conditions, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_conditions),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, x: torch.Tensor, conditions: list) -> torch.Tensor:
        # 主干特征
        main_features = self.main_net(x)
        
        # 计算条件权重
        condition_concat = torch.cat(conditions, dim=-1)
        condition_weights = self.weight_net(condition_concat)
        
        # 应用条件调制
        modulated_features = main_features
        for i, (condition, weight) in enumerate(zip(conditions, condition_weights.chunk(len(conditions), dim=-1))):
            # 生成调制参数
            modulation = self.condition_nets[i](condition)
            scale, shift = modulation.chunk(2, dim=-1)
            
            # FiLM调制
            conditioned_features = scale * main_features + shift
            
            # 加权融合
            modulated_features = modulated_features + weight * conditioned_features
        
        # 残差连接
        return x + modulated_features


class CrossAttentionCore(nn.Module):
    """基于交叉注意力的多模态融合"""
    
    def __init__(self,
                 lift_dim: int,
                 splat_dim: int,
                 action_dim: int,
                 output_dim: int,
                 hidden_dim: int,
                 num_heads: int,
                 num_layers: int,
                 dropout: float):
        super().__init__()
        
        # 输入投影
        self.lift_proj = nn.Linear(lift_dim, hidden_dim)
        self.splat_proj = nn.Linear(splat_dim, hidden_dim)
        self.action_proj = nn.Linear(action_dim, hidden_dim)
        
        # 位置编码
        self.pos_encoding = nn.Parameter(torch.randn(3, hidden_dim))
        
        # 交叉注意力层
        self.cross_attn_layers = nn.ModuleList([
            CrossAttentionLayer(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # 输出层
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
    def forward(self,
                lift_features: torch.Tensor,
                splat_features: torch.Tensor,
                action_features: torch.Tensor) -> torch.Tensor:
        
        # 投影到相同维度
        lift_emb = self.lift_proj(lift_features)
        splat_emb = self.splat_proj(splat_features)
        action_emb = self.action_proj(action_features)
        
        # 堆叠为序列 [B, 3, hidden_dim]
        x = torch.stack([lift_emb, splat_emb, action_emb], dim=1)
        
        # 添加位置编码
        x = x + self.pos_encoding.unsqueeze(0)
        
        # 交叉注意力层
        for layer in self.cross_attn_layers:
            x = layer(x)
        
        # 提取splat位置的特征
        splat_output = x[:, 1, :]
        
        # 输出delta_Y
        return self.output_proj(splat_output)


class CrossAttentionLayer(nn.Module):
    """交叉注意力层"""
    
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float):
        super().__init__()
        
        self.self_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention
        attn_output, _ = self.self_attn(x, x, x)
        x = self.norm1(x + attn_output)
        
        # Feed forward
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)
        
        return x


class FiLMCore(nn.Module):
    """基于FiLM的条件控制"""
    
    def __init__(self,
                 lift_dim: int,
                 splat_dim: int,
                 action_dim: int,
                 output_dim: int,
                 hidden_dim: int,
                 num_layers: int):
        super().__init__()
        
        # 主干网络
        self.backbone = nn.ModuleList([
            nn.Sequential(
                nn.Linear(splat_dim if i == 0 else hidden_dim, hidden_dim),
                nn.ReLU()
            )
            for i in range(num_layers)
        ])
        
        # FiLM生成器
        condition_dim = lift_dim + action_dim
        self.film_generators = nn.ModuleList([
            nn.Sequential(
                nn.Linear(condition_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim * 2)  # gamma和beta
            )
            for _ in range(num_layers)
        ])
        
        # 输出层
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        
    def forward(self,
                lift_features: torch.Tensor,
                splat_features: torch.Tensor,
                action_features: torch.Tensor) -> torch.Tensor:
        
        # 条件特征
        condition = torch.cat([lift_features, action_features], dim=-1)
        
        # 初始特征
        x = splat_features
        
        # 逐层应用FiLM调制
        for i, (backbone_layer, film_generator) in enumerate(zip(self.backbone, self.film_generators)):
            # 主干特征
            x = backbone_layer(x)
            
            # 生成FiLM参数
            film_params = film_generator(condition)
            gamma, beta = film_params.chunk(2, dim=-1)
            
            # FiLM调制
            x = gamma * x + beta
        
        # 输出delta_Y
        return self.output_proj(x) 