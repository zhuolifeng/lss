"""
动作编码器模块
用于将车辆动作信息（位置、旋转、速度等）编码为特征向量，供HMM插件使用

定义的类：
- ActionEncoder: 主要的动作编码器，支持MLP、RNN、Transformer三种编码方式
- PositionalEncoding: 位置编码器，用于序列数据的位置信息编码
- EgoMotionExtractor: 静态方法类，用于从LSS输入中提取自车运动信息

主要功能：
- 将6维动作向量编码为高维特征表示
- 支持序列和单帧两种输入模式
- 提供多种编码策略以适应不同的模型需求
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict, Any


class ActionEncoder(nn.Module):
    """
    动作编码器
    
    将多维的动作信息编码为固定维度的特征向量
    支持不同类型的编码策略
    """
    
    def __init__(self,
                 action_dim: int = 6,
                 hidden_dim: int = 128,
                 encoding_type: str = 'mlp',
                 use_positional_encoding: bool = True,
                 dropout: float = 0.1):
        super().__init__()
        
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.encoding_type = encoding_type
        self.use_positional_encoding = use_positional_encoding
        self.dropout = dropout
        
        # 创建编码器
        if encoding_type == 'mlp':
            self.encoder = self._create_mlp_encoder()
        elif encoding_type == 'rnn':
            self.encoder = self._create_rnn_encoder()
        elif encoding_type == 'transformer':
            self.encoder = self._create_transformer_encoder()
        else:
            raise ValueError(f"Unknown encoding type: {encoding_type}")
        
        # 位置编码（如果启用）
        if use_positional_encoding:
            self.pos_encoder = PositionalEncoding(hidden_dim, dropout)
        
        # 输出投影
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        
    def _create_mlp_encoder(self) -> nn.Module:
        """创建MLP编码器"""
        return nn.Sequential(
            nn.Linear(self.action_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
    
    def _create_rnn_encoder(self) -> nn.Module:
        """创建RNN编码器"""
        return nn.ModuleDict({
            'input_proj': nn.Linear(self.action_dim, self.hidden_dim),
            'rnn': nn.GRU(self.hidden_dim, self.hidden_dim, batch_first=True),
            'output_proj': nn.Linear(self.hidden_dim, self.hidden_dim)
        })
    
    def _create_transformer_encoder(self) -> nn.Module:
        """创建Transformer编码器"""
        return nn.ModuleDict({
            'input_proj': nn.Linear(self.action_dim, self.hidden_dim),
            'transformer': nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=self.hidden_dim,
                    nhead=8,
                    dim_feedforward=self.hidden_dim * 4,
                    dropout=self.dropout,
                    batch_first=True
                ),
                num_layers=2
            ),
            'output_proj': nn.Linear(self.hidden_dim, self.hidden_dim)
        })
    
    def forward(self, actions: torch.Tensor, sequence_length: Optional[int] = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            actions: 动作信息 [B, action_dim] 或 [B, seq_len, action_dim]
            sequence_length: 序列长度（用于RNN/Transformer）
            
        Returns:
            encoded_actions: 编码后的动作特征 [B, hidden_dim]
        """
        if self.encoding_type == 'mlp':
            return self._mlp_forward(actions)
        elif self.encoding_type == 'rnn':
            return self._rnn_forward(actions, sequence_length)
        elif self.encoding_type == 'transformer':
            return self._transformer_forward(actions, sequence_length)
    
    def _mlp_forward(self, actions: torch.Tensor) -> torch.Tensor:
        """MLP前向传播"""
        # 确保输入是2D
        if actions.dim() == 3:
            actions = actions.mean(dim=1)  # 平均池化序列维度
        
        encoded = self.encoder(actions)
        
        if self.use_positional_encoding:
            encoded = self.pos_encoder(encoded.unsqueeze(1)).squeeze(1)
        
        return self.output_proj(encoded)
    
    def _rnn_forward(self, actions: torch.Tensor, sequence_length: Optional[int] = None) -> torch.Tensor:
        """RNN前向传播"""
        # 确保输入是3D
        if actions.dim() == 2:
            actions = actions.unsqueeze(1)  # 添加序列维度
        
        # 输入投影
        x = self.encoder['input_proj'](actions)
        
        # 位置编码
        if self.use_positional_encoding:
            x = self.pos_encoder(x)
        
        # RNN编码
        output, _ = self.encoder['rnn'](x)
        
        # 取最后一个时间步
        encoded = output[:, -1, :]
        
        return self.encoder['output_proj'](encoded)
    
    def _transformer_forward(self, actions: torch.Tensor, sequence_length: Optional[int] = None) -> torch.Tensor:
        """Transformer前向传播"""
        # 确保输入是3D
        if actions.dim() == 2:
            actions = actions.unsqueeze(1)
        
        # 输入投影
        x = self.encoder['input_proj'](actions)
        
        # 位置编码
        if self.use_positional_encoding:
            x = self.pos_encoder(x)
        
        # Transformer编码
        encoded = self.encoder['transformer'](x)
        
        # 取最后一个时间步
        encoded = encoded[:, -1, :]
        
        return self.encoder['output_proj'](encoded)


class PositionalEncoding(nn.Module):
    """位置编码"""
    
    def __init__(self, hidden_dim: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, hidden_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2).float() * (-math.log(10000.0) / hidden_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return self.dropout(x)


class EgoMotionExtractor:
    """
    自车运动信息提取器
    从LSS模型的输入中提取动作信息
    """
    
    @staticmethod
    def extract_from_lss_inputs(rots: torch.Tensor, 
                               trans: torch.Tensor,
                               timestamps: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        从LSS输入中提取动作信息
        
        Args:
            rots: 旋转矩阵 [B, N, 3, 3]
            trans: 平移向量 [B, N, 3]
            timestamps: 时间戳 [B, N] (可选)
            
        Returns:
            actions: 动作特征 [B, 6]
        """
        batch_size = rots.shape[0]
        device = rots.device
        
        # 计算平移量的幅度和方向
        translation_magnitude = torch.norm(trans, dim=-1)  # [B, N]
        avg_translation = translation_magnitude.mean(dim=-1)  # [B]
        
        # 计算旋转角度
        # 使用旋转矩阵的迹来计算旋转角度
        trace = torch.diagonal(rots, dim1=-2, dim2=-1).sum(dim=-1)  # [B, N]
        rotation_angle = torch.acos(torch.clamp((trace - 1) / 2, -1, 1))  # [B, N]
        avg_rotation = rotation_angle.mean(dim=-1)  # [B]
        
        # 构造6维动作向量
        actions = torch.zeros(batch_size, 6, device=device)
        
        # 线速度（简化表示）
        actions[:, 0] = avg_translation  # 主要运动方向
        actions[:, 1] = torch.std(translation_magnitude, dim=-1)  # 运动变化
        
        # 角速度（简化表示）
        actions[:, 3] = avg_rotation  # 主要旋转
        actions[:, 4] = torch.std(rotation_angle, dim=-1)  # 旋转变化
        
        # 如果有时间戳，计算时间相关特征
        if timestamps is not None:
            dt = timestamps[:, 1:] - timestamps[:, :-1]  # [B, N-1]
            actions[:, 2] = dt.mean(dim=-1) if dt.shape[1] > 0 else 0  # 平均时间间隔
            actions[:, 5] = dt.std(dim=-1) if dt.shape[1] > 0 else 0   # 时间间隔变化
        
        return actions
    
    @staticmethod
    def extract_velocity_acceleration(positions: torch.Tensor, 
                                    timestamps: torch.Tensor) -> torch.Tensor:
        """
        从位置和时间戳计算速度和加速度
        
        Args:
            positions: 位置序列 [B, seq_len, 3]
            timestamps: 时间戳 [B, seq_len]
            
        Returns:
            motion_features: 运动特征 [B, 9] (速度3维 + 加速度3维 + 角速度3维)
        """
        batch_size, seq_len, _ = positions.shape
        device = positions.device
        
        if seq_len < 2:
            return torch.zeros(batch_size, 9, device=device)
        
        # 计算时间差
        dt = timestamps[:, 1:] - timestamps[:, :-1]  # [B, seq_len-1]
        dt = dt.unsqueeze(-1)  # [B, seq_len-1, 1]
        
        # 计算速度
        velocity = (positions[:, 1:] - positions[:, :-1]) / (dt + 1e-8)  # [B, seq_len-1, 3]
        avg_velocity = velocity.mean(dim=1)  # [B, 3]
        
        # 计算加速度
        if seq_len > 2:
            acceleration = (velocity[:, 1:] - velocity[:, :-1]) / (dt[:, 1:] + 1e-8)  # [B, seq_len-2, 3]
            avg_acceleration = acceleration.mean(dim=1)  # [B, 3]
        else:
            avg_acceleration = torch.zeros(batch_size, 3, device=device)
        
        # 计算角速度（简化版本）
        if seq_len > 1:
            direction_vectors = F.normalize(velocity, dim=-1)  # [B, seq_len-1, 3]
            if seq_len > 2:
                angular_velocity = torch.cross(direction_vectors[:, :-1], direction_vectors[:, 1:], dim=-1)
                avg_angular_velocity = angular_velocity.mean(dim=1)  # [B, 3]
            else:
                avg_angular_velocity = torch.zeros(batch_size, 3, device=device)
        else:
            avg_angular_velocity = torch.zeros(batch_size, 3, device=device)
        
        # 组合所有特征
        motion_features = torch.cat([avg_velocity, avg_acceleration, avg_angular_velocity], dim=-1)
        
        return motion_features 