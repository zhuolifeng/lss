"""
Mamba-based Diffusion Model
基于状态空间模型(SSM)的diffusion模型

论文参考: "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
相比Transformer，Mamba具有线性复杂度和长序列建模能力
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict
from einops import rearrange, repeat


class MambaDiffusion(nn.Module):
    """
    基于Mamba的diffusion模型
    
    Mamba是一种高效的状态空间模型，特别适合处理长序列
    在diffusion中可以更好地建模时序依赖关系
    """
    
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 512,
                 num_layers: int = 8,
                 condition_dim: int = 128,
                 state_dim: int = 64,
                 expand_factor: int = 2,
                 dt_rank: int = 8,
                 dropout: float = 0.1,
                 use_conv: bool = True,
                 **kwargs):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.condition_dim = condition_dim
        self.state_dim = state_dim
        self.expand_factor = expand_factor
        self.dt_rank = dt_rank
        self.dropout = dropout
        self.use_conv = use_conv
        
        # 输入嵌入
        self.input_embedding = nn.Linear(input_dim, hidden_dim)
        
        # 时间嵌入
        self.time_embedding = TimestepEmbedding(hidden_dim)
        
        # 条件嵌入
        self.condition_embedding = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Mamba块
        self.mamba_blocks = nn.ModuleList([
            MambaBlock(
                hidden_dim=hidden_dim,
                state_dim=state_dim,
                expand_factor=expand_factor,
                dt_rank=dt_rank,
                dropout=dropout,
                use_conv=use_conv,
                layer_idx=i
            )
            for i in range(num_layers)
        ])
        
        # 输出层
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, input_dim)
        
        # 初始化
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """初始化权重"""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
    
    def forward(self, 
                x: torch.Tensor,
                timestep: torch.Tensor,
                condition: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入特征 [B, input_dim]
            timestep: 时间步 [B]
            condition: 条件信息 [B, condition_dim]
            
        Returns:
            predicted_noise: 预测的噪声 [B, input_dim]
        """
        B, D = x.shape
        
        # 添加序列维度 (Mamba需要序列输入)
        x = x.unsqueeze(1)  # [B, 1, D]
        
        # 输入嵌入
        x = self.input_embedding(x)  # [B, 1, hidden_dim]
        
        # 时间嵌入
        time_emb = self.time_embedding(timestep)  # [B, hidden_dim]
        
        # 条件嵌入
        if condition is not None:
            cond_emb = self.condition_embedding(condition)  # [B, hidden_dim]
        else:
            cond_emb = torch.zeros_like(time_emb)
        
        # 结合时间和条件嵌入
        combined_emb = time_emb + cond_emb  # [B, hidden_dim]
        
        # 通过Mamba块
        for block in self.mamba_blocks:
            x = block(x, combined_emb)
        
        # 输出
        x = self.output_norm(x)  # [B, 1, hidden_dim]
        x = self.output_proj(x)  # [B, 1, input_dim]
        
        return x.squeeze(1)  # [B, input_dim]
    
    def get_model_size(self) -> int:
        """获取模型大小"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MambaBlock(nn.Module):
    """
    Mamba块
    实现选择性状态空间模型
    """
    
    def __init__(self,
                 hidden_dim: int,
                 state_dim: int = 64,
                 expand_factor: int = 2,
                 dt_rank: int = 8,
                 dropout: float = 0.1,
                 use_conv: bool = True,
                 layer_idx: int = 0):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim
        self.expand_factor = expand_factor
        self.dt_rank = dt_rank
        self.intermediate_dim = hidden_dim * expand_factor
        self.layer_idx = layer_idx
        
        # 输入投影
        self.input_projection = nn.Linear(hidden_dim, self.intermediate_dim * 2)
        
        # 1D卷积 (用于局部信息)
        if use_conv:
            self.conv1d = nn.Conv1d(
                in_channels=self.intermediate_dim,
                out_channels=self.intermediate_dim,
                kernel_size=3,
                padding=1,
                groups=self.intermediate_dim
            )
        else:
            self.conv1d = None
        
        # 选择性扫描参数
        self.dt_proj = nn.Linear(self.intermediate_dim, dt_rank)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, state_dim + 1, dtype=torch.float32)))
        self.D = nn.Parameter(torch.ones(self.intermediate_dim))
        
        # 选择性投影
        self.B_proj = nn.Linear(self.intermediate_dim, state_dim)
        self.C_proj = nn.Linear(self.intermediate_dim, state_dim)
        
        # dt投影
        self.dt_bias = nn.Parameter(torch.rand(self.intermediate_dim))
        
        # 输出投影
        self.output_projection = nn.Linear(self.intermediate_dim, hidden_dim)
        
        # 归一化
        self.norm = nn.LayerNorm(hidden_dim)
        
        # 条件调制
        self.condition_projection = nn.Linear(hidden_dim, self.intermediate_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入特征 [B, seq_len, hidden_dim]
            condition: 条件信息 [B, hidden_dim]
            
        Returns:
            output: 输出特征 [B, seq_len, hidden_dim]
        """
        B, L, D = x.shape
        
        # 保存残差
        residual = x
        
        # 层归一化
        x = self.norm(x)
        
        # 输入投影
        x_proj = self.input_projection(x)  # [B, L, intermediate_dim * 2]
        x_proj, gate = x_proj.chunk(2, dim=-1)  # [B, L, intermediate_dim]
        
        # 1D卷积
        if self.conv1d is not None:
            x_proj = rearrange(x_proj, 'b l d -> b d l')
            x_proj = self.conv1d(x_proj)
            x_proj = rearrange(x_proj, 'b d l -> b l d')
        
        # 激活函数
        x_proj = F.silu(x_proj)
        
        # 条件调制
        cond_proj = self.condition_projection(condition)  # [B, intermediate_dim]
        cond_proj = cond_proj.unsqueeze(1)  # [B, 1, intermediate_dim]
        x_proj = x_proj + cond_proj
        
        # 选择性扫描
        y = self.selective_scan(x_proj)  # [B, L, intermediate_dim]
        
        # 门控
        y = y * F.silu(gate)
        
        # 输出投影
        y = self.output_projection(y)  # [B, L, hidden_dim]
        
        # Dropout
        y = self.dropout(y)
        
        # 残差连接
        return residual + y
    
    def selective_scan(self, x: torch.Tensor) -> torch.Tensor:
        """
        选择性扫描算法
        
        Args:
            x: 输入特征 [B, L, intermediate_dim]
            
        Returns:
            y: 扫描结果 [B, L, intermediate_dim]
        """
        B, L, D = x.shape
        
        # 计算dt, B, C
        dt = self.dt_proj(x)  # [B, L, dt_rank]
        B_proj = self.B_proj(x)  # [B, L, state_dim]
        C_proj = self.C_proj(x)  # [B, L, state_dim]
        
        # 计算A
        A = -torch.exp(self.A_log)  # [state_dim]
        A = A.unsqueeze(0).unsqueeze(0)  # [1, 1, state_dim]
        
        # 计算D
        D = self.D.unsqueeze(0).unsqueeze(0)  # [1, 1, intermediate_dim]
        
        # 离散化
        dt = F.softplus(dt + self.dt_bias.unsqueeze(0).unsqueeze(0))  # [B, L, dt_rank]
        
        # 扩展dt到状态维度
        dt = dt.mean(dim=-1, keepdim=True)  # [B, L, 1]
        dt = dt.expand(-1, -1, self.state_dim)  # [B, L, state_dim]
        
        # 离散化A和B
        A_discrete = torch.exp(A * dt)  # [B, L, state_dim]
        B_discrete = B_proj * dt  # [B, L, state_dim]
        
        # 初始化状态
        h = torch.zeros(B, self.state_dim, device=x.device, dtype=x.dtype)
        
        # 扫描
        ys = []
        for i in range(L):
            # 更新状态
            h = A_discrete[:, i] * h + B_discrete[:, i] * x[:, i].mean(dim=-1, keepdim=True)
            
            # 计算输出
            y = (C_proj[:, i] * h).sum(dim=-1)  # [B]
            ys.append(y)
        
        # 堆叠输出
        y = torch.stack(ys, dim=1)  # [B, L]
        
        # 扩展到中间维度
        y = y.unsqueeze(-1).expand(-1, -1, D)  # [B, L, intermediate_dim]
        
        # 加上直接连接
        y = y + x * D
        
        return y


class TimestepEmbedding(nn.Module):
    """时间步嵌入"""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # 频率嵌入
        half_dim = hidden_dim // 2
        self.register_buffer('freqs', torch.exp(
            -math.log(10000) * torch.arange(half_dim, dtype=torch.float32) / half_dim
        ))
        
        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
    
    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        # 正弦嵌入
        args = timestep.float().unsqueeze(-1) * self.freqs
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        
        # MLP
        return self.mlp(embedding)


class MambaSequenceModel(nn.Module):
    """
    Mamba序列模型
    处理更长的序列输入
    """
    
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 512,
                 num_layers: int = 8,
                 condition_dim: int = 128,
                 sequence_length: int = 32,
                 **kwargs):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.condition_dim = condition_dim
        self.sequence_length = sequence_length
        
        # 将输入拆分为多个patch
        self.patch_size = input_dim // sequence_length
        if input_dim % sequence_length != 0:
            self.patch_size += 1
            self.padded_input_dim = self.patch_size * sequence_length
        else:
            self.padded_input_dim = input_dim
        
        # 输入嵌入
        self.input_embedding = nn.Linear(self.patch_size, hidden_dim)
        
        # 时间嵌入
        self.time_embedding = TimestepEmbedding(hidden_dim)
        
        # 条件嵌入
        self.condition_embedding = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 位置编码
        self.pos_embedding = nn.Parameter(torch.randn(1, sequence_length, hidden_dim))
        
        # Mamba块
        self.mamba_blocks = nn.ModuleList([
            MambaBlock(
                hidden_dim=hidden_dim,
                layer_idx=i,
                **kwargs
            )
            for i in range(num_layers)
        ])
        
        # 输出层
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, self.patch_size)
        
    def forward(self, 
                x: torch.Tensor,
                timestep: torch.Tensor,
                condition: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入特征 [B, input_dim]
            timestep: 时间步 [B]
            condition: 条件信息 [B, condition_dim]
            
        Returns:
            predicted_noise: 预测的噪声 [B, input_dim]
        """
        B, D = x.shape
        
        # 填充输入
        if D < self.padded_input_dim:
            x = F.pad(x, (0, self.padded_input_dim - D))
        
        # 重塑为序列
        x = x.view(B, self.sequence_length, self.patch_size)
        
        # 输入嵌入
        x = self.input_embedding(x)  # [B, seq_len, hidden_dim]
        
        # 位置编码
        x = x + self.pos_embedding
        
        # 时间嵌入
        time_emb = self.time_embedding(timestep)  # [B, hidden_dim]
        
        # 条件嵌入
        if condition is not None:
            cond_emb = self.condition_embedding(condition)  # [B, hidden_dim]
        else:
            cond_emb = torch.zeros_like(time_emb)
        
        # 结合时间和条件嵌入
        combined_emb = time_emb + cond_emb  # [B, hidden_dim]
        
        # 通过Mamba块
        for block in self.mamba_blocks:
            x = block(x, combined_emb)
        
        # 输出
        x = self.output_norm(x)  # [B, seq_len, hidden_dim]
        x = self.output_proj(x)  # [B, seq_len, patch_size]
        
        # 重塑回原来的形状
        x = x.view(B, self.padded_input_dim)
        
        # 裁剪到原始维度
        if D < self.padded_input_dim:
            x = x[:, :D]
        
        return x
