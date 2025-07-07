"""
UNet-based Diffusion Model
用于transition1的经典UNet架构diffusion模型

支持CFG（Classifier-Free Guidance）和条件生成
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class UNetDiffusion(nn.Module):
    """
    基于UNet的diffusion模型
    
    用于预测delta_X，支持CFG（Classifier-Free Guidance）
    """
    
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 256,
                 num_layers: int = 4,
                 num_heads: int = 8,
                 condition_dim: int = 128,
                 dropout: float = 0.1,
                 time_embed_dim: int = 128,
                 use_attention: bool = True):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.condition_dim = condition_dim
        self.dropout = dropout
        self.time_embed_dim = time_embed_dim
        self.use_attention = use_attention
        
        # 时间嵌入
        self.time_embed = TimeEmbedding(time_embed_dim, hidden_dim)
        
        # 输入投影
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # 条件投影
        self.condition_proj = nn.Linear(condition_dim, hidden_dim)
        
        # UNet编码器
        self.encoder_blocks = nn.ModuleList([
            UNetBlock(
                hidden_dim * (2 ** i),
                hidden_dim * (2 ** (i + 1)),
                time_embed_dim=hidden_dim,
                dropout=dropout,
                use_attention=use_attention and i >= 1
            )
            for i in range(num_layers // 2)
        ])
        
        # 中间层
        self.middle_block = UNetBlock(
            hidden_dim * (2 ** (num_layers // 2)),
            hidden_dim * (2 ** (num_layers // 2)),
            time_embed_dim=hidden_dim,
            dropout=dropout,
            use_attention=True
        )
        
        # UNet解码器
        self.decoder_blocks = nn.ModuleList([
            UNetBlock(
                hidden_dim * (2 ** (num_layers // 2 - i)) + hidden_dim * (2 ** (num_layers // 2 - i - 1)),
                hidden_dim * (2 ** (num_layers // 2 - i - 1)),
                time_embed_dim=hidden_dim,
                dropout=dropout,
                use_attention=use_attention and (num_layers // 2 - i - 1) >= 1
            )
            for i in range(num_layers // 2)
        ])
        
        # 输出投影
        self.output_proj = nn.Sequential(
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
        # CFG参数
        self.cfg_scale = 1.0
        
    def forward(self, 
                x: torch.Tensor,
                timestep: torch.Tensor,
                condition: Optional[torch.Tensor] = None,
                cfg_scale: float = 1.0) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入特征 [B, input_dim]
            timestep: 时间步 [B]
            condition: 条件信息 [B, condition_dim]
            cfg_scale: CFG强度
            
        Returns:
            predicted_noise: 预测的噪声 [B, input_dim]
        """
        self.cfg_scale = cfg_scale
        
        # 时间嵌入
        time_emb = self.time_embed(timestep)
        
        # 输入投影
        h = self.input_proj(x)
        
        # 条件融合
        if condition is not None:
            condition_emb = self.condition_proj(condition)
            h = h + condition_emb
        
        # 编码器
        skip_connections = []
        for encoder_block in self.encoder_blocks:
            h = encoder_block(h, time_emb)
            skip_connections.append(h)
        
        # 中间层
        h = self.middle_block(h, time_emb)
        
        # 解码器
        for decoder_block, skip in zip(self.decoder_blocks, reversed(skip_connections)):
            h = torch.cat([h, skip], dim=-1)
            h = decoder_block(h, time_emb)
        
        # 输出
        predicted_noise = self.output_proj(h)
        
        # CFG处理
        if cfg_scale > 1.0 and condition is not None:
            predicted_noise = self._apply_cfg(x, timestep, condition, predicted_noise, cfg_scale)
        
        return predicted_noise
    
    def _apply_cfg(self, 
                   x: torch.Tensor,
                   timestep: torch.Tensor,
                   condition: torch.Tensor,
                   conditional_pred: torch.Tensor,
                   cfg_scale: float) -> torch.Tensor:
        """应用CFG"""
        # 无条件预测
        unconditional_pred = self.forward(x, timestep, condition=None, cfg_scale=1.0)
        
        # CFG融合
        return unconditional_pred + cfg_scale * (conditional_pred - unconditional_pred)
    
    def sample(self,
               shape: Tuple[int, ...],
               condition: Optional[torch.Tensor] = None,
               num_timesteps: int = 100,
               cfg_scale: float = 1.0,
               device: torch.device = torch.device('cpu')) -> torch.Tensor:
        """
        采样生成
        
        Args:
            shape: 输出形状
            condition: 条件信息
            num_timesteps: 采样步数
            cfg_scale: CFG强度
            device: 设备
            
        Returns:
            generated_samples: 生成的样本
        """
        # 初始化噪声
        x = torch.randn(shape, device=device)
        
        # 逐步去噪
        for t in reversed(range(num_timesteps)):
            timestep = torch.full((shape[0],), t, device=device, dtype=torch.long)
            
            # 预测噪声
            predicted_noise = self.forward(x, timestep, condition, cfg_scale)
            
            # 更新x
            alpha = 1.0 - t / num_timesteps
            sigma = t / num_timesteps
            x = (x - sigma * predicted_noise) / alpha
        
        return x


class UNetBlock(nn.Module):
    """UNet基础块"""
    
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 time_embed_dim: int,
                 dropout: float = 0.1,
                 use_attention: bool = False):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_attention = use_attention
        
        # 时间投影
        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embed_dim, out_channels)
        )
        
        # 第一个卷积层（这里用Linear替代）
        self.conv1 = nn.Sequential(
            nn.GroupNorm(8, in_channels),
            nn.SiLU(),
            nn.Linear(in_channels, out_channels),
            nn.Dropout(dropout)
        )
        
        # 第二个卷积层
        self.conv2 = nn.Sequential(
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Linear(out_channels, out_channels),
            nn.Dropout(dropout)
        )
        
        # 残差连接
        self.residual_proj = nn.Linear(in_channels, out_channels) if in_channels != out_channels else nn.Identity()
        
        # 注意力机制
        if use_attention:
            self.attention = SelfAttention(out_channels, num_heads=8, dropout=dropout)
        
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        # 残差连接
        residual = self.residual_proj(x)
        
        # 第一个卷积
        h = self.conv1(x)
        
        # 时间嵌入
        h = h + self.time_proj(time_emb)
        
        # 第二个卷积
        h = self.conv2(h)
        
        # 注意力
        if self.use_attention:
            h = self.attention(h)
        
        # 残差连接
        return h + residual


class TimeEmbedding(nn.Module):
    """时间嵌入层"""
    
    def __init__(self, embed_dim: int, hidden_dim: int):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        
        # 正弦位置编码
        self.register_buffer('freq_embed', self._get_freq_embed(embed_dim))
        
        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def _get_freq_embed(self, embed_dim: int) -> torch.Tensor:
        """获取频率嵌入"""
        half_dim = embed_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
        return emb
    
    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        """
        Args:
            timestep: [B] 时间步
        Returns:
            time_embed: [B, hidden_dim]
        """
        # 确保timestep是1D张量
        if timestep.dim() > 1:
            timestep = timestep.flatten()
        
        # 正弦嵌入
        emb = timestep.float()[:, None] * self.freq_embed[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        
        # MLP
        return self.mlp(emb)


class SelfAttention(nn.Module):
    """自注意力机制"""
    
    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        # 注意力投影
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        
        # 输出投影
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # 层归一化
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, embed_dim]
        Returns:
            output: [B, embed_dim]
        """
        B, D = x.shape
        
        # 添加序列维度用于注意力计算
        x = x.unsqueeze(1)  # [B, 1, D]
        
        # 计算QKV
        q = self.q_proj(x)  # [B, 1, D]
        k = self.k_proj(x)  # [B, 1, D]
        v = self.v_proj(x)  # [B, 1, D]
        
        # 重塑为多头
        q = q.view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, 1, D/H]
        k = k.view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, 1, D/H]
        v = v.view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, 1, D/H]
        
        # 注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 加权求和
        attn_output = torch.matmul(attn_weights, v)  # [B, H, 1, D/H]
        
        # 重塑回原来的形状
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, 1, D)  # [B, 1, D]
        attn_output = attn_output.squeeze(1)  # [B, D]
        
        # 输出投影
        output = self.out_proj(attn_output)
        
        # 残差连接和层归一化
        return self.norm(x.squeeze(1) + output)