"""
DiT (Diffusion Transformer) Model
基于Transformer的可扩展diffusion模型

论文参考: "Scalable Diffusion Models with Transformers"
支持不同的架构配置和条件输入
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from einops import rearrange, repeat


class DiTDiffusion(nn.Module):
    """
    DiT (Diffusion Transformer) 模型
    
    相比UNet，DiT具有更好的可扩展性和表达能力
    特别适合处理高维特征和复杂条件
    """
    
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 512,
                 num_layers: int = 12,
                 num_heads: int = 8,
                 condition_dim: int = 128,
                 patch_size: int = 1,  # 对于1D特征，patch_size=1
                 dropout: float = 0.1,
                 use_flash_attention: bool = False,
                 rope: bool = True,  # 使用RoPE位置编码
                 **kwargs):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.condition_dim = condition_dim
        self.patch_size = patch_size
        self.dropout = dropout
        self.use_flash_attention = use_flash_attention
        self.rope = rope
        
        # 输入嵌入
        self.input_embedding = nn.Linear(input_dim * patch_size, hidden_dim)
        
        # 时间嵌入
        self.time_embedding = TimestepEmbedding(hidden_dim)
        
        # 条件嵌入
        self.condition_embedding = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 位置编码
        if rope:
            self.pos_encoder = RoPEEmbedding(hidden_dim // num_heads)
        else:
            self.pos_encoder = PositionalEncoding(hidden_dim, dropout)
        
        # DiT Transformer块
        self.transformer_blocks = nn.ModuleList([
            DiTBlock(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                use_flash_attention=use_flash_attention,
                rope=rope
            )
            for _ in range(num_layers)
        ])
        
        # 最终层归一化
        self.final_norm = nn.LayerNorm(hidden_dim)
        
        # 输出投影
        self.output_proj = nn.Linear(hidden_dim, input_dim * patch_size)
        
        # 初始化权重
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
        
        # 添加序列维度进行patch化
        x = x.view(B, 1, D)  # [B, 1, D]
        
        # 输入嵌入
        x = self.input_embedding(x)  # [B, 1, hidden_dim]
        
        # 时间嵌入
        time_emb = self.time_embedding(timestep)  # [B, hidden_dim]
        
        # 条件嵌入
        if condition is not None:
            cond_emb = self.condition_embedding(condition)  # [B, hidden_dim]
        else:
            cond_emb = torch.zeros_like(time_emb)
        
        # 位置编码
        if not self.rope:
            x = self.pos_encoder(x)
        
        # 通过DiT块
        for block in self.transformer_blocks:
            x = block(x, time_emb, cond_emb)
        
        # 最终层归一化
        x = self.final_norm(x)
        
        # 输出投影
        x = self.output_proj(x)  # [B, 1, input_dim]
        
        return x.squeeze(1)  # [B, input_dim]
    
    def get_model_size(self) -> int:
        """获取模型大小"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class DiTBlock(nn.Module):
    """DiT Transformer块"""
    
    def __init__(self,
                 hidden_dim: int,
                 num_heads: int,
                 dropout: float = 0.1,
                 use_flash_attention: bool = False,
                 rope: bool = True):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.use_flash_attention = use_flash_attention
        self.rope = rope
        
        # 归一化层
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # 自注意力
        if use_flash_attention:
            try:
                from flash_attn import flash_attn_func
                self.use_flash_attn = True
            except ImportError:
                self.use_flash_attn = False
                print("Flash attention not available, using standard attention")
        else:
            self.use_flash_attn = False
        
        self.attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        
        # 前馈网络
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        
        # 条件调制层 (AdaLN)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim * 6)  # 6个调制参数
        )
        
        # 门控机制
        self.gate_proj = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, 
                x: torch.Tensor,
                time_emb: torch.Tensor,
                cond_emb: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入特征 [B, seq_len, hidden_dim]
            time_emb: 时间嵌入 [B, hidden_dim]
            cond_emb: 条件嵌入 [B, hidden_dim]
            
        Returns:
            output: 输出特征 [B, seq_len, hidden_dim]
        """
        # 结合时间和条件嵌入
        combined_emb = time_emb + cond_emb
        
        # AdaLN调制参数
        modulation = self.adaLN_modulation(combined_emb)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = modulation.chunk(6, dim=1)
        
        # 自注意力分支
        h = self.norm1(x)
        h = h * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        
        # 注意力计算
        attn_output, _ = self.attention(h, h, h)
        
        # 门控和残差连接
        attn_output = attn_output * gate_msa.unsqueeze(1)
        x = x + attn_output
        
        # 前馈网络分支
        h = self.norm2(x)
        h = h * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        
        # 前馈网络
        ff_output = self.feed_forward(h)
        
        # 门控和残差连接
        ff_output = ff_output * gate_mlp.unsqueeze(1)
        x = x + ff_output
        
        return x


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


class RoPEEmbedding(nn.Module):
    """旋转位置编码 (RoPE)"""
    
    def __init__(self, dim: int, max_len: int = 2048):
        super().__init__()
        self.dim = dim
        
        # 创建频率
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # 预计算位置编码
        t = torch.arange(max_len, dtype=torch.float32)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer('cos_cached', emb.cos())
        self.register_buffer('sin_cached', emb.sin())
    
    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        返回cos和sin值用于旋转位置编码
        
        Args:
            x: 输入张量 [B, seq_len, dim]
            seq_len: 序列长度
            
        Returns:
            cos, sin: 旋转位置编码 [1, seq_len, dim]
        """
        return (
            self.cos_cached[:seq_len].unsqueeze(0),
            self.sin_cached[:seq_len].unsqueeze(0)
        )


class PositionalEncoding(nn.Module):
    """标准正弦位置编码"""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return self.dropout(x)
