"""
基于Diffusion的Transition1
使用diffusion模型预测delta_X，支持CFG（Classifier-Free Guidance）

支持的架构：
1. UNet: 经典的diffusion架构
2. DiT: Diffusion Transformer
3. Mamba: 基于状态空间模型的diffusion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any
import math

from .base_transition import BaseTransition


class DiffusionTransition1(BaseTransition):
    """
    基于Diffusion的Transition1
    
    预测delta_X的过程：
    1. 输入：concat([Xt-1, encoded_actions])
    2. 加噪声：x_noisy = x_0 + noise * sigma
    3. 去噪预测：delta_X = model(x_noisy, t, condition)
    4. CFG：如果cfg_scale > 1，使用classifier-free guidance
    """
    
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 hidden_dim: int = 256,
                 model_type: str = 'unet',
                 num_layers: int = 4,
                 num_heads: int = 8,
                 diffusion_steps: int = 100,
                 cfg_scale: float = 1.0,
                 beta_schedule: str = 'cosine',
                 **kwargs):
        super().__init__(input_dim, output_dim, hidden_dim, model_type, **kwargs)
        
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.diffusion_steps = diffusion_steps
        self.cfg_scale = cfg_scale
        self.beta_schedule = beta_schedule
        
        # 设置噪声调度
        self.register_buffer('betas', self._get_beta_schedule(diffusion_steps, beta_schedule))
        self.register_buffer('alphas', 1.0 - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))
        
        # 创建核心diffusion模型
        self.core_network = self._create_diffusion_model()
        
        # 条件编码（用于CFG）- 延迟初始化以匹配实际输入维度
        self.condition_embedding = None
        self.expected_input_dim = input_dim
        
        # 输出投影
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        
    def _get_beta_schedule(self, timesteps: int, schedule: str) -> torch.Tensor:
        """获取噪声调度"""
        if schedule == 'linear':
            return torch.linspace(0.0001, 0.02, timesteps)
        elif schedule == 'cosine':
            s = 0.008
            steps = timesteps + 1
            x = torch.linspace(0, timesteps, steps)
            alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            return torch.clip(betas, 0, 0.999)
        else:
            raise ValueError(f"Unknown beta schedule: {schedule}")
    
    def _create_diffusion_model(self) -> nn.Module:
        """创建diffusion模型"""
        if self.model_type == 'unet':
            from ..models import UNetDiffusion
            return UNetDiffusion(
                input_dim=self.output_dim,  # 去噪的目标维度
                hidden_dim=self.hidden_dim,
                condition_dim=self.hidden_dim,
                num_layers=self.num_layers
            )
        elif self.model_type == 'dit':
            from ..models import DiTDiffusion
            return DiTDiffusion(
                input_dim=self.output_dim,
                hidden_dim=self.hidden_dim,
                condition_dim=self.hidden_dim,
                num_layers=self.num_layers,
                num_heads=self.num_heads
            )
        elif self.model_type == 'mamba':
            from ..models import MambaDiffusion
            return MambaDiffusion(
                input_dim=self.output_dim,
                hidden_dim=self.hidden_dim,
                condition_dim=self.hidden_dim,
                num_layers=self.num_layers
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def forward(self, x: torch.Tensor, training: bool = True) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入特征 [B, input_dim] = [B, lift_dim + action_dim]
            training: 是否为训练模式
            
        Returns:
            delta_X: 预测的delta_X [B, output_dim]
        """
        batch_size = x.shape[0]
        device = x.device
        
        if training:
            # 训练时使用随机时间步
            t = torch.randint(0, self.diffusion_steps, (batch_size,), device=device)
            
            # 目标是zero向量（我们预测的是delta）
            target = torch.zeros(batch_size, self.output_dim, device=device)
            
            # 添加噪声
            noise = torch.randn_like(target)
            alpha_t = self.alphas_cumprod[t].unsqueeze(-1)
            noisy_target = torch.sqrt(alpha_t) * target + torch.sqrt(1 - alpha_t) * noise
            
            # 预测噪声
            predicted_noise = self._predict_noise(noisy_target, t, x)
            
            # 恢复delta_X
            delta_X = (noisy_target - torch.sqrt(1 - alpha_t) * predicted_noise) / torch.sqrt(alpha_t)
            
            return delta_X
        else:
            # 推理时使用DDPM采样
            return self._sample_ddpm(x)
    
    def _predict_noise(self, noisy_x: torch.Tensor, t: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """预测噪声"""
        # 动态创建条件嵌入层（如果尚未创建）
        if self.condition_embedding is None:
            actual_input_dim = condition.shape[-1]
            self.condition_embedding = nn.Linear(actual_input_dim, self.hidden_dim).to(condition.device)
            print(f"动态创建条件嵌入层: {actual_input_dim} -> {self.hidden_dim}")
        
        # 条件编码
        cond_emb = self.condition_embedding(condition)
        
        # 如果使用CFG，需要同时进行条件和无条件预测
        if self.cfg_scale > 1.0 and self.training:
            # 无条件预测（条件置零）
            uncond_emb = torch.zeros_like(cond_emb)
            
            # 合并条件和无条件
            combined_x = torch.cat([noisy_x, noisy_x], dim=0)
            combined_t = torch.cat([t, t], dim=0)  # 直接传递时间步，不进行嵌入
            combined_cond = torch.cat([cond_emb, uncond_emb], dim=0)
            
            # 预测
            combined_pred = self.core_network(combined_x, combined_t, combined_cond)
            
            # 分离条件和无条件预测
            cond_pred, uncond_pred = combined_pred.chunk(2, dim=0)
            
            # CFG
            pred_noise = uncond_pred + self.cfg_scale * (cond_pred - uncond_pred)
        else:
            # 标准预测 - 直接传递时间步，让UNetDiffusion内部处理时间嵌入
            pred_noise = self.core_network(noisy_x, t, cond_emb)
        
        return pred_noise
    
    def _sample_ddpm(self, condition: torch.Tensor) -> torch.Tensor:
        """DDPM采样"""
        batch_size = condition.shape[0]
        device = condition.device
        
        # 从纯噪声开始
        x = torch.randn(batch_size, self.output_dim, device=device)
        
        # 逐步去噪
        for t in reversed(range(self.diffusion_steps)):
            t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)
            
            # 预测噪声
            pred_noise = self._predict_noise(x, t_tensor, condition)
            
            # 更新x
            alpha_t = self.alphas[t]
            alpha_t_cumprod = self.alphas_cumprod[t]
            beta_t = self.betas[t]
            
            # DDPM更新公式
            x = (1 / torch.sqrt(alpha_t)) * (x - (beta_t / torch.sqrt(1 - alpha_t_cumprod)) * pred_noise)
            
            # 添加噪声（除了最后一步）
            if t > 0:
                noise = torch.randn_like(x)
                x = x + torch.sqrt(beta_t) * noise
        
        return x
    
    def sample_with_cfg(self, condition: torch.Tensor, cfg_scale: float = None) -> torch.Tensor:
        """使用CFG采样"""
        if cfg_scale is None:
            cfg_scale = self.cfg_scale
        
        old_cfg_scale = self.cfg_scale
        self.cfg_scale = cfg_scale
        
        result = self._sample_ddpm(condition)
        
        self.cfg_scale = old_cfg_scale
        return result


# 示例使用
if __name__ == "__main__":
    # 测试DiffusionTransition1
    diffusion_transition = DiffusionTransition1(
        input_dim=192,  # 64 + 128
        output_dim=64,
        hidden_dim=256,
        model_type='unet',
        num_layers=4,
        diffusion_steps=100,
        cfg_scale=1.5
    )
    
    # 测试输入
    x = torch.randn(4, 192)
    
    # 训练模式
    diffusion_transition.train()
    delta_X_train = diffusion_transition(x, training=True)
    
    print(f"Diffusion Transition1:")
    print(f"  输入形状: {x.shape}")
    print(f"  输出形状 (训练): {delta_X_train.shape}")
    print(f"  参数数量: {sum(p.numel() for p in diffusion_transition.parameters())}")
    
    # 推理模式
    diffusion_transition.eval()
    with torch.no_grad():
        delta_X_eval = diffusion_transition(x, training=False)
        print(f"  输出形状 (推理): {delta_X_eval.shape}")
        
        # 使用CFG采样
        delta_X_cfg = diffusion_transition.sample_with_cfg(x, cfg_scale=2.0)
        print(f"  输出形状 (CFG): {delta_X_cfg.shape}") 