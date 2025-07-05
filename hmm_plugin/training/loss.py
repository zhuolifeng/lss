"""
HMM插件损失函数
实现HMM理论的四个损失项

损失函数公式：
L = Σ log P(Dt|Xt) + Σ log P(Mt|Yt) - Σ DKL(q(Yt+1|Xt+1)||P(Yt+1|Yt,At)) - Σ DKL(q(Xt+1|Ot)||P(Xt+1|Xt,At))

四个部分：
1. log P(Dt|Xt): 深度似然 - 给定特征Xt预测深度Dt的概率
2. log P(Mt|Yt): 测量似然 - 给定BEV特征Yt预测测量Mt的概率
3. DKL(q(Yt+1|Xt+1)||P(Yt+1|Yt,At)): BEV特征KL散度
4. DKL(q(Xt+1|Ot)||P(Xt+1|Xt,At)): 状态转移KL散度
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple
import math


class HMMLoss(nn.Module):
    """
    HMM插件损失函数
    
    实现四个损失项的计算，支持时间步求和
    """
    
    def __init__(self,
                 depth_weight: float = 1.0,
                 measurement_weight: float = 1.0,
                 bev_kl_weight: float = 0.5,
                 state_kl_weight: float = 0.5,
                 temperature: float = 1.0):
        super().__init__()
        
        self.depth_weight = depth_weight
        self.measurement_weight = measurement_weight
        self.bev_kl_weight = bev_kl_weight
        self.state_kl_weight = state_kl_weight
        self.temperature = temperature
        
        # 深度似然网络 - 计算 P(Dt|Xt)
        self.depth_net = nn.Sequential(
            nn.Linear(128, 64),  # Xt_dim + Dt_dim
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # 测量似然网络 - 计算 P(Mt|Yt)
        self.measurement_net = nn.Sequential(
            nn.Linear(264, 128),  # Yt_dim + Mt_dim (64 + 200)
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # BEV特征KL散度网络 - 计算分布参数
        self.bev_q_net = nn.ModuleDict({
            'mu': nn.Sequential(
                nn.Linear(64, 128),
                nn.ReLU(),
                nn.Linear(128, 64)
            ),
            'logvar': nn.Sequential(
                nn.Linear(64, 128),
                nn.ReLU(),
                nn.Linear(128, 64)
            )
        })
        
        self.bev_p_net = nn.ModuleDict({
            'mu': nn.Sequential(
                nn.Linear(68, 128),  # Yt_dim + At_dim
                nn.ReLU(),
                nn.Linear(128, 64)
            ),
            'logvar': nn.Sequential(
                nn.Linear(68, 128),
                nn.ReLU(),
                nn.Linear(128, 64)
            )
        })
        
        # 状态转移KL散度网络 - 计算分布参数
        self.state_q_net = nn.ModuleDict({
            'mu': nn.Sequential(
                nn.Linear(64, 128),
                nn.ReLU(),
                nn.Linear(128, 64)
            ),
            'logvar': nn.Sequential(
                nn.Linear(64, 128),
                nn.ReLU(),
                nn.Linear(128, 64)
            )
        })
        
        self.state_p_net = nn.ModuleDict({
            'mu': nn.Sequential(
                nn.Linear(68, 128),  # Xt_dim + At_dim
                nn.ReLU(),
                nn.Linear(128, 64)
            ),
            'logvar': nn.Sequential(
                nn.Linear(68, 128),
                nn.ReLU(),
                nn.Linear(128, 64)
            )
        })
        
    def forward(self, 
                # 深度似然数据
                lift_features: torch.Tensor,      # [B, T, 64]
                depth_features: torch.Tensor,     # [B, T, 64]
                
                # 测量似然数据
                splat_features: torch.Tensor,     # [B, T, 64]
                measurements: torch.Tensor,       # [B, T, 200]
                
                # BEV特征KL散度数据
                lift_next: torch.Tensor,          # [B, T-1, 64]
                splat_current: torch.Tensor,      # [B, T-1, 64]
                actions: torch.Tensor,            # [B, T-1, 4]
                splat_next_true: torch.Tensor,    # [B, T-1, 64]
                
                # 状态转移KL散度数据
                observations: torch.Tensor,       # [B, T-1, 64]
                lift_current: torch.Tensor,       # [B, T-1, 64]
                lift_next_true: torch.Tensor      # [B, T-1, 64]
                ) -> Dict[str, torch.Tensor]:
        """
        计算HMM损失函数的四个部分
        """
        
        losses = {}
        
        # 1. 深度似然损失: log P(Dt|Xt)
        depth_loss = self._compute_depth_loss(lift_features, depth_features)
        losses['depth_loss'] = depth_loss
        
        # 2. 测量似然损失: log P(Mt|Yt)
        measurement_loss = self._compute_measurement_loss(splat_features, measurements)
        losses['measurement_loss'] = measurement_loss
        
        # 3. BEV特征KL散度损失
        bev_kl_loss = self._compute_bev_kl_loss(lift_next, splat_current, actions, splat_next_true)
        losses['bev_kl_loss'] = bev_kl_loss
        
        # 4. 状态转移KL散度损失
        state_kl_loss = self._compute_state_kl_loss(observations, lift_current, actions, lift_next_true)
        losses['state_kl_loss'] = state_kl_loss
        
        # 总损失（包含时间步求和）
        total_loss = (
            self.depth_weight * depth_loss +
            self.measurement_weight * measurement_loss +
            self.bev_kl_weight * bev_kl_loss +
            self.state_kl_weight * state_kl_loss
        )
        
        losses['total_loss'] = total_loss
        
        return losses
    
    def _compute_depth_loss(self, lift_features: torch.Tensor, depth_features: torch.Tensor) -> torch.Tensor:
        """计算深度似然损失: -log P(Dt|Xt)"""
        B, T, _ = lift_features.shape
        
        # 时间步求和（这里实现了图像中的Σ t=1 to T）
        log_likelihoods = []
        for t in range(T):
            combined = torch.cat([lift_features[:, t], depth_features[:, t]], dim=-1)
            likelihood = self.depth_net(combined)
            log_likelihood = torch.log(likelihood + 1e-8)
            log_likelihoods.append(log_likelihood.squeeze())
        
        # 对时间维度求和
        total_log_likelihood = torch.stack(log_likelihoods, dim=1).sum(dim=1)
        return -total_log_likelihood.mean()
    
    def _compute_measurement_loss(self, splat_features: torch.Tensor, measurements: torch.Tensor) -> torch.Tensor:
        """计算测量似然损失: -log P(Mt|Yt)"""
        B, T, _ = splat_features.shape
        
        # 时间步求和
        log_likelihoods = []
        for t in range(T):
            combined = torch.cat([splat_features[:, t], measurements[:, t]], dim=-1)
            likelihood = self.measurement_net(combined)
            log_likelihood = torch.log(likelihood + 1e-8)
            log_likelihoods.append(log_likelihood.squeeze())
        
        # 对时间维度求和
        total_log_likelihood = torch.stack(log_likelihoods, dim=1).sum(dim=1)
        return -total_log_likelihood.mean()
    
    def _compute_bev_kl_loss(self, lift_next: torch.Tensor, splat_current: torch.Tensor, 
                           actions: torch.Tensor, splat_next_true: torch.Tensor) -> torch.Tensor:
        """计算BEV特征KL散度损失"""
        B, T, _ = lift_next.shape
        
        # 时间步求和
        kl_divergences = []
        for t in range(T):
            # q分布参数
            q_mu = self.bev_q_net['mu'](lift_next[:, t])
            q_logvar = self.bev_q_net['logvar'](lift_next[:, t])
            
            # p分布参数
            p_input = torch.cat([splat_current[:, t], actions[:, t]], dim=-1)
            p_mu = self.bev_p_net['mu'](p_input)
            p_logvar = self.bev_p_net['logvar'](p_input)
            
            # KL散度
            kl_div = self._kl_divergence(q_mu, q_logvar, p_mu, p_logvar)
            kl_divergences.append(kl_div)
        
        # 对时间维度求和
        total_kl = torch.stack(kl_divergences, dim=1).sum(dim=1)
        return total_kl.mean()
    
    def _compute_state_kl_loss(self, observations: torch.Tensor, lift_current: torch.Tensor,
                             actions: torch.Tensor, lift_next_true: torch.Tensor) -> torch.Tensor:
        """计算状态转移KL散度损失"""
        B, T, _ = observations.shape
        
        # 时间步求和
        kl_divergences = []
        for t in range(T):
            # q分布参数
            q_mu = self.state_q_net['mu'](observations[:, t])
            q_logvar = self.state_q_net['logvar'](observations[:, t])
            
            # p分布参数
            p_input = torch.cat([lift_current[:, t], actions[:, t]], dim=-1)
            p_mu = self.state_p_net['mu'](p_input)
            p_logvar = self.state_p_net['logvar'](p_input)
            
            # KL散度
            kl_div = self._kl_divergence(q_mu, q_logvar, p_mu, p_logvar)
            kl_divergences.append(kl_div)
        
        # 对时间维度求和
        total_kl = torch.stack(kl_divergences, dim=1).sum(dim=1)
        return total_kl.mean()
    
    def _kl_divergence(self, q_mu: torch.Tensor, q_logvar: torch.Tensor,
                      p_mu: torch.Tensor, p_logvar: torch.Tensor) -> torch.Tensor:
        """计算KL散度 KL(q||p)"""
        kl_div = 0.5 * torch.sum(
            torch.exp(q_logvar - p_logvar) + 
            (p_mu - q_mu).pow(2) / torch.exp(p_logvar) - 
            1 + 
            p_logvar - q_logvar,
            dim=-1
        )
        return kl_div
    
    def get_loss_info(self) -> Dict[str, str]:
        """返回损失函数说明"""
        return {
            'depth_loss': 'log P(Dt|Xt) - 深度似然：给定特征预测深度的概率',
            'measurement_loss': 'log P(Mt|Yt) - 测量似然：给定BEV特征预测测量的概率',
            'bev_kl_loss': 'DKL(q(Yt+1|Xt+1)||P(Yt+1|Yt,At)) - BEV特征KL散度',
            'state_kl_loss': 'DKL(q(Xt+1|Ot)||P(Xt+1|Xt,At)) - 状态转移KL散度'
        }


def create_hmm_loss(config: Dict[str, Any]) -> nn.Module:
    """创建HMM损失函数"""
    return HMMLoss(
        depth_weight=config.get('depth_weight', 1.0),
        measurement_weight=config.get('measurement_weight', 1.0),
        bev_kl_weight=config.get('bev_kl_weight', 0.5),
        state_kl_weight=config.get('state_kl_weight', 0.5),
        temperature=config.get('temperature', 1.0)
    ) 