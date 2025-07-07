"""
HMM训练损失函数模块
实现基于马尔可夫性质的损失函数

主要损失组件：
1. 深度似然损失 P(Dt|Xt)
2. 测量似然损失 P(Mt|Yt)  
3. BEV特征KL散度损失
4. 状态转移KL散度损失
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple
import math


class HMMLoss(nn.Module):
    """
    HMM训练损失函数
    
    符合马尔可夫性质的概率似然损失：
    - P(Dt|Xt): 深度似然
    - P(Mt|Yt): 测量似然
    - KL散度用于正则化状态转移
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
        
        # 损失函数
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.l1_loss = nn.L1Loss()
        
        # 分布估计网络（简化的高斯参数估计）
        self.depth_dist_net = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64 * 2)  # 均值和方差
        )
        
        self.measurement_dist_net = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64 * 2)  # 均值和方差
        )
        
        self.bev_pred_net = nn.Sequential(
            nn.Linear(64 + 6, 128),  # splat + action
            nn.ReLU(),
            nn.Linear(128, 64 * 2)  # 预测下一状态的均值和方差
        )
        
        self.state_pred_net = nn.Sequential(
            nn.Linear(64 + 6, 128),  # lift + action
            nn.ReLU(),
            nn.Linear(128, 64 * 2)  # 预测下一状态的均值和方差
        )
    
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
        计算HMM损失
        
        Returns:
            损失字典，包含各个组件和总损失
        """
        
        losses = {}
        
        # 1. 深度似然损失 P(Dt|Xt)
        depth_loss = self._compute_depth_loss(lift_features, depth_features)
        losses['depth_loss'] = depth_loss
        
        # 2. 测量似然损失 P(Mt|Yt)
        measurement_loss = self._compute_measurement_loss(splat_features, measurements)
        losses['measurement_loss'] = measurement_loss
        
        # 3. BEV特征KL散度损失
        bev_kl_loss = self._compute_bev_kl_loss(lift_next, splat_current, 
                                               actions, splat_next_true)
        losses['bev_kl_loss'] = bev_kl_loss
        
        # 4. 状态转移KL散度损失
        state_kl_loss = self._compute_state_kl_loss(observations, lift_current,
                                                   actions, lift_next_true)
        losses['state_kl_loss'] = state_kl_loss
        
        # 5. 总损失
        total_loss = (
            self.depth_weight * depth_loss +
            self.measurement_weight * measurement_loss +
            self.bev_kl_weight * bev_kl_loss +
            self.state_kl_weight * state_kl_loss
        )
        
        losses['total_loss'] = total_loss
        
        return losses
    
    def _compute_depth_loss(self, lift_features: torch.Tensor, depth_features: torch.Tensor) -> torch.Tensor:
        """
        计算深度似然损失 P(Dt|Xt)
        
        使用高斯分布假设：Dt ~ N(μ(Xt), σ²(Xt))
        """
        B, T, D = lift_features.shape
        
        # 展平时序维度进行处理
        lift_flat = lift_features.view(-1, D)  # [B*T, D]
        depth_flat = depth_features.view(-1, D)  # [B*T, D]
        
        # 预测深度分布参数
        depth_params = self.depth_dist_net(lift_flat)  # [B*T, D*2]
        depth_mu, depth_logvar = depth_params.chunk(2, dim=-1)  # [B*T, D]
        
        # 计算负对数似然
        depth_sigma = torch.exp(0.5 * depth_logvar)
        likelihood = -0.5 * torch.sum(
            torch.log(2 * math.pi * depth_sigma**2) + 
            (depth_flat - depth_mu)**2 / depth_sigma**2,
            dim=-1
        )
        
        # 返回负似然（即损失）
        return -likelihood.mean()
    
    def _compute_measurement_loss(self, splat_features: torch.Tensor, measurements: torch.Tensor) -> torch.Tensor:
        """
        计算测量似然损失 P(Mt|Yt)
        
        这里measurements是BEV分割标签，使用伯努利分布假设
        """
        B, T, D = splat_features.shape
        B, T, M = measurements.shape
        
        # 展平处理
        splat_flat = splat_features.view(-1, D)  # [B*T, D]
        measurements_flat = measurements.view(-1, M)  # [B*T, M]
        
        # 预测测量分布参数
        measurement_params = self.measurement_dist_net(splat_flat)  # [B*T, D*2]
        
        # 简化：直接用特征预测测量
        # 需要一个投影层将splat特征投影到测量空间
        if not hasattr(self, '_measurement_proj'):
            self._measurement_proj = nn.Linear(D, M).to(splat_flat.device)
        
        measurement_pred = self._measurement_proj(splat_flat)  # [B*T, M]
        
        # 使用BCE损失作为伯努利似然的近似
        measurement_loss = self.bce_loss(measurement_pred, measurements_flat)
        
        return measurement_loss
    
    def _compute_bev_kl_loss(self, lift_next: torch.Tensor, splat_current: torch.Tensor, 
                           actions: torch.Tensor, splat_next_true: torch.Tensor) -> torch.Tensor:
        """
        计算BEV特征KL散度损失
        
        KL(P(Yt+1|Yt, At) || Q(Yt+1|Xt+1))
        """
        B, T, D = splat_current.shape
        B, T, A = actions.shape
        
        # 展平处理
        splat_flat = splat_current.view(-1, D)  # [B*T, D]
        actions_flat = actions.view(-1, A)  # [B*T, A]
        splat_next_flat = splat_next_true.view(-1, D)  # [B*T, D]
        
        # 预测下一个BEV特征的分布 P(Yt+1|Yt, At)
        pred_input = torch.cat([splat_flat, actions_flat], dim=-1)  # [B*T, D+A]
        pred_params = self.bev_pred_net(pred_input)  # [B*T, D*2]
        pred_mu, pred_logvar = pred_params.chunk(2, dim=-1)  # [B*T, D]
        
        # 真实下一状态作为目标分布 Q(Yt+1|Xt+1)
        # 假设为单位方差高斯分布
        true_mu = splat_next_flat
        true_logvar = torch.zeros_like(true_mu)
        
        # 计算KL散度
        kl_loss = self._kl_divergence(pred_mu, pred_logvar, true_mu, true_logvar)
        
        return kl_loss.mean()
    
    def _compute_state_kl_loss(self, observations: torch.Tensor, lift_current: torch.Tensor,
                             actions: torch.Tensor, lift_next_true: torch.Tensor) -> torch.Tensor:
        """
        计算状态转移KL散度损失
        
        KL(P(Xt+1|Xt, At) || Q(Xt+1|Ot+1))
        """
        B, T, D = lift_current.shape
        B, T, A = actions.shape
        
        # 展平处理
        lift_flat = lift_current.view(-1, D)  # [B*T, D]
        actions_flat = actions.view(-1, A)  # [B*T, A]
        lift_next_flat = lift_next_true.view(-1, D)  # [B*T, D]
        
        # 预测下一个状态的分布 P(Xt+1|Xt, At)
        pred_input = torch.cat([lift_flat, actions_flat], dim=-1)  # [B*T, D+A]
        pred_params = self.state_pred_net(pred_input)  # [B*T, D*2]
        pred_mu, pred_logvar = pred_params.chunk(2, dim=-1)  # [B*T, D]
        
        # 真实下一状态作为目标分布 Q(Xt+1|Ot+1)
        # 假设为单位方差高斯分布
        true_mu = lift_next_flat
        true_logvar = torch.zeros_like(true_mu)
        
        # 计算KL散度
        kl_loss = self._kl_divergence(pred_mu, pred_logvar, true_mu, true_logvar)
        
        return kl_loss.mean()
    
    def _kl_divergence(self, q_mu: torch.Tensor, q_logvar: torch.Tensor,
                      p_mu: torch.Tensor, p_logvar: torch.Tensor) -> torch.Tensor:
        """
        计算两个高斯分布的KL散度
        
        KL(q||p) = log(σp/σq) + (σq² + (μq-μp)²)/(2σp²) - 1/2
        """
        kl = (p_logvar - q_logvar) + \
             (torch.exp(q_logvar) + (q_mu - p_mu)**2) / (2 * torch.exp(p_logvar)) - 0.5
        
        return torch.sum(kl, dim=-1)
    
    def get_loss_info(self) -> Dict[str, str]:
        """获取损失函数信息"""
        return {
            'depth_weight': f'{self.depth_weight}',
            'measurement_weight': f'{self.measurement_weight}',
            'bev_kl_weight': f'{self.bev_kl_weight}',
            'state_kl_weight': f'{self.state_kl_weight}',
            'temperature': f'{self.temperature}'
        }


def create_hmm_loss(config: Dict[str, Any]) -> nn.Module:
    """创建HMM损失函数"""
    loss_config = config.get('loss', {})
    
    return HMMLoss(
        depth_weight=loss_config.get('depth_weight', 1.0),
        measurement_weight=loss_config.get('measurement_weight', 1.0),
        bev_kl_weight=loss_config.get('bev_kl_weight', 0.5),
        state_kl_weight=loss_config.get('state_kl_weight', 0.5),
        temperature=loss_config.get('temperature', 1.0)
    ) 