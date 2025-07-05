"""
权重学习模块
学习如何融合不同来源的特征

核心功能：
- 学习Xt + Xt' → final_Xt的权重
- 学习Yt + Yt' → final_Yt的权重  
- 学习Yt + Yt' + Yt'' → final_Yt的三路权重
- 根据特征质量动态调整融合权重
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional


class WeightLearner(nn.Module):
    """
    权重学习器
    
    学习如何融合：
    - fused_feature = w * original_feature + (1-w) * predicted_feature
    """
    
    def __init__(self,
                 lift_dim: int,
                 splat_dim: int,
                 hidden_dim: int = 256):
        super().__init__()
        
        self.lift_dim = lift_dim
        self.splat_dim = splat_dim
        self.hidden_dim = hidden_dim
        
        # Lift权重网络：学习Xt + Xt' → final_Xt
        self.lift_weight_net = nn.Sequential(
            nn.Linear(self.lift_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Splat权重网络：学习Yt + Yt' → final_Yt
        self.splat_weight_net = nn.Sequential(
            nn.Linear(self.splat_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # 质量评估网络
        self.quality_net = nn.Sequential(
            nn.Linear(self.lift_dim + self.splat_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self,
                lift_orig: torch.Tensor,        # 原始lift特征
                lift_pred: torch.Tensor,        # HMM预测lift特征
                splat_orig: torch.Tensor,       # 原始splat特征
                splat_pred: torch.Tensor        # HMM预测splat特征
                ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        前向传播：学习权重并融合特征
        
        Returns:
            final_lift: 融合后的lift特征
            final_splat: 融合后的splat特征
            info: 权重和损失信息
        """
        
        # 1. 学习lift融合权重
        lift_concat = torch.cat([lift_orig, lift_pred], dim=-1)
        lift_weight = self.lift_weight_net(lift_concat)  # [B, 1]
        
        # 2. 学习splat融合权重
        splat_concat = torch.cat([splat_orig, splat_pred], dim=-1)
        splat_weight = self.splat_weight_net(splat_concat)  # [B, 1]
        
        # 3. 权重融合
        final_lift = lift_weight * lift_orig + (1 - lift_weight) * lift_pred
        final_splat = splat_weight * splat_orig + (1 - splat_weight) * splat_pred
        
        # 4. 质量评估
        combined_features = torch.cat([final_lift, final_splat], dim=-1)
        quality_score = self.quality_net(combined_features)
        
        # 5. 计算损失
        consistency_loss = self._compute_consistency_loss(
            lift_orig, lift_pred, splat_orig, splat_pred, 
            lift_weight, splat_weight
        )
        
        weight_reg_loss = self._compute_weight_regularization(lift_weight, splat_weight)
        
        info = {
            'lift_weight': lift_weight,
            'splat_weight': splat_weight,
            'quality_score': quality_score,
            'consistency_loss': consistency_loss,
            'weight_reg_loss': weight_reg_loss,
            'total_fusion_loss': consistency_loss + 0.1 * weight_reg_loss
        }
        
        return final_lift, final_splat, info
    
    def _compute_consistency_loss(self, 
                                lift_orig: torch.Tensor,
                                lift_pred: torch.Tensor,
                                splat_orig: torch.Tensor,
                                splat_pred: torch.Tensor,
                                lift_weight: torch.Tensor,
                                splat_weight: torch.Tensor) -> torch.Tensor:
        """计算一致性损失"""
        
        # 特征相似度
        lift_similarity = F.cosine_similarity(lift_orig, lift_pred, dim=-1)
        splat_similarity = F.cosine_similarity(splat_orig, splat_pred, dim=-1)
        
        # 期望权重：相似度高时权重接近0.5
        expected_lift_weight = 0.5 + 0.3 * (1 - lift_similarity)
        expected_splat_weight = 0.5 + 0.3 * (1 - splat_similarity)
        
        # 一致性损失
        lift_consistency = F.mse_loss(lift_weight.squeeze(), expected_lift_weight)
        splat_consistency = F.mse_loss(splat_weight.squeeze(), expected_splat_weight)
        
        return lift_consistency + splat_consistency
    
    def _compute_weight_regularization(self, 
                                     lift_weight: torch.Tensor,
                                     splat_weight: torch.Tensor) -> torch.Tensor:
        """权重正则化损失"""
        # 防止权重过于极端
        lift_reg = torch.mean(torch.min(lift_weight, 1 - lift_weight))
        splat_reg = torch.mean(torch.min(splat_weight, 1 - splat_weight))
        
        return -(lift_reg + splat_reg)


class TripleFusion(nn.Module):
    """
    三路融合模块：Yt + Yt' + Yt'' → final_Yt
    
    说明：
    - Yt: 原始splat特征
    - Yt': HMM预测splat特征  
    - Yt'': 深度增强splat特征（由LSS模型的深度处理模块提供）
    """
    
    def __init__(self, splat_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        self.splat_dim = splat_dim
        self.hidden_dim = hidden_dim
        
        # 三路权重网络
        self.weight_net = nn.Sequential(
            nn.Linear(self.splat_dim * 3, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, 3),
            nn.Softmax(dim=-1)  # 权重和为1
        )
        
        # 质量评估网络
        self.quality_net = nn.Sequential(
            nn.Linear(self.splat_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # 最终融合网络（可选的后处理）
        self.final_net = nn.Sequential(
            nn.Linear(self.splat_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.splat_dim)
        )
    
    def forward(self,
                splat_orig: torch.Tensor,      # 原始splat特征
                splat_pred: torch.Tensor,      # HMM预测splat特征
                splat_enhanced: Optional[torch.Tensor] = None   # 深度增强splat特征（LSS提供）
                ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        三路融合
        
        Args:
            splat_orig: 原始splat特征 [B, splat_dim]
            splat_pred: HMM预测splat特征 [B, splat_dim]
            splat_enhanced: 深度增强splat特征 [B, splat_dim]，由LSS模型提供
        
        Returns:
            final_splat: 融合后的splat特征
            info: 权重和质量信息
        """
        
        # 如果没有深度增强特征，使用零特征
        if splat_enhanced is None:
            splat_enhanced = torch.zeros_like(splat_orig)
        
        # 1. 拼接三个特征
        all_features = torch.cat([splat_orig, splat_pred, splat_enhanced], dim=-1)
        
        # 2. 学习基础权重
        base_weights = self.weight_net(all_features)  # [B, 3]
        
        # 3. 评估各路径质量
        quality_orig = self.quality_net(splat_orig)
        quality_pred = self.quality_net(splat_pred)
        quality_enhanced = self.quality_net(splat_enhanced)
        
        # 4. 质量权重
        quality_weights = torch.cat([quality_orig, quality_pred, quality_enhanced], dim=-1)
        quality_weights = F.softmax(quality_weights, dim=-1)
        
        # 5. 最终权重（基础权重 × 质量权重）
        final_weights = base_weights * quality_weights
        final_weights = final_weights / (final_weights.sum(dim=-1, keepdim=True) + 1e-8)
        
        # 6. 加权融合
        final_splat = (final_weights[:, 0:1] * splat_orig + 
                      final_weights[:, 1:2] * splat_pred + 
                      final_weights[:, 2:3] * splat_enhanced)
        
        # 7. 可选的后处理（残差连接）
        enhanced_splat = self.final_net(final_splat) + final_splat
        
        # 8. 计算损失
        diversity_loss = self._compute_diversity_loss(final_weights)
        balance_loss = self._compute_balance_loss(final_weights)
        
        info = {
            'base_weights': base_weights,
            'quality_weights': quality_weights,
            'final_weights': final_weights,
            'diversity_loss': diversity_loss,
            'balance_loss': balance_loss,
            'total_fusion_loss': diversity_loss + 0.1 * balance_loss,
            'weight_distribution': {
                'original': final_weights[:, 0].mean(),
                'predicted': final_weights[:, 1].mean(),
                'enhanced': final_weights[:, 2].mean()
            }
        }
        
        return enhanced_splat, info
    
    def _compute_diversity_loss(self, weights: torch.Tensor) -> torch.Tensor:
        """计算多样性损失，鼓励适度的权重分布"""
        diversity = torch.std(weights, dim=-1).mean()
        target_diversity = 0.2
        return F.mse_loss(diversity, torch.tensor(target_diversity, device=weights.device))
    
    def _compute_balance_loss(self, weights: torch.Tensor) -> torch.Tensor:
        """计算平衡损失，防止权重过于极端"""
        uniform_target = torch.ones_like(weights) / 3.0
        return F.mse_loss(weights, uniform_target) 