"""
增强的隐状态实现
明确体现HMM中隐状态的概念，并支持深度差值预测
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class HiddenStateConfig:
    """隐状态配置"""
    world_state_dim: int = 512      # 世界状态维度
    depth_state_dim: int = 256      # 深度状态维度
    feature_state_dim: int = 256    # 特征状态维度
    context_state_dim: int = 128    # 上下文状态维度
    memory_size: int = 1000         # 记忆银行大小
    predict_depth_delta: bool = True  # 是否预测深度差值


class WorldStateRepresentation(nn.Module):
    """
    世界状态表示 - 核心隐状态
    
    这是HMM中最重要的隐状态，表示对整个环境的理解
    """
    
    def __init__(self, config: HiddenStateConfig):
        super().__init__()
        self.config = config
        
        # 世界状态记忆银行 - 长期隐状态
        self.world_memory = nn.Parameter(
            torch.randn(config.memory_size, config.world_state_dim) * 0.1
        )
        
        # 世界状态编码器 - 将观测编码为世界状态
        self.world_encoder = nn.Sequential(
            nn.Linear(config.world_state_dim, config.world_state_dim),
            nn.ReLU(),
            nn.LayerNorm(config.world_state_dim),
            nn.Linear(config.world_state_dim, config.world_state_dim)
        )
        
        # 世界状态解码器 - 从世界状态解码观测
        self.world_decoder = nn.Sequential(
            nn.Linear(config.world_state_dim, config.world_state_dim),
            nn.ReLU(),
            nn.LayerNorm(config.world_state_dim),
            nn.Linear(config.world_state_dim, config.world_state_dim)
        )
        
        # 世界状态转移网络 - 状态演化
        self.world_transition = nn.Sequential(
            nn.Linear(config.world_state_dim + 16, config.world_state_dim),  # +16 for ego motion
            nn.ReLU(),
            nn.LayerNorm(config.world_state_dim),
            nn.Linear(config.world_state_dim, config.world_state_dim)
        )
        
        # 记忆访问网络
        self.memory_attention = nn.MultiheadAttention(
            config.world_state_dim, num_heads=8, dropout=0.1
        )
        
        # 世界状态质量评估网络
        self.state_quality = nn.Sequential(
            nn.Linear(config.world_state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # 初始化世界状态
        self.current_world_state = nn.Parameter(
            torch.randn(1, config.world_state_dim) * 0.1
        )
        
    def forward(self, observations: torch.Tensor, ego_motion: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        世界状态更新
        
        Args:
            observations: 当前观测 [B, obs_dim]
            ego_motion: 自车运动 [B, 16]
            
        Returns:
            updated_world_state: 更新后的世界状态
            state_info: 状态信息
        """
        batch_size = observations.shape[0]
        
        # 1. 编码当前观测为世界状态
        encoded_obs = self.world_encoder(observations)
        
        # 2. 从记忆银行中检索相关状态
        memory_states, memory_weights = self.memory_attention(
            encoded_obs.unsqueeze(1),  # query
            self.world_memory.unsqueeze(0).expand(batch_size, -1, -1),  # key
            self.world_memory.unsqueeze(0).expand(batch_size, -1, -1)   # value
        )
        memory_states = memory_states.squeeze(1)
        
        # 3. 融合当前观测和记忆状态
        fused_state = 0.7 * encoded_obs + 0.3 * memory_states
        
        # 4. 状态转移（考虑自车运动）
        state_with_motion = torch.cat([fused_state, ego_motion], dim=-1)
        next_world_state = self.world_transition(state_with_motion)
        
        # 5. 世界状态质量评估
        state_quality = self.state_quality(next_world_state)
        
        # 6. 更新记忆银行（选择性更新）
        self._update_memory_bank(next_world_state, state_quality)
        
        state_info = {
            'encoded_obs': encoded_obs,
            'memory_states': memory_states,
            'memory_weights': memory_weights,
            'fused_state': fused_state,
            'state_quality': state_quality,
            'world_state': next_world_state
        }
        
        return next_world_state, state_info
    
    def _update_memory_bank(self, new_states: torch.Tensor, quality_scores: torch.Tensor):
        """更新记忆银行"""
        # 只更新高质量的状态
        high_quality_mask = quality_scores.squeeze(-1) > 0.7
        
        if high_quality_mask.any():
            # 随机选择要更新的记忆位置
            update_indices = torch.randint(0, self.config.memory_size, (high_quality_mask.sum(),))
            high_quality_states = new_states[high_quality_mask]
            
            # 更新记忆银行
            with torch.no_grad():
                for i, idx in enumerate(update_indices):
                    self.world_memory[idx] = 0.9 * self.world_memory[idx] + 0.1 * high_quality_states[i]
    
    def get_world_representation(self) -> torch.Tensor:
        """获取当前世界状态表示"""
        return self.world_decoder(self.current_world_state)


class DepthStateSpace(nn.Module):
    """深度状态空间 - 专门处理深度的隐状态"""
    
    def __init__(self, config: HiddenStateConfig):
        super().__init__()
        self.config = config
        
        # 深度差值预测网络
        if config.predict_depth_delta:
            self.depth_delta_predictor = nn.Sequential(
                nn.Linear(config.depth_state_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 64),  # 预测深度差值
                nn.Tanh()  # 限制差值范围
            )
    
    def forward(self, depth_features: torch.Tensor, ego_motion: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """深度状态处理"""
        # 预测深度差值 delta_D
        if self.config.predict_depth_delta:
            depth_delta = self.depth_delta_predictor(depth_features)
            depth_output = depth_features + depth_delta
            return depth_output, {'depth_delta': depth_delta}
        else:
            return depth_features, {}


# 使用示例
if __name__ == "__main__":
    config = HiddenStateConfig(predict_depth_delta=True)
    depth_state = DepthStateSpace(config)
    
    batch_size = 4
    depth_features = torch.randn(batch_size, 64)
    ego_motion = torch.randn(batch_size, 16)
    
    depth_pred, depth_info = depth_state(depth_features, ego_motion)
    
    print("=== 深度差值预测测试 ===")
    print(f"输入深度特征: {depth_features.shape}")
    print(f"预测深度特征: {depth_pred.shape}")
    if 'depth_delta' in depth_info:
        print(f"深度差值: {depth_info['depth_delta'].shape}")
        print(f"深度差值范围: [{depth_info['depth_delta'].min():.3f}, {depth_info['depth_delta'].max():.3f}]") 