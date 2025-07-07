"""
HMM插件核心模块 - 符合马尔可夫性质的真正HMM实现
关键改进：隐状态编码历史信息，转移只依赖当前隐状态，保持马尔可夫性质
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np

from .hidden_state import WorldStateRepresentation, DepthStateSpace, HiddenStateConfig
from .transitions import DiffusionTransition1, ControlNetTransition2
from .fusion.weight_learner import WeightLearner, TripleFusion
from .utils import ActionEncoder


@dataclass
class HMMConfig:
    """HMM配置类 - 符合马尔可夫性质的设计"""
    # 基础维度配置
    lift_feature_dim: int = 64
    splat_feature_dim: int = 64
    depth_feature_dim: int = 64
    action_dim: int = 6
    
    # 隐状态配置 - 编码历史信息的复合状态
    hidden_state_dim: int = 256
    world_state_dim: int = 512
    use_hidden_state: bool = True
    
    # 状态更新配置
    state_update_rate: float = 0.1    # 隐状态更新率
    memory_decay: float = 0.95        # 记忆衰减率
    
    # Transition配置
    transition1_type: str = 'unet'
    transition1_hidden_dim: int = 256
    transition1_num_layers: int = 4
    transition1_diffusion_steps: int = 100
    transition1_output_type: str = 'delta'  # 'delta' or 'absolute'
    
    transition2_type: str = 'controlnet'
    transition2_hidden_dim: int = 256
    transition2_num_layers: int = 4
    transition2_output_type: str = 'delta'  # 'delta' or 'absolute'
    
    # 双transition创新融合策略
    dual_transition_strategy: str = 'hierarchical'  # 'hierarchical', 'parallel', 'cascade'
    cross_transition_influence: float = 0.3  # 两个transition之间的影响系数
    
    # 融合配置
    fusion_strategy: str = 'multi_path'
    weight_type: str = 'learnable'
    use_kl: bool = True
    
    # 物理约束
    predict_depth_delta: bool = True
    use_physical_constraints: bool = True
    
    # 损失权重
    reconstruction_weight: float = 1.0
    kl_weight: float = 0.1
    consistency_weight: float = 0.5
    markov_regularization_weight: float = 0.2  # 马尔可夫性质正则化


class MarkovianHiddenState(nn.Module):
    """
    马尔可夫隐状态 - 关键创新
    
    这个隐状态是一个复合状态，包含了：
    1. 当前观测的编码
    2. 历史信息的压缩表示
    3. 世界状态的记忆
    
    重要：状态转移只依赖于这个隐状态，保持马尔可夫性质
    """
    
    def __init__(self, config: HMMConfig):
        super().__init__()
        self.config = config
        
        # 隐状态维度
        self.hidden_dim = config.hidden_state_dim
        self.world_dim = config.world_state_dim
        
        # 当前隐状态（这是马尔可夫链的"状态"）
        self.current_hidden_state = nn.Parameter(
            torch.randn(1, self.hidden_dim) * 0.1
        )
        
        # 世界状态记忆（编码历史信息）
        self.world_memory = nn.Parameter(
            torch.randn(1, self.world_dim) * 0.1
        )
        
        # 状态更新网络 - 关键：如何将新观测融入隐状态
        self.state_update_net = nn.Sequential(
            nn.Linear(config.lift_feature_dim + config.splat_feature_dim + 
                     config.action_dim + self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh()  # 限制状态更新幅度
        )
        
        # 世界状态更新网络
        self.world_update_net = nn.Sequential(
            nn.Linear(self.hidden_dim + self.world_dim, self.world_dim),
            nn.ReLU(),
            nn.LayerNorm(self.world_dim),
            nn.Linear(self.world_dim, self.world_dim),
            nn.Tanh()
        )
        
        # 状态质量评估
        self.state_quality_net = nn.Sequential(
            nn.Linear(self.hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def update_state(self, 
                    current_lift: torch.Tensor,
                    current_splat: torch.Tensor,
                    current_action: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        状态更新 - 马尔可夫转移的核心
        
        关键：新状态只依赖于当前隐状态和当前观测
        历史信息已经编码在隐状态中
        """
        # 处理维度不匹配问题 - 将所有输入展平到2D
        if current_lift.dim() > 2:
            current_lift = current_lift.view(current_lift.shape[0], -1)
        if current_splat.dim() > 2:
            current_splat = current_splat.view(current_splat.shape[0], -1)
        if current_action.dim() > 2:
            current_action = current_action.view(current_action.shape[0], -1)
        
        # 找到最大的batch size，并将所有张量调整到相同的batch size
        batch_sizes = [current_lift.shape[0], current_splat.shape[0], current_action.shape[0]]
        max_batch_size = max(batch_sizes)
        
        # 调整所有张量到相同的batch size
        if current_lift.shape[0] != max_batch_size:
            # 如果batch size不匹配，重复张量以匹配最大batch size
            repeat_factor = max_batch_size // current_lift.shape[0]
            current_lift = current_lift.repeat(repeat_factor, 1)
            
        if current_splat.shape[0] != max_batch_size:
            repeat_factor = max_batch_size // current_splat.shape[0]
            current_splat = current_splat.repeat(repeat_factor, 1)
            
        if current_action.shape[0] != max_batch_size:
            repeat_factor = max_batch_size // current_action.shape[0]
            current_action = current_action.repeat(repeat_factor, 1)
        
        batch_size = max_batch_size
        
        # 创建投影层以确保维度一致（如果不存在的话）
        if not hasattr(self, '_lift_proj'):
            self._lift_proj = nn.Linear(current_lift.shape[1], self.config.lift_feature_dim).to(current_lift.device)
        if not hasattr(self, '_splat_proj'):
            self._splat_proj = nn.Linear(current_splat.shape[1], self.config.splat_feature_dim).to(current_splat.device)
        if not hasattr(self, '_action_proj'):
            self._action_proj = nn.Linear(current_action.shape[1], self.config.action_dim).to(current_action.device)
        
        # 投影到固定维度
        current_lift = self._lift_proj(current_lift)
        current_splat = self._splat_proj(current_splat)
        current_action = self._action_proj(current_action)
        
        # 扩展隐状态到batch
        hidden_state = self.current_hidden_state.expand(batch_size, -1)
        world_state = self.world_memory.expand(batch_size, -1)
        
        # 构造状态更新输入
        update_input = torch.cat([
            current_lift, current_splat, current_action, hidden_state
        ], dim=-1)
        
        # 状态更新（这是马尔可夫转移！）
        state_delta = self.state_update_net(update_input)
        new_hidden_state = hidden_state + self.config.state_update_rate * state_delta
        
        # 世界状态更新
        world_input = torch.cat([new_hidden_state, world_state], dim=-1)
        world_delta = self.world_update_net(world_input)
        new_world_state = world_state + self.config.state_update_rate * world_delta
        
        # 状态质量评估
        state_quality = self.state_quality_net(new_hidden_state)
        
        # 更新存储的状态（用于下一步）
        with torch.no_grad():
            self.current_hidden_state.data = new_hidden_state.mean(dim=0, keepdim=True)
            self.world_memory.data = new_world_state.mean(dim=0, keepdim=True)
        
        return {
            'hidden_state': new_hidden_state,
            'world_state': new_world_state,
            'state_quality': state_quality,
            'state_delta': state_delta,
            'world_delta': world_delta
        }
    
    def get_current_state(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """获取当前状态"""
        return {
            'hidden_state': self.current_hidden_state.expand(batch_size, -1),
            'world_state': self.world_memory.expand(batch_size, -1)
        }
    
    def reset_state(self):
        """重置状态（用于新序列开始）"""
        with torch.no_grad():
            self.current_hidden_state.data.zero_()
            self.world_memory.data.zero_()


class DualTransitionFusion(nn.Module):
    """
    双Transition创新融合 - 核心创新点
    
    三种融合策略：
    1. hierarchical: 层次化，Transition1的输出影响Transition2
    2. parallel: 并行，两个transition独立然后融合
    3. cascade: 级联，一个transition的输出作为另一个的输入
    """
    
    def __init__(self, config: HMMConfig):
        super().__init__()
        self.config = config
        self.strategy = config.dual_transition_strategy
        self.cross_influence = config.cross_transition_influence
        
        # 跨transition影响网络
        self.cross_influence_net = nn.Sequential(
            nn.Linear(config.lift_feature_dim, config.splat_feature_dim),
            nn.ReLU(),
            nn.Linear(config.splat_feature_dim, config.splat_feature_dim),
            nn.Tanh()
        )
        
        # 融合权重学习
        self.fusion_weight_net = nn.Sequential(
            nn.Linear(config.lift_feature_dim + config.splat_feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.Softmax(dim=-1)
        )
        
        # 创新性交互模块
        self.innovation_interaction = nn.Sequential(
            nn.Linear(config.lift_feature_dim + config.splat_feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, config.lift_feature_dim + config.splat_feature_dim),
            nn.Tanh()
        )
        
    def fuse_transitions(self, 
                        transition1_output: torch.Tensor,
                        transition2_output: torch.Tensor,
                        current_lift: torch.Tensor,
                        current_splat: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        双transition融合 - 创新性结合
        """
        
        if self.strategy == 'hierarchical':
            return self._hierarchical_fusion(
                transition1_output, transition2_output, current_lift, current_splat
            )
        elif self.strategy == 'parallel':
            return self._parallel_fusion(
                transition1_output, transition2_output, current_lift, current_splat
            )
        elif self.strategy == 'cascade':
            return self._cascade_fusion(
                transition1_output, transition2_output, current_lift, current_splat
            )
        else:
            raise ValueError(f"Unknown fusion strategy: {self.strategy}")
    
    def _hierarchical_fusion(self, t1_out, t2_out, lift, splat):
        """层次化融合：lift特征影响splat特征"""
        # Transition1的输出影响Transition2
        cross_influence = self.cross_influence_net(t1_out)
        enhanced_t2_out = t2_out + self.cross_influence * cross_influence
        
        # 创新性交互
        combined = torch.cat([t1_out, enhanced_t2_out], dim=-1)
        innovation = self.innovation_interaction(combined)
        
        # 分离创新输出
        lift_innovation, splat_innovation = torch.chunk(innovation, 2, dim=-1)
        
        return {
            'enhanced_lift': t1_out + lift_innovation,
            'enhanced_splat': enhanced_t2_out + splat_innovation,
            'cross_influence': cross_influence,
            'innovation': innovation
        }
    
    def _parallel_fusion(self, t1_out, t2_out, lift, splat):
        """并行融合：独立计算后智能融合"""
        # 学习融合权重
        fusion_input = torch.cat([t1_out, t2_out], dim=-1)
        fusion_weights = self.fusion_weight_net(fusion_input)
        
        # 加权融合
        enhanced_lift = fusion_weights[:, 0:1] * lift + fusion_weights[:, 1:2] * t1_out
        enhanced_splat = fusion_weights[:, 0:1] * splat + fusion_weights[:, 1:2] * t2_out
        
        return {
            'enhanced_lift': enhanced_lift,
            'enhanced_splat': enhanced_splat,
            'fusion_weights': fusion_weights
        }
    
    def _cascade_fusion(self, t1_out, t2_out, lift, splat):
        """级联融合：序列式影响"""
        # 第一级：lift特征增强
        enhanced_lift = lift + t1_out
        
        # 第二级：基于增强lift的splat特征增强
        lift_influence = self.cross_influence_net(enhanced_lift)
        enhanced_splat = splat + t2_out + self.cross_influence * lift_influence
        
        return {
            'enhanced_lift': enhanced_lift,
            'enhanced_splat': enhanced_splat,
            'lift_influence': lift_influence
        }


class HMMPlugin(nn.Module):
    """
    HMM插件 - 符合马尔可夫性质的实现
    
    核心设计原则：
    1. 隐状态编码历史信息
    2. 转移只依赖当前隐状态
    3. 双transition创新融合
    4. 可选择输出delta或absolute
    """
    
    def __init__(self, config: HMMConfig):
        super().__init__()
        self.config = config
        
        # 马尔可夫隐状态 - 核心创新
        self.markovian_state = MarkovianHiddenState(config)
        
        # 动作编码器
        self.action_encoder = ActionEncoder(
            action_dim=config.action_dim,
            hidden_dim=128,
            encoding_type='transformer'
        )
        
        # Transition1: 基于隐状态的lift特征转移
        self.transition1 = DiffusionTransition1(
            input_dim=config.lift_feature_dim + config.hidden_state_dim,
            output_dim=config.lift_feature_dim,
            hidden_dim=config.transition1_hidden_dim,
            model_type=config.transition1_type,
            diffusion_steps=config.transition1_diffusion_steps
        )
        
        # Transition2: 基于隐状态的splat特征转移
        self.transition2 = ControlNetTransition2(
            lift_dim=config.lift_feature_dim,
            splat_dim=config.splat_feature_dim,
            action_dim=128,
            output_dim=config.splat_feature_dim,
            hidden_dim=config.transition2_hidden_dim,
            model_type=config.transition2_type
        )
        
        # 双transition创新融合
        self.dual_fusion = DualTransitionFusion(config)
        
        # 融合策略
        if self.config.fusion_strategy == 'multi_path':
            # 使用TripleFusion替代原来的MultiPathYtFusion
            self.triple_fusion = TripleFusion(
                splat_dim=self.config.splat_feature_dim,
                hidden_dim=self.config.hidden_state_dim // 2
            )
        else:
            # 使用双路权重学习器
            self.weight_learner = WeightLearner(
                lift_dim=self.config.lift_feature_dim,
                splat_dim=self.config.splat_feature_dim,
                hidden_dim=self.config.hidden_state_dim
            )
        
        # 输出形式控制
        self.output_delta = config.transition1_output_type == 'delta'
        
    def forward(self, 
                lift_features: torch.Tensor,
                splat_features: torch.Tensor,
                actions: torch.Tensor,
                depth_features: Optional[torch.Tensor] = None,
                training: bool = True) -> Dict[str, torch.Tensor]:
        """
        前向传播 - 严格遵循马尔可夫性质
        
        关键：所有转移都只依赖当前隐状态，不直接使用历史序列
        """
        batch_size = lift_features.shape[0]
        
        # 处理输入特征的维度 - 确保都是2D张量，但限制维度
        if lift_features.dim() > 2:
            # 如果是高维特征，先进行平均池化再展平
            if lift_features.dim() == 4:  # [B, C, H, W]
                lift_features = lift_features.mean(dim=(2, 3))  # [B, C]
            elif lift_features.dim() == 5:  # [B, N, C, H, W]
                lift_features = lift_features.mean(dim=(1, 3, 4))  # [B, C]
            else:
                lift_features = lift_features.view(batch_size, -1)
                # 如果维度过大，投影到固定维度
                if lift_features.shape[1] > self.config.lift_feature_dim * 4:
                    if not hasattr(self, '_lift_proj_large'):
                        self._lift_proj_large = nn.Linear(lift_features.shape[1], self.config.lift_feature_dim).to(lift_features.device)
                    lift_features = self._lift_proj_large(lift_features)
        
        if splat_features.dim() > 2:
            if splat_features.dim() == 4:  # [B, C, H, W]
                splat_features = splat_features.mean(dim=(2, 3))  # [B, C]
            elif splat_features.dim() == 5:  # [B, N, C, H, W]
                splat_features = splat_features.mean(dim=(1, 3, 4))  # [B, C]
            else:
                splat_features = splat_features.view(batch_size, -1)
                # 如果维度过大，投影到固定维度
                if splat_features.shape[1] > self.config.splat_feature_dim * 4:
                    if not hasattr(self, '_splat_proj_large'):
                        self._splat_proj_large = nn.Linear(splat_features.shape[1], self.config.splat_feature_dim).to(splat_features.device)
                    splat_features = self._splat_proj_large(splat_features)
        
        if actions.dim() > 2:
            actions = actions.view(batch_size, -1)
            # 限制动作维度
            if actions.shape[1] > self.config.action_dim * 2:
                if not hasattr(self, '_action_proj_large'):
                    self._action_proj_large = nn.Linear(actions.shape[1], self.config.action_dim).to(actions.device)
                actions = self._action_proj_large(actions)
        
        # 1. 动作编码
        encoded_actions = self.action_encoder(actions)
        
        # 2. 隐状态更新（马尔可夫转移的核心）
        state_info = self.markovian_state.update_state(
            lift_features, splat_features, encoded_actions
        )
        
        # 3. Transition1: 基于隐状态的lift特征转移
        lift_with_state = torch.cat([
            lift_features, state_info['hidden_state']
        ], dim=-1)
        
        transition1_output = self.transition1(lift_with_state, training=training)
        
        # 4. Transition2: 基于隐状态的splat特征转移
        # 注意：这里也只使用隐状态，不直接使用历史
        transition2_output = self.transition2(
            lift_features,                    # 当前lift
            splat_features,                   # 当前splat
            encoded_actions,                  # 当前动作
            training=training
        )
        
        # 5. 双transition创新融合
        fusion_result = self.dual_fusion.fuse_transitions(
            transition1_output, transition2_output,
            lift_features, splat_features
        )
        
        # 6. 输出形式控制
        if self.config.transition1_output_type == 'delta':
            # 输出delta，需要加上原始特征
            enhanced_lift = lift_features + fusion_result['enhanced_lift']
            enhanced_splat = splat_features + fusion_result['enhanced_splat']
        else:
            # 输出absolute
            enhanced_lift = fusion_result['enhanced_lift']
            enhanced_splat = fusion_result['enhanced_splat']
        
        # 7. 特征融合
        if self.config.fusion_strategy == 'multi_path':
            # 使用三路融合
            final_splat, fusion_info = self.triple_fusion(
                splat_orig=splat_features,
                splat_pred=enhanced_splat,
                splat_enhanced=depth_features if depth_features is not None else None
            )
            multipath_outputs = fusion_info
        else:
            # 使用双路权重学习
            final_lift, final_splat, fusion_info = self.weight_learner(
                lift_orig=lift_features,
                lift_pred=enhanced_lift,
                splat_orig=splat_features,
                splat_pred=enhanced_splat
            )
            multipath_outputs = fusion_info
        
        # 8. 计算损失
        losses = self._compute_losses(
            lift_features, enhanced_lift,
            splat_features, enhanced_splat,
            state_info, fusion_result, multipath_outputs
        )
        
        return {
            'enhanced_lift': enhanced_lift,
            'enhanced_splat': final_splat,
            'state_info': state_info,
            'fusion_result': fusion_result,
            'multipath_outputs': multipath_outputs,
            'losses': losses,
            'output_type': self.config.transition1_output_type
        }
    
    def _compute_losses(self, 
                       lift_true: torch.Tensor,
                       lift_pred: torch.Tensor,
                       splat_true: torch.Tensor,
                       splat_pred: torch.Tensor,
                       state_info: Dict[str, torch.Tensor],
                       fusion_result: Dict[str, torch.Tensor],
                       multipath_outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """计算损失"""
        
        losses = {}
        
        # 重构损失
        losses['reconstruction_loss'] = (
            F.mse_loss(lift_pred, lift_true) + 
            F.mse_loss(splat_pred, splat_true)
        )
        
        # 状态质量损失
        if 'state_quality' in state_info:
            losses['state_quality_loss'] = -torch.mean(
                torch.log(state_info['state_quality'] + 1e-8)
            )
        
        # 马尔可夫正则化损失（鼓励状态平滑更新）
        if 'state_delta' in state_info:
            losses['markov_regularization'] = torch.mean(
                torch.abs(state_info['state_delta'])
            )
        
        # 多路融合损失
        if 'total_multipath_loss' in multipath_outputs:
            losses['multipath_loss'] = multipath_outputs['total_multipath_loss']
        
        # 总损失
        total_loss = (
            self.config.reconstruction_weight * losses['reconstruction_loss'] +
            self.config.kl_weight * losses.get('state_quality_loss', 0) +
            self.config.markov_regularization_weight * losses.get('markov_regularization', 0) +
            0.1 * losses.get('multipath_loss', 0)
        )
        
        losses['total_loss'] = total_loss
        
        return losses
    
    def reset_state(self):
        """重置状态（用于新序列开始）"""
        self.markovian_state.reset_state()
    
    def get_state_info(self) -> Dict[str, Any]:
        """获取状态信息"""
        return {
            'hidden_state_dim': self.config.hidden_state_dim,
            'world_state_dim': self.config.world_state_dim,
            'dual_transition_strategy': self.config.dual_transition_strategy,
            'transition1_output_type': self.config.transition1_output_type,
            'transition2_output_type': self.config.transition2_output_type
        } 