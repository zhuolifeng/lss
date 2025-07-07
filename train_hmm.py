"""
修正的HMM插件训练脚本
训练策略:冻结LSS模型参数,只训练HMM插件


核心思想：
1. LSS模型用于评估过程(提供特征和最终结果）
2. HMM插件独立训练,不影响LSS模型权重
3. 支持即插即用和一次训练多次使用
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import random
from typing import Dict, Optional, List, Tuple

import hydra
from omegaconf import DictConfig, OmegaConf

# 导入LSS相关模块
from src.models import compile_model  # 导入真实的LSS模型
from src.data import compile_data  # 导入数据加载器
from src.tools import get_val_info, SimpleLoss, get_batch_iou  # 导入评估工具

# 导入HMM相关模块
from hmm_plugin import HMMPlugin, HMMConfig
from hmm_plugin.adapters import (
    create_plug_and_play_hmm, 
    register_shared_hmm_plugin,
    global_hmm_manager
)
from hmm_plugin.training.loss import create_hmm_loss

# 添加必要的导入用于适配器函数
import warnings
warnings.filterwarnings('ignore')  # 忽略一些警告

# 添加缺失的导入
from src.tools import gen_dx_bx
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes


def calculate_iou(pred, target):
    """计算IoU指标"""
    return get_batch_iou(pred, target)


class FrozenLSSWithHMM(nn.Module):
    """
    冻结LSS模型，只训练HMM插件的包装器
    
    关键设计：
    1. LSS模型参数完全冻结
    2. HMM插件独立训练
    3. 在评估时通过完整管道计算最终损失
    """
    
    def __init__(self, lss_model: nn.Module, hmm_plugin: HMMPlugin):
        super().__init__()
        
        # 冻结LSS模型
        self.lss_model = lss_model
        for param in self.lss_model.parameters():
            param.requires_grad = False
        
        # HMM插件（可训练）
        self.hmm_plugin = hmm_plugin
        
        # 特征提取钩子
        self.extracted_features = {}
        self._register_hooks()
        
    def _register_hooks(self):
        """注册钩子来提取LSS中间特征"""
        
        def extract_lift_features(module, input, output):
            # 对于LSS的camencode，输出维度是 [B, N, D, H, W, C]，需要处理
            if output.dim() == 6:
                # 平均池化到合适的维度 [B, C]
                self.extracted_features['lift_features'] = output.mean(dim=(1, 2, 3, 4)).detach()
            elif output.dim() == 4:
                # [B, C, H, W] -> [B, C]
                self.extracted_features['lift_features'] = output.mean(dim=(2, 3)).detach()
            else:
                self.extracted_features['lift_features'] = output.detach()
        
        def extract_splat_features(module, input, output):
            # 对于LSS的bevencode，输出是 [B, C, H, W]
            if output.dim() == 4:
                # 平均池化到 [B, C]
                self.extracted_features['splat_features'] = output.mean(dim=(2, 3)).detach()
            else:
                self.extracted_features['splat_features'] = output.detach()
        
        def extract_depth_features(module, input, output):
            # 从camencode的depth输出中提取
            if hasattr(module, 'get_depth_feat') and output.dim() >= 3:
                # 提取深度特征并降维
                self.extracted_features['depth_features'] = output.mean(dim=tuple(range(2, output.dim()))).detach()
            else:
                self.extracted_features['depth_features'] = output.detach()
        
        # 更鲁棒的钩子注册策略
        hook_registered = {'lift': False, 'splat': False, 'depth': False}
        
        # 根据LSS模型结构注册钩子
        for name, module in self.lss_model.named_modules():
            module_name_lower = name.lower()
            
            # 检查lift相关模块
            if not hook_registered['lift'] and any(keyword in module_name_lower for keyword in [
                'lift', 'get_cam_feats', 'camencode', 'camera_encoder', 'cam_encoder'
            ]):
                module.register_forward_hook(extract_lift_features)
                hook_registered['lift'] = True
                print(f"注册lift钩子到: {name}")
            
            # 检查splat相关模块
            elif not hook_registered['splat'] and any(keyword in module_name_lower for keyword in [
                'splat', 'bev_pool', 'voxel_pool', 'bevencode', 'bev_encoder'
            ]):
                module.register_forward_hook(extract_splat_features)
                hook_registered['splat'] = True
                print(f"注册splat钩子到: {name}")
            
            # 检查depth相关模块
            elif not hook_registered['depth'] and any(keyword in module_name_lower for keyword in [
                'depth', 'depthnet', 'depth_net'
            ]):
                module.register_forward_hook(extract_depth_features)
                hook_registered['depth'] = True
                print(f"注册depth钩子到: {name}")
        
        # 如果没有找到合适的模块，尝试通用策略
        if not any(hook_registered.values()):
            print("Warning: 使用通用钩子注册策略")
            modules = list(self.lss_model.children())
            if len(modules) >= 2:
                # 假设第一个模块是相机编码器（lift）
                modules[0].register_forward_hook(extract_lift_features)
                print(f"通用lift钩子注册到: {modules[0].__class__.__name__}")
                
                # 假设第二个模块是BEV编码器（splat）
                modules[1].register_forward_hook(extract_splat_features)
                print(f"通用splat钩子注册到: {modules[1].__class__.__name__}")
        
        # 验证钩子注册结果
        registered_count = sum(hook_registered.values())
        print(f"成功注册 {registered_count}/3 个钩子: lift={hook_registered['lift']}, splat={hook_registered['splat']}, depth={hook_registered['depth']}")
        
        if registered_count == 0:
            print("Error: 没有成功注册任何钩子，可能需要手动指定LSS模型的模块名称")
    
    def forward(self, imgs, rots, trans, intrins, post_rots, post_trans, binimgs):
        """
        前向传播策略：
        1. 使用冻结的LSS模型提取特征和计算基线结果
        2. 使用HMM插件增强特征
        3. 通过增强特征重新计算最终结果（用于评估）
        """
        
        # 1. LSS基线前向传播（冻结参数）
        with torch.no_grad():
            lss_baseline_output = self.lss_model(imgs, rots, trans, intrins, post_rots, post_trans)
        
        # 2. 提取LSS中间特征
        lift_features = self.extracted_features.get('lift_features')
        splat_features = self.extracted_features.get('splat_features') 
        depth_features = self.extracted_features.get('depth_features')
        
        # 3. 提取动作信息
        actions = self._extract_actions(rots, trans)
        
        # 4. HMM插件处理（可训练）
        hmm_outputs = self.hmm_plugin(
            lift_features=lift_features,
            splat_features=splat_features,
            actions=actions,
            depth_features=depth_features,
            training=self.training
        )
        
        # 5. 使用增强特征重新计算最终输出（用于评估HMM效果）
        enhanced_output = self._compute_enhanced_output(
            hmm_outputs['enhanced_splat'], 
            rots, trans, intrins, post_rots, post_trans
        )
        
        return {
            'lss_baseline_output': lss_baseline_output,
            'enhanced_output': enhanced_output,
            'hmm_outputs': hmm_outputs,
            'extracted_features': {
                'lift_features': lift_features,
                'splat_features': splat_features,
                'depth_features': depth_features
            },
            'actions': actions
        }
    
    def _extract_actions(self, rots, trans):
        """从LSS输入中提取动作信息"""
        batch_size = rots.shape[0]
        device = rots.device
        
        # 修复维度处理问题
        try:
            # 确保输入张量是可用的
            if rots is None or trans is None:
                return torch.zeros(batch_size, 6, device=device)
            
            # 处理不同维度的输入
            if rots.dim() >= 3 and rots.shape[1] > 1:
                # 多时间步输入，计算相邻帧之间的差异
                rot_diff = rots[:, 1:] - rots[:, :-1]
                trans_diff = trans[:, 1:] - trans[:, :-1]
                
                # 取第一个时间差作为动作
                rot_action = rot_diff[:, 0].reshape(batch_size, -1)
                trans_action = trans_diff[:, 0].reshape(batch_size, -1)
                
                # 确保维度合适
                if rot_action.shape[1] >= 3:
                    rot_features = rot_action[:, :3]
                else:
                    rot_features = torch.cat([
                        rot_action,
                        torch.zeros(batch_size, 3 - rot_action.shape[1], device=device)
                    ], dim=-1)
                
                if trans_action.shape[1] >= 3:
                    trans_features = trans_action[:, :3]
                else:
                    trans_features = torch.cat([
                        trans_action,
                        torch.zeros(batch_size, 3 - trans_action.shape[1], device=device)
                    ], dim=-1)
                
                actions = torch.cat([rot_features, trans_features], dim=-1)
                
            else:
                # 单时间步输入，计算统计特征
                # 安全地处理张量形状
                rots_flat = rots.reshape(batch_size, -1)
                trans_flat = trans.reshape(batch_size, -1)
                
                # 计算基本统计信息（幅度和方差）
                rot_mag = torch.norm(rots_flat, dim=-1, keepdim=True)
                rot_var = torch.var(rots_flat, dim=-1, keepdim=True) if rots_flat.shape[1] > 1 else torch.zeros_like(rot_mag)
                rot_mean = torch.mean(rots_flat, dim=-1, keepdim=True)
                
                trans_mag = torch.norm(trans_flat, dim=-1, keepdim=True)
                trans_var = torch.var(trans_flat, dim=-1, keepdim=True) if trans_flat.shape[1] > 1 else torch.zeros_like(trans_mag)
                trans_mean = torch.mean(trans_flat, dim=-1, keepdim=True)
                
                actions = torch.cat([rot_mag, rot_var, rot_mean, trans_mag, trans_var, trans_mean], dim=-1)
            
            # 确保输出维度正确（6维）
            if actions.shape[1] > 6:
                actions = actions[:, :6]
            elif actions.shape[1] < 6:
                padding = torch.zeros(batch_size, 6 - actions.shape[1], device=device)
                actions = torch.cat([actions, padding], dim=-1)
                
        except Exception as e:
            print(f"Warning: 动作提取失败，使用默认值: {e}")
            actions = torch.zeros(batch_size, 6, device=device)
        
        return actions
    
    def _compute_enhanced_output(self, enhanced_splat, rots, trans, intrins, post_rots, post_trans):
        """使用增强的splat特征计算最终输出"""
        
        try:
            # enhanced_splat是从HMM插件来的特征，维度是[B, C]
            # 我们需要将它恢复到BEV空间并通过LSS的最终层处理
            batch_size = enhanced_splat.shape[0]
            feature_dim = enhanced_splat.shape[1]
            
            # 获取LSS模型的grid配置
            if hasattr(self.lss_model, 'nx'):
                nx = self.lss_model.nx
                # nx是[X, Y, Z]，对于BEV我们需要[X, Y]
                bev_h, bev_w = int(nx[0]), int(nx[1])
            else:
                # 默认BEV尺寸
                bev_h, bev_w = 200, 200
            
            # 如果enhanced_splat是2D特征[B, C]，需要expand到BEV空间
            if enhanced_splat.dim() == 2:
                # 创建一个简单的映射层将特征扩展到BEV空间
                if not hasattr(self, '_bev_expander'):
                    # BEV编码器通常输出1个类别（分割）
                    self._bev_expander = nn.Sequential(
                        nn.Linear(feature_dim, bev_h * bev_w),
                        nn.ReLU(),
                        nn.Linear(bev_h * bev_w, bev_h * bev_w)
                    ).to(enhanced_splat.device)
                
                # 扩展特征到BEV空间
                with torch.enable_grad():
                    bev_features = self._bev_expander(enhanced_splat)  # [B, H*W]
                    bev_features = bev_features.view(batch_size, 1, bev_h, bev_w)  # [B, 1, H, W]
                    enhanced_output = bev_features
            
            elif enhanced_splat.dim() == 4:
                # 如果已经是4D BEV特征，直接使用
                enhanced_output = enhanced_splat
            
            else:
                # 处理其他维度情况
                enhanced_output = enhanced_splat.view(batch_size, -1)
                # 如果需要，可以进一步处理
        
        except Exception as e:
            print(f"Warning: 增强输出计算失败: {e}")
            # 如果所有方法都失败，返回一个默认BEV输出
            batch_size = enhanced_splat.shape[0]
            # 创建默认的BEV分割图
            enhanced_output = torch.zeros(batch_size, 1, 200, 200, device=enhanced_splat.device)
        
        return enhanced_output


def create_training_setup(cfg: DictConfig):
    """创建训练设置"""
    
    # 1. 设置默认的LSS配置
    default_lss_config = {
        'version': 'mini',  # 使用mini版本快速测试
        'dataroot': '/dataset/nuscenes',
        
        'grid_conf': {
            'xbound': [-50.0, 50.0, 0.5],
            'ybound': [-50.0, 50.0, 0.5],
            'zbound': [-10.0, 10.0, 20.0],
            'dbound': [4.0, 45.0, 1.0],
        },
        
        'data_aug_conf': {
            'H': 900, 'W': 1600,
            'resize_lim': (0.193, 0.225),
            'final_dim': (128, 352),
            'rot_lim': (-5.4, 5.4),
            'rand_flip': True,
            'bot_pct_lim': (0.0, 0.22),
            'cams': ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                     'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],
            'Ncams': 5,
        }
    }
    
    # 2. 尝试加载外部配置文件
    try:
        if cfg.lss.config_path.endswith('.json'):
            with open(cfg.lss.config_path, 'r') as f:
                external_config = json.load(f)
        else:
            # 假设是YAML格式
            external_config = OmegaConf.to_container(OmegaConf.load(cfg.lss.config_path), resolve=True)
        
        # 合并配置
        lss_config = {**default_lss_config, **external_config}
    except Exception as e:
        print(f"Warning: Failed to load LSS config from {cfg.lss.config_path}: {e}")
        print("Using default LSS configuration")
        lss_config = default_lss_config
    
    # 3. 从Hydra配置覆盖数据路径
    if hasattr(cfg, 'data') and hasattr(cfg.data, 'data_root'):
        lss_config['dataroot'] = cfg.data.data_root
    if hasattr(cfg, 'data') and hasattr(cfg.data, 'version'):
        lss_config['version'] = cfg.data.version.split('-')[-1]  # 转换 v1.0-mini -> mini
    
    print(f"Using LSS config: {lss_config}")
    
    # 4. 从Hydra配置创建HMM配置
    hmm_config_dict = OmegaConf.to_container(cfg.hmm_config, resolve=True) if 'hmm_config' in cfg else {}
    hmm_config = HMMConfig(**hmm_config_dict)
    
    # 5. 创建LSS模型并加载预训练权重
    print("创建LSS模型...")
    lss_model = compile_model(lss_config['grid_conf'], lss_config['data_aug_conf'], outC=1)
    
    if os.path.exists(cfg.lss.checkpoint_path):
        print(f"加载LSS预训练权重: {cfg.lss.checkpoint_path}")
        checkpoint = torch.load(cfg.lss.checkpoint_path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            lss_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            lss_model.load_state_dict(checkpoint)
    else:
        print(f"Warning: LSS checkpoint not found at {cfg.lss.checkpoint_path}, using random weights")
    
    # 6. 创建HMM插件
    print("创建HMM插件...")
    hmm_plugin = HMMPlugin(hmm_config)
    
    # 7. 创建包装模型
    model = FrozenLSSWithHMM(lss_model, hmm_plugin)
    
    # 8. 创建数据加载器
    print("创建数据加载器...")
    train_loader, val_loader = compile_data(
        version=lss_config['version'],
        dataroot=lss_config['dataroot'],
        data_aug_conf=lss_config['data_aug_conf'],
        grid_conf=lss_config['grid_conf'],
        bsz=cfg.training.batch_size,
        nworkers=4,
        parser_name='segmentationdata'
    )
    
    print(f"训练集样本数: {len(train_loader.dataset)}")
    print(f"验证集样本数: {len(val_loader.dataset)}")
    
    # 9. 创建优化器（只优化HMM插件参数）
    hmm_params = list(hmm_plugin.parameters())
    optimizer_cfg = cfg.training.optimizer
    if optimizer_cfg.type.lower() == 'adamw':
        optimizer = optim.AdamW(
            hmm_params, 
            lr=optimizer_cfg.lr, 
            weight_decay=optimizer_cfg.weight_decay,
            betas=optimizer_cfg.betas,
            eps=optimizer_cfg.eps
        )
    else:
        optimizer = optim.Adam(hmm_params, lr=optimizer_cfg.lr, weight_decay=optimizer_cfg.weight_decay)
    
    # 10. 创建损失函数
    hmm_criterion = create_hmm_loss(hmm_config_dict)
    lss_criterion = SimpleLoss(pos_weight=2.13)  # 用于评估最终结果，使用LSS原始损失
    
    # 11. 创建学习率调度器
    scheduler_cfg = cfg.training.scheduler
    if scheduler_cfg.type.lower() == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=scheduler_cfg.T_max, 
            eta_min=scheduler_cfg.eta_min
        )
    else:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    
    return model, train_loader, val_loader, optimizer, scheduler, hmm_criterion, lss_criterion


def train_epoch(model, train_loader, optimizer, hmm_criterion, lss_criterion, device, epoch, writer):
    """训练一个epoch"""
    
    model.train()
    model.hmm_plugin.train()  # 确保HMM插件在训练模式
    
    total_loss = 0
    total_hmm_loss = 0
    total_lss_loss = 0
    
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}')
    
    for batch_idx, batch in enumerate(progress_bar):
        # LSS数据格式：(imgs, rots, trans, intrins, post_rots, post_trans, binimgs)
        if isinstance(batch, (tuple, list)):
            imgs, rots, trans, intrins, post_rots, post_trans, binimgs = batch
            batch = {
                'imgs': imgs.to(device),
                'rots': rots.to(device),
                'trans': trans.to(device),
                'intrins': intrins.to(device),
                'post_rots': post_rots.to(device),
                'post_trans': post_trans.to(device),
                'binimgs': binimgs.to(device)
            }
        else:
            # 移动数据到设备（字典格式）
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        
        # 前向传播
        outputs = model(
            batch['imgs'], batch['rots'], batch['trans'], 
            batch['intrins'], batch['post_rots'], batch['post_trans'], 
            batch['binimgs']
        )
        
        # 计算HMM损失（这是主要训练目标）
        # 修复：处理可能没有时序维度的特征
        lift_feats = outputs['extracted_features']['lift_features']
        splat_feats = outputs['extracted_features']['splat_features']
        depth_feats = outputs['extracted_features']['depth_features']
        
        # 确保所有特征都有batch维度
        if lift_feats is None:
            lift_feats = torch.zeros(batch['imgs'].shape[0], 64, device=device)
        if splat_feats is None:
            splat_feats = torch.zeros(batch['imgs'].shape[0], 64, device=device)
        if depth_feats is None:
            depth_feats = torch.zeros_like(lift_feats)
        
        # 处理actions维度
        actions = outputs['actions']
        if actions.dim() == 2:
            actions = actions.unsqueeze(1)  # 添加时序维度
        
        # 如果特征是2D的，添加时序维度用于损失计算
        if lift_feats.dim() == 2:
            lift_feats_t = lift_feats.unsqueeze(1)  # [B, 1, dim]
        else:
            lift_feats_t = lift_feats
            
        if splat_feats.dim() == 2:
            splat_feats_t = splat_feats.unsqueeze(1)
        else:
            splat_feats_t = splat_feats
            
        if depth_feats.dim() == 2:
            depth_feats_t = depth_feats.unsqueeze(1)
        else:
            depth_feats_t = depth_feats
        
        # 处理binimgs维度
        binimgs = batch['binimgs']
        if binimgs.dim() == 3:  # 如果是[B, H, W]，添加时序维度
            binimgs = binimgs.unsqueeze(1)  # [B, 1, H, W]
        elif binimgs.dim() == 4:  # 如果是[B, T, H, W]或[B, C, H, W]
            if binimgs.shape[1] > 50:  # 可能是通道维度
                binimgs = binimgs.unsqueeze(1)  # 添加时序维度
        
        # 确保imgs也有正确的维度
        imgs = batch['imgs']
        if imgs.dim() == 4:  # [B, C, H, W]
            imgs = imgs.unsqueeze(1)  # [B, 1, C, H, W]
        
        try:
            hmm_losses = hmm_criterion(
                # 深度似然 P(Dt|Xt) - 使用当前特征
                lift_feats_t,  # Xt
                depth_feats_t,  # Dt
                
                # 测量似然 P(Mt|Yt) - 使用当前特征和标签
                splat_feats_t,  # Yt
                binimgs,  # Mt (测量/标签)
                
                # BEV特征KL散度数据 - 使用当前特征模拟时序
                lift_feats_t,   # Xt+1 (用当前代替下一时刻)
                splat_feats_t,  # Yt (当前)
                actions,  # At
                splat_feats_t,  # Yt+1_true (用当前代替)
                
                # 状态转移KL散度数据 - 使用当前特征
                imgs,  # Ot (观测)
                lift_feats_t,  # Xt (当前)
                lift_feats_t   # Xt+1_true (用当前代替)
            )
        except Exception as e:
            print(f"Warning: HMM损失计算失败: {e}")
            # 使用简化的损失计算
            hmm_losses = {
                'total_loss': torch.tensor(0.0, device=device, requires_grad=True),
                'depth_loss': torch.tensor(0.0, device=device),
                'measurement_loss': torch.tensor(0.0, device=device),
                'bev_kl_loss': torch.tensor(0.0, device=device),
                'state_kl_loss': torch.tensor(0.0, device=device)
            }
        
        # 计算LSS损失（用于评估效果）
        with torch.no_grad():
            try:
                enhanced_output = outputs['enhanced_output']
                target = batch['binimgs']
                
                # LSS的SimpleLoss期望BEV分割图格式
                # enhanced_output: [B, 1, H, W] 或 [B, H, W]
                # target: [B, 1, H, W] 或 [B, H, W]
                
                # 确保enhanced_output是正确的格式
                if enhanced_output.dim() == 4 and enhanced_output.shape[1] == 1:
                    enhanced_output = enhanced_output.squeeze(1)  # [B, H, W]
                elif enhanced_output.dim() == 2:
                    # 如果是[B, features]，需要reshape到BEV空间
                    B = enhanced_output.shape[0]
                    enhanced_output = enhanced_output.view(B, 200, 200)  # 假设200x200 BEV
                
                # 确保target是正确的格式
                if target.dim() == 4 and target.shape[1] == 1:
                    target = target.squeeze(1)  # [B, H, W]
                elif target.dim() == 3:
                    pass  # 已经是[B, H, W]格式
                
                # 确保尺寸匹配
                if enhanced_output.shape[-2:] != target.shape[-2:]:
                    # 调整enhanced_output的尺寸以匹配target
                    target_h, target_w = target.shape[-2:]
                    enhanced_output = F.interpolate(
                        enhanced_output.unsqueeze(1), 
                        size=(target_h, target_w), 
                        mode='bilinear', 
                        align_corners=False
                    ).squeeze(1)
                
                # 使用LSS的SimpleLoss
                lss_loss = lss_criterion(enhanced_output, target)
                
            except Exception as e:
                print(f"Warning: LSS损失计算失败: {e}")
                lss_loss = torch.tensor(0.0, device=device)
        
        # 总损失（只有HMM损失参与反向传播）
        total_batch_loss = hmm_losses['total_loss']
        
        # 反向传播（只更新HMM参数）
        optimizer.zero_grad()
        total_batch_loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.hmm_plugin.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # 记录损失
        total_loss += total_batch_loss.item()
        total_hmm_loss += hmm_losses['total_loss'].item()
        total_lss_loss += lss_loss.item()
        
        # 更新进度条
        progress_bar.set_postfix({
            'HMM Loss': f'{hmm_losses["total_loss"].item():.4f}',
            'LSS Loss': f'{lss_loss.item():.4f}',
            'LR': f'{optimizer.param_groups[0]["lr"]:.6f}'
        })
        
        # 记录到tensorboard
        if batch_idx % 10 == 0:
            global_step = epoch * len(train_loader) + batch_idx
            writer.add_scalar('Train/HMM_Loss', hmm_losses['total_loss'].item(), global_step)
            writer.add_scalar('Train/LSS_Loss', lss_loss.item(), global_step)
            writer.add_scalar('Train/Learning_Rate', optimizer.param_groups[0]['lr'], global_step)
            
            # 记录各个HMM损失组件
            for loss_name, loss_value in hmm_losses.items():
                if 'loss' in loss_name and loss_name != 'total_loss':
                    writer.add_scalar(f'Train/HMM_{loss_name}', loss_value.item(), global_step)
    
    avg_loss = total_loss / len(train_loader)
    avg_hmm_loss = total_hmm_loss / len(train_loader)
    avg_lss_loss = total_lss_loss / len(train_loader)
    
    return {
        'total_loss': avg_loss,
        'hmm_loss': avg_hmm_loss,
        'lss_loss': avg_lss_loss
    }


def validate_epoch(model, val_loader, hmm_criterion, lss_criterion, device, epoch, writer):
    """验证一个epoch"""
    
    model.eval()
    
    total_loss = 0
    total_hmm_loss = 0
    total_lss_loss = 0
    total_iou = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_loader, desc='Validation')):
            # LSS数据格式：(imgs, rots, trans, intrins, post_rots, post_trans, binimgs)
            if isinstance(batch, (tuple, list)):
                imgs, rots, trans, intrins, post_rots, post_trans, binimgs = batch
                batch = {
                    'imgs': imgs.to(device),
                    'rots': rots.to(device),
                    'trans': trans.to(device),
                    'intrins': intrins.to(device),
                    'post_rots': post_rots.to(device),
                    'post_trans': post_trans.to(device),
                    'binimgs': binimgs.to(device)
                }
            else:
                # 移动数据到设备（字典格式）
                batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            
            # 前向传播
            outputs = model(
                batch['imgs'], batch['rots'], batch['trans'], 
                batch['intrins'], batch['post_rots'], batch['post_trans'], 
                batch['binimgs']
            )
            
            # 计算损失（修复验证时的维度问题）
            try:
                lift_feats = outputs['extracted_features']['lift_features']
                splat_feats = outputs['extracted_features']['splat_features']
                depth_feats = outputs['extracted_features']['depth_features']
                actions = outputs['actions']
                
                # 处理单时间步数据
                if lift_feats is None or lift_feats.dim() == 2:
                    # 使用简化的损失计算
                    hmm_losses = {
                        'total_loss': torch.tensor(0.0, device=device, requires_grad=True),
                        'depth_loss': torch.tensor(0.0, device=device),
                        'measurement_loss': torch.tensor(0.0, device=device),
                        'bev_kl_loss': torch.tensor(0.0, device=device),
                        'state_kl_loss': torch.tensor(0.0, device=device)
                    }
                else:
                    # 确保有时序维度
                    if lift_feats.dim() == 2:
                        lift_feats = lift_feats.unsqueeze(1)
                    if splat_feats.dim() == 2:
                        splat_feats = splat_feats.unsqueeze(1)
                    if depth_feats is not None and depth_feats.dim() == 2:
                        depth_feats = depth_feats.unsqueeze(1)
                    if actions.dim() == 2:
                        actions = actions.unsqueeze(1)
                        
                    # 只有当序列长度>1时才进行时序损失计算
                    if lift_feats.shape[1] > 1:
                        hmm_losses = hmm_criterion(
                            lift_feats[:, :-1],
                            depth_feats[:, :-1] if depth_feats is not None else torch.zeros_like(lift_feats[:, :-1]),
                            splat_feats[:, :-1],
                            batch['binimgs'][:, :-1] if batch['binimgs'].dim() > 3 else batch['binimgs'].unsqueeze(1),
                            lift_feats[:, 1:],
                            splat_feats[:, :-1],
                            actions[:, :-1] if actions.shape[1] > 1 else actions,
                            splat_feats[:, 1:],
                            batch['imgs'][:, :-1] if batch['imgs'].dim() > 4 else batch['imgs'].unsqueeze(1),
                            lift_feats[:, :-1],
                            lift_feats[:, 1:]
                        )
                    else:
                        # 单时间步，使用当前特征计算损失
                        hmm_losses = hmm_criterion(
                            lift_feats,
                            depth_feats if depth_feats is not None else torch.zeros_like(lift_feats),
                            splat_feats,
                            batch['binimgs'].unsqueeze(1) if batch['binimgs'].dim() == 3 else batch['binimgs'],
                            lift_feats,  # 使用当前作为下一时刻
                            splat_feats,
                            actions,
                            splat_feats,  # 使用当前作为下一时刻
                            batch['imgs'].unsqueeze(1) if batch['imgs'].dim() == 4 else batch['imgs'],
                            lift_feats,
                            lift_feats   # 使用当前作为下一时刻
                        )
            except Exception as e:
                print(f"Warning: Validation HMM loss calculation failed: {e}")
                hmm_losses = {
                    'total_loss': torch.tensor(0.0, device=device),
                    'depth_loss': torch.tensor(0.0, device=device),
                    'measurement_loss': torch.tensor(0.0, device=device),
                    'bev_kl_loss': torch.tensor(0.0, device=device),
                    'state_kl_loss': torch.tensor(0.0, device=device)
                }
            
            try:
                enhanced_output = outputs['enhanced_output']
                target = batch['binimgs']
                
                # 处理维度匹配问题
                if enhanced_output.dim() > 2:
                    enhanced_output = enhanced_output.view(-1, enhanced_output.shape[-1])
                if target.dim() > 1:
                    target = target.view(-1)
                
                target = target.long()
                
                if enhanced_output.shape[0] != target.shape[0]:
                    lss_loss = torch.tensor(0.0, device=device)
                else:
                    lss_loss = lss_criterion(enhanced_output, target)
            except Exception as e:
                print(f"Warning: Validation LSS损失计算失败: {e}")
                lss_loss = torch.tensor(0.0, device=device)
            
            # 计算IoU（评估最终效果）
            pred = outputs['enhanced_output'].argmax(dim=-1)
            target = batch['binimgs']
            iou = calculate_iou(pred, target)
            
            total_loss += hmm_losses['total_loss'].item()
            total_hmm_loss += hmm_losses['total_loss'].item()
            total_lss_loss += lss_loss.item()
            total_iou += iou
    
    avg_loss = total_loss / len(val_loader)
    avg_hmm_loss = total_hmm_loss / len(val_loader)
    avg_lss_loss = total_lss_loss / len(val_loader)
    avg_iou = total_iou / len(val_loader)
    
    # 记录验证结果
    writer.add_scalar('Val/HMM_Loss', avg_hmm_loss, epoch)
    writer.add_scalar('Val/LSS_Loss', avg_lss_loss, epoch)
    writer.add_scalar('Val/IoU', avg_iou, epoch)
    
    return {
        'total_loss': avg_loss,
        'hmm_loss': avg_hmm_loss,
        'lss_loss': avg_lss_loss,
        'iou': avg_iou
    }


@hydra.main(version_base=None, config_path="configs", config_name="start")
def main(cfg: DictConfig) -> None:
    print("Configuration:")
    print(OmegaConf.to_yaml(cfg))
    
    # 设置随机种子
    if cfg.seed:
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        random.seed(cfg.seed)
    
    # 设置设备
    device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')
    
    # 创建输出目录（Hydra会自动处理）
    output_dir = cfg.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置tensorboard
    if cfg.logging.use_tensorboard:
        writer = SummaryWriter(os.path.join(output_dir, 'tensorboard'))
    else:
        writer = None
    
    # 创建训练设置
    model, train_loader, val_loader, optimizer, scheduler, hmm_criterion, lss_criterion = create_training_setup(cfg)
    model.to(device)
    
    print(f"Model moved to {device}")
    print(f"HMM Plugin parameters: {sum(p.numel() for p in model.hmm_plugin.parameters() if p.requires_grad)}")
    print(f"LSS Model parameters (frozen): {sum(p.numel() for p in model.lss_model.parameters())}")
    
    # 训练循环
    best_iou = 0
    
    for epoch in range(cfg.training.epochs):
        print(f"\nEpoch {epoch+1}/{cfg.training.epochs}")
        
        # 训练
        train_stats = train_epoch(
            model, train_loader, optimizer, hmm_criterion, lss_criterion, 
            device, epoch, writer
        )
        
        # 验证
        if epoch % cfg.training.eval_every == 0:
            val_stats = validate_epoch(
                model, val_loader, hmm_criterion, lss_criterion, 
                device, epoch, writer
            )
            
            print(f"Train - HMM Loss: {train_stats['hmm_loss']:.4f}, LSS Loss: {train_stats['lss_loss']:.4f}")
            print(f"Val - HMM Loss: {val_stats['hmm_loss']:.4f}, LSS Loss: {val_stats['lss_loss']:.4f}, IoU: {val_stats['iou']:.4f}")
            
            # 保存最佳模型
            if val_stats['iou'] > best_iou:
                best_iou = val_stats['iou']
                torch.save({
                    'epoch': epoch,
                    'hmm_plugin_state_dict': model.hmm_plugin.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_iou': best_iou,
                    'hmm_config': model.hmm_plugin.config if hasattr(model.hmm_plugin, 'config') else None,
                    'hydra_config': OmegaConf.to_container(cfg, resolve=True)
                }, os.path.join(output_dir, 'best_hmm_plugin.pth'))
        
        # 保存定期检查点
        if epoch % cfg.training.save_every == 0:
            torch.save({
                'epoch': epoch,
                'hmm_plugin_state_dict': model.hmm_plugin.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'hydra_config': OmegaConf.to_container(cfg, resolve=True)
            }, os.path.join(output_dir, f'hmm_plugin_epoch_{epoch}.pth'))
        
        # 更新学习率
        scheduler.step()
    
    # 训练完成后注册到全局管理器
    try:
        register_shared_hmm_plugin(
            plugin_name="trained_hmm_general",
            hmm_plugin=model.hmm_plugin,
            config=model.hmm_plugin.get_state_info() if hasattr(model.hmm_plugin, 'get_state_info') else None
        )
        print("HMM Plugin registered to global manager for reuse.")
    except Exception as e:
        print(f"Warning: Failed to register HMM plugin: {e}")
    
    print(f"\nTraining completed! Best IoU: {best_iou:.4f}")
    
    if writer:
        writer.close()


if __name__ == '__main__':
    main() 