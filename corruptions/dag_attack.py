import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Callable, Dict, List, Tuple


class DAGBEVAttack:
    """
    Dense Adversary Generation (DAG) for BEV Perception
    基于Xie et al. ICCV 2017论文实现的BEV感知专用DAG攻击
    
    核心思想：在BEV网格的每个位置上优化损失函数，生成稠密的对抗扰动
    """
    
    def __init__(self, model):
        """
        初始化DAG攻击器
        
        Args:
            model: LSS模型或其他BEV感知模型
        """
        self.model = model
        
    def dag_bev_attack(self, images, rots, trans, intrins, post_rots, post_trans, binimgs,
                       epsilon=0.03, alpha=0.005, num_steps=20, 
                       attack_pattern='dense', target_density=0.8):
        """
        DAG BEV攻击：在BEV空间的所有位置上进行稠密对抗优化
        
        Args:
            images: 输入图像 (S, C, H, W)
            rots, trans, intrins, post_rots, post_trans: 相机参数
            binimgs: BEV ground truth (1, H_bev, W_bev)
            epsilon: 最大扰动幅度
            alpha: 每步扰动大小
            num_steps: 迭代步数
            attack_pattern: 攻击模式 'dense'(全图), 'vehicle_focus'(车辆重点), 'boundary'(边界重点)
            target_density: 目标稠密度 (0-1)
            
        Returns:
            对抗样本图像
        """
        S, C, H, W = images.shape
        print(f"[DAG-BEV攻击] 模式: {attack_pattern}, epsilon: {epsilon:.6f}, 目标密度: {target_density}")
        
        # 添加批次维度
        images_tensor = images.unsqueeze(0).clone().detach()
        target_bev = binimgs.unsqueeze(0).float()
        
        # 初始化随机扰动
        delta = (torch.rand_like(images_tensor) * 2 * epsilon - epsilon)
        delta = delta.detach()
        
        # 获取BEV网格信息
        with torch.no_grad():
            original_output = self.model(images_tensor, rots.unsqueeze(0), trans.unsqueeze(0),
                                       intrins.unsqueeze(0), post_rots.unsqueeze(0), post_trans.unsqueeze(0))
            bev_h, bev_w = original_output.shape[-2:]
        
        # 生成攻击掩码（决定在BEV的哪些区域进行攻击）
        attack_mask = self._generate_attack_mask(target_bev, attack_pattern, target_density)
        
        best_delta = delta.clone()
        best_dag_loss = float('-inf')
        
        print(f"[DAG-BEV] 开始在{bev_h}x{bev_w}的BEV网格上进行稠密攻击...")
        
        for step in range(num_steps):
            delta.requires_grad_(True)
            
            # 前向传播
            output = self.model(images_tensor + delta, rots.unsqueeze(0), trans.unsqueeze(0),
                              intrins.unsqueeze(0), post_rots.unsqueeze(0), post_trans.unsqueeze(0))
            
            pred = output.sigmoid()
            
            # DAG核心：在每个BEV位置计算损失
            dag_loss = self._compute_dense_bev_loss(pred, target_bev, attack_mask)
            
            # 几何一致性攻击（利用3D投影的脆弱性）
            geometry_loss = self._compute_geometry_consistency_loss(pred, target_bev)
            
            # 多尺度攻击（不同分辨率的BEV特征）
            multiscale_loss = self._compute_multiscale_attack_loss(pred, target_bev)
            
            # 时空一致性攻击（破坏相邻帧的一致性，虽然这里是单帧）
            temporal_loss = self._compute_temporal_consistency_loss(pred, target_bev)
            
            # 组合DAG损失
            total_dag_loss = (3.0 * dag_loss + 
                             2.0 * geometry_loss + 
                             1.5 * multiscale_loss + 
                             1.0 * temporal_loss)
            
            total_dag_loss.backward()
            
            if delta.grad is not None:
                # 自适应步长
                adaptive_alpha = alpha * (1.0 + step * 0.03)
                delta = delta + adaptive_alpha * delta.grad.sign()
                delta = torch.clamp(delta, -epsilon, epsilon)
                
                # 记录最佳结果
                current_loss = total_dag_loss.item()
                if current_loss > best_dag_loss:
                    best_dag_loss = current_loss
                    best_delta = delta.clone().detach()
            
            delta = delta.detach()
            self.model.zero_grad()
            
            if step % 5 == 0:
                print(f"[DAG-BEV] Step {step}: DAG损失={dag_loss.item():.4f}, "
                      f"几何损失={geometry_loss.item():.4f}, 总损失={total_dag_loss.item():.4f}")
        
        print(f"[DAG-BEV] 最佳DAG损失: {best_dag_loss:.6f}")
        
        # 生成最终对抗样本
        final_adv_images = images_tensor + best_delta
        final_adv_images = final_adv_images.squeeze(0)
        
        # 计算攻击效果统计
        self._compute_attack_statistics(images, final_adv_images, target_bev, attack_mask)
        
        return final_adv_images
    
    def _generate_attack_mask(self, target_bev, attack_pattern, target_density):
        """
        生成攻击掩码，决定在BEV的哪些区域进行稠密攻击
        """
        B, _, H, W = target_bev.shape
        device = target_bev.device
        
        if attack_pattern == 'dense':
            # 全图稠密攻击
            mask = torch.ones((B, 1, H, W), device=device)
            
        elif attack_pattern == 'vehicle_focus':
            # 聚焦车辆区域及其周围
            vehicle_mask = target_bev > 0.5
            # 扩展车辆区域
            kernel = torch.ones(1, 1, 5, 5, device=device)
            expanded_mask = F.conv2d(vehicle_mask.float(), kernel, padding=2) > 0
            mask = expanded_mask.float()
            
        elif attack_pattern == 'boundary':
            # 聚焦边界区域
            kernel = torch.tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], 
                                 dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
            boundary_mask = F.conv2d(target_bev, kernel, padding=1).abs()
            mask = (boundary_mask > 0.1).float()
            
        else:
            mask = torch.ones((B, 1, H, W), device=device)
        
        # 根据目标密度调整掩码
        if target_density < 1.0:
            random_mask = torch.rand_like(mask) < target_density
            mask = mask * random_mask.float()
        
        return mask
    
    def _compute_dense_bev_loss(self, pred, target, attack_mask):
        """
        计算稠密BEV损失 - DAG的核心
        在BEV的每个位置上计算损失，而不是全局平均
        """
        # 位置相关的损失权重
        position_weights = self._compute_position_weights(pred, target)
        
        # 像素级二元交叉熵
        pixel_bce = F.binary_cross_entropy(pred, target, reduction='none')
        
        # 像素级IoU损失
        intersection = pred * target
        union = pred + target - intersection
        pixel_iou_loss = 1.0 - intersection / (union + 1e-8)
        
        # 应用攻击掩码和位置权重
        weighted_bce = (pixel_bce * position_weights * attack_mask).sum() / (attack_mask.sum() + 1e-8)
        weighted_iou = (pixel_iou_loss * position_weights * attack_mask).sum() / (attack_mask.sum() + 1e-8)
        
        # 稠密性惩罚 - 鼓励在更多位置产生错误
        density_bonus = (attack_mask * (pred - target).abs()).mean()
        
        return weighted_bce + weighted_iou + 0.5 * density_bonus
    
    def _compute_position_weights(self, pred, target):
        """
        计算位置相关的权重，对重要区域（如车辆中心、边界）给予更高权重
        """
        B, C, H, W = pred.shape
        device = pred.device
        
        # 车辆中心权重
        vehicle_mask = target > 0.5
        center_weight = vehicle_mask.float() * 2.0
        
        # 边界权重
        boundary_kernel = torch.tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], 
                                      dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
        boundary_strength = F.conv2d(target, boundary_kernel, padding=1).abs()
        boundary_weight = (boundary_strength > 0.1).float() * 1.5
        
        # 距离权重（距离图像中心越近权重越高，模拟自车周围重要性）
        y_coords = torch.arange(H, device=device).float().unsqueeze(1).expand(H, W)
        x_coords = torch.arange(W, device=device).float().unsqueeze(0).expand(H, W)
        center_y, center_x = H // 2, W // 2
        distance = torch.sqrt((y_coords - center_y)**2 + (x_coords - center_x)**2)
        distance_weight = 1.0 / (1.0 + distance / max(H, W))
        distance_weight = distance_weight.unsqueeze(0).unsqueeze(0).expand(B, C, H, W)
        
        # 组合权重
        total_weight = 1.0 + center_weight + boundary_weight + distance_weight * 0.5
        
        return total_weight
    
    def _compute_geometry_consistency_loss(self, pred, target):
        """
        几何一致性攻击 - 利用3D到BEV投影的几何约束脆弱性
        """
        # 形状保持损失 - 破坏车辆的矩形形状
        # 使用形态学操作检测形状一致性
        eroded = F.max_pool2d(-pred, 3, stride=1, padding=1)
        eroded = -eroded
        dilated = F.max_pool2d(pred, 3, stride=1, padding=1)
        
        # 车辆应该有相对规整的形状，攻击这种一致性
        shape_consistency = F.mse_loss(eroded * target, pred * target, reduction='mean')
        
        # 连通性攻击 - 破坏车辆区域的连通性
        connectivity_loss = self._compute_connectivity_loss(pred, target)
        
        return shape_consistency + connectivity_loss
    
    def _compute_connectivity_loss(self, pred, target):
        """
        连通性损失 - 破坏车辆区域的空间连通性
        """
        # 8连通的卷积核
        connectivity_kernel = torch.tensor([
            [[1, 1, 1],
             [1, 0, 1], 
             [1, 1, 1]]
        ], dtype=torch.float32, device=pred.device).unsqueeze(0)
        
        # 计算每个像素的连通性
        neighbors = F.conv2d(pred, connectivity_kernel, padding=1)
        target_neighbors = F.conv2d(target, connectivity_kernel, padding=1)
        
        # 在真实车辆区域，鼓励预测不连通
        vehicle_mask = target > 0.5
        connectivity_loss = (neighbors * vehicle_mask).mean()
        
        return connectivity_loss
    
    def _compute_multiscale_attack_loss(self, pred, target):
        """
        多尺度攻击损失 - 在不同分辨率下攻击BEV特征
        """
        scales = [1, 2, 4]  # 不同的下采样比例
        multiscale_loss = 0.0
        
        for scale in scales:
            if scale > 1:
                # 下采样
                pred_scaled = F.avg_pool2d(pred, scale, stride=scale)
                target_scaled = F.avg_pool2d(target, scale, stride=scale)
            else:
                pred_scaled = pred
                target_scaled = target
            
            # 在每个尺度上计算损失
            scale_loss = F.binary_cross_entropy(pred_scaled, target_scaled, reduction='mean')
            multiscale_loss += scale_loss / len(scales)
        
        return multiscale_loss
    
    def _compute_temporal_consistency_loss(self, pred, target):
        """
        时空一致性损失 - 虽然这里是单帧，但可以攻击预测的时空平滑性
        """
        # 时间梯度损失 - 假设连续帧应该相似，攻击这种平滑性
        # 这里用空间梯度来近似时间变化
        grad_x = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        grad_y = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        
        target_grad_x = target[:, :, :, 1:] - target[:, :, :, :-1]
        target_grad_y = target[:, :, 1:, :] - target[:, :, :-1, :]
        
        # 鼓励预测梯度与真实梯度不一致
        temporal_loss = -F.mse_loss(grad_x, target_grad_x) - F.mse_loss(grad_y, target_grad_y)
        
        return temporal_loss
    
    def _compute_attack_statistics(self, original_images, adv_images, target_bev, attack_mask):
        """
        计算并打印攻击效果统计
        """
        # 计算扰动统计
        diff = (adv_images - original_images).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        
        # 计算攻击覆盖率
        attack_coverage = attack_mask.sum().item() / attack_mask.numel()
        
        print(f"[DAG-BEV统计] 扰动幅度 - 最大: {max_diff:.6f}, 平均: {mean_diff:.6f}")
        print(f"[DAG-BEV统计] 攻击覆盖率: {attack_coverage:.2%}")
        
        # 计算像素级影响统计
        pixel_change_ratio = (diff > 1e-6).float().mean().item()
        significant_change_ratio = (diff > 0.01).float().mean().item()
        
        print(f"[DAG-BEV统计] 像素变化比例: {pixel_change_ratio:.2%}")
        print(f"[DAG-BEV统计] 显著变化比例: {significant_change_ratio:.2%}")
        
        # 计算特征空间变化 - 修复维度参数问题
        orig_norm = torch.norm(original_images.flatten(start_dim=1), dim=1, keepdim=True)
        adv_norm = torch.norm(adv_images.flatten(start_dim=1), dim=1, keepdim=True)
        norm_change = ((adv_norm - orig_norm) / (orig_norm + 1e-8)).abs().mean().item()
        
        print(f"[DAG-BEV统计] 特征范数变化: {norm_change:.4f}")
        print(f"[DAG-BEV统计] DAG攻击成功应用到{len(adv_images)}张图像")
    
    def adaptive_dag_attack(self, images, rots, trans, intrins, post_rots, post_trans, binimgs,
                           max_epsilon=0.05, target_iou_drop=0.3, max_iterations=50):
        """
        自适应DAG攻击 - 自动调整攻击强度直到达到目标效果
        """
        print(f"[自适应DAG] 目标IoU下降: {target_iou_drop:.2%}, 最大迭代: {max_iterations}")
        
        # 计算原始IoU
        with torch.no_grad():
            original_output = self.model(images.unsqueeze(0), rots.unsqueeze(0), trans.unsqueeze(0),
                                       intrins.unsqueeze(0), post_rots.unsqueeze(0), post_trans.unsqueeze(0))
            original_pred = original_output.sigmoid()
            target_bev = binimgs.unsqueeze(0).float()
            
            orig_intersection = (original_pred * target_bev).sum()
            orig_union = original_pred.sum() + target_bev.sum() - orig_intersection
            original_iou = orig_intersection / (orig_union + 1e-8)
            target_iou = original_iou * (1.0 - target_iou_drop)
        
        print(f"[自适应DAG] 原始IoU: {original_iou.item():.4f}, 目标IoU: {target_iou.item():.4f}")
        
        # 二分搜索最优epsilon
        low_eps, high_eps = 0.001, max_epsilon
        best_adv_images = images.clone()
        
        for iteration in range(max_iterations):
            current_eps = (low_eps + high_eps) / 2
            
            # 使用当前epsilon进行攻击
            adv_images = self.dag_bev_attack(
                images, rots, trans, intrins, post_rots, post_trans, binimgs,
                epsilon=current_eps, num_steps=15, attack_pattern='dense'
            )
            
            # 评估攻击效果
            with torch.no_grad():
                adv_output = self.model(adv_images.unsqueeze(0), rots.unsqueeze(0), trans.unsqueeze(0),
                                      intrins.unsqueeze(0), post_rots.unsqueeze(0), post_trans.unsqueeze(0))
                adv_pred = adv_output.sigmoid()
                
                adv_intersection = (adv_pred * target_bev).sum()
                adv_union = adv_pred.sum() + target_bev.sum() - adv_intersection
                current_iou = adv_intersection / (adv_union + 1e-8)
            
            print(f"[自适应DAG] 迭代{iteration}: eps={current_eps:.4f}, IoU={current_iou.item():.4f}")
            
            if current_iou <= target_iou:
                # 攻击效果足够，尝试更小的epsilon
                high_eps = current_eps
                best_adv_images = adv_images.clone()
                if abs(current_iou - target_iou) < 0.01:  # 足够接近目标
                    break
            else:
                # 攻击效果不足，增加epsilon
                low_eps = current_eps
            
            if abs(high_eps - low_eps) < 0.001:  # 收敛
                break
        
        return best_adv_images
    
    def apply_dag_attack(self, images, rots, trans, intrins, post_rots, post_trans, binimgs, 
                        intensity='medium'):
        """
        应用DAG攻击的统一接口，与现有攻击系统兼容
        """
        configs = {
            'low': {
                'epsilon': 0.015,
                'num_steps': 10,
                'attack_pattern': 'vehicle_focus',
                'target_density': 0.5
            },
            'medium': {
                'epsilon': 0.015,
                'num_steps': 15,
                'attack_pattern': 'dense',
                'target_density': 0.5
            },
            'high': {
                'epsilon': 0.04,
                'num_steps': 30,
                'attack_pattern': 'dense',
                'target_density': 1.0
            },
            'adaptive': {
                'method': 'adaptive',
                'target_iou_drop': 0.4
            }
        }
        
        if intensity not in configs:
            raise ValueError(f"不支持的DAG攻击强度: {intensity}")
        
        config = configs[intensity]
        print(f"[DAG攻击] 配置: {intensity} - {config}")
        
        if config.get('method') == 'adaptive':
            return self.adaptive_dag_attack(
                images, rots, trans, intrins, post_rots, post_trans, binimgs,
                target_iou_drop=config['target_iou_drop']
            )
        else:
            return self.dag_bev_attack(
                images, rots, trans, intrins, post_rots, post_trans, binimgs,
                epsilon=config['epsilon'],
                num_steps=config['num_steps'],
                attack_pattern=config['attack_pattern'],
                target_density=config['target_density']
            ) 