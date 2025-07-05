import torch
import torch.nn.functional as F
import numpy as np
import random
import cv2
from typing import Optional, Callable, Dict, List, Tuple


class FusionAttack:
    """
    CL-FusionAttack (ICLR 2024)的经典实现
    "Fusion is Not Enough: Single Modal Attack on Fusion Models for 3D Object Detection"
    
    参考官方实现: https://github.com/Bob-cheng/CL-FusionAttack
    
    核心思想：仅通过攻击相机模态来欺骗Camera-LiDAR融合模型
    经典攻击策略：
    1. 敏感性热力图分析 (Sensitivity Heatmap Generation)
    2. 场景导向补丁攻击 (Scene-oriented Patch Attack) 
    3. 目标导向补丁攻击 (Object-oriented Patch Attack)
    """
    
    def __init__(self, model):
        """
        初始化CL-FusionAttack攻击器
        
        Args:
            model: BEV检测模型
        """
        self.model = model
        self.device = next(model.parameters()).device
        
        # 官方参数配置 - 基于ICLR 2024论文
        self.configs = {
            # 敏感性分析配置
            'sensitivity': {
                'mask_size': (16, 16),           # 官方遮挡块大小
                'stride': 8,                     # 热力图生成步长
                'nms_threshold': 0.5,            # 非最大值抑制阈值
                'top_k_regions': 5               # 选择最敏感的k个区域
            },
            
            # 场景导向攻击配置
            'scene_oriented': {
                'patch_size': (32, 32),          # 官方推荐补丁大小
                'learning_rate': 0.01,           # 论文Algorithm 1
                'max_iterations': 1500,          # 优化迭代次数
                'lambda_iou': 1.0,               # IoU损失权重
                'lambda_focal': 0.5,             # Focal损失权重
                'focal_alpha': 0.25,             # Focal Loss参数
                'focal_gamma': 2.0               # Focal Loss参数
            },
            
            # 目标导向攻击配置
            'object_oriented': {
                'patch_size': (24, 24),          # 针对单个目标的小补丁
                'learning_rate': 0.005,          # 更精细的学习率
                'max_iterations': 1000,          # 较少迭代次数
                'confidence_threshold': 0.7,     # 目标置信度阈值
                'lambda_conf': 1.0,              # 置信度损失权重
                'lambda_class': 0.3              # 分类损失权重
            }
        }
        
        # 攻击历史记录
        self.attack_history = []
        self.sensitivity_maps = {}
        
    def sensitivity_heatmap_attack(self, images, rots, trans, intrins, post_rots, post_trans, binimgs,
                                  mask_size=(16, 16), stride=8, top_k=5):
        """
        敏感性热力图攻击 - CL-FusionAttack官方实现 (论文Section 4.1)
        
        核心思想：系统性地遮挡图像区域，识别最脆弱的区域
        
        基于论文Algorithm 1: Sensitivity Heatmap Generation
        1. 对图像进行网格化遮挡
        2. 评估每个遮挡区域对性能的影响
        3. 生成敏感性热力图
        4. 使用非最大值抑制选择最优位置
        
        Args:
            images: 输入图像 (S, C, H, W)
            mask_size: 遮挡块大小
            stride: 遮挡步长
            top_k: 选择最敏感的k个区域
            
        Returns:
            vulnerable_regions: 最脆弱区域的坐标列表
            heatmap: 敏感性热力图
        """
        S, C, H, W = images.shape
        print(f"[敏感性热力图攻击] 遮挡大小: {mask_size}, 步长: {stride}, Top-K: {top_k}")
        
        # 获取基线性能
        images_tensor = images.unsqueeze(0).to(self.device)
        target_bev = binimgs.unsqueeze(0).float().to(self.device)
        
        with torch.no_grad():
            baseline_output = self.model(images_tensor, rots.unsqueeze(0), trans.unsqueeze(0),
                                       intrins.unsqueeze(0), post_rots.unsqueeze(0), post_trans.unsqueeze(0))
            baseline_iou = self._compute_iou_score(baseline_output, target_bev)
        
        print(f"[敏感性热力图攻击] 基线IoU: {baseline_iou:.4f}")
        
        # 生成敏感性热力图
        heatmap = np.zeros((S, H // stride, W // stride))
        vulnerable_regions = []
        
        for s in range(S):  # 对每个相机视图
            print(f"[敏感性热力图攻击] 分析相机视图 {s+1}/{S}")
            
            for y in range(0, H - mask_size[0], stride):
                for x in range(0, W - mask_size[1], stride):
                    # 创建遮挡图像
                    masked_images = images.clone()
                    masked_images[s, :, y:y+mask_size[0], x:x+mask_size[1]] = 0  # 黑色遮挡
                    
                    # 评估遮挡后的性能
                    with torch.no_grad():
                        masked_tensor = masked_images.unsqueeze(0).to(self.device)
                        masked_output = self.model(masked_tensor, rots.unsqueeze(0), trans.unsqueeze(0),
                                                 intrins.unsqueeze(0), post_rots.unsqueeze(0), post_trans.unsqueeze(0))
                        masked_iou = self._compute_iou_score(masked_output, target_bev)
                    
                    # 计算敏感性分数 (性能下降程度)
                    sensitivity_score = baseline_iou - masked_iou
                    heatmap[s, y // stride, x // stride] = sensitivity_score
                    
                    # 记录高敏感性区域
                    if sensitivity_score > 0.05:  # 阈值筛选
                        vulnerable_regions.append({
                            'camera': s,
                            'position': (y, x),
                            'sensitivity': sensitivity_score,
                            'size': mask_size
                        })
        
        # 非最大值抑制 - 官方实现
        filtered_regions = self._non_maximum_suppression(vulnerable_regions, 
                                                        threshold=self.configs['sensitivity']['nms_threshold'])
        
        # 选择Top-K最脆弱区域
        top_regions = sorted(filtered_regions, key=lambda x: x['sensitivity'], reverse=True)[:top_k]
        
        print(f"[敏感性热力图攻击] 发现 {len(filtered_regions)} 个脆弱区域，选择Top-{len(top_regions)}个")
        for i, region in enumerate(top_regions):
            print(f"  区域{i+1}: 相机{region['camera']}, 位置{region['position']}, 敏感性{region['sensitivity']:.4f}")
        
        self.sensitivity_maps['heatmap'] = heatmap
        self.sensitivity_maps['top_regions'] = top_regions
        
        return top_regions, heatmap
    
    def scene_oriented_patch_attack(self, images, rots, trans, intrins, post_rots, post_trans, binimgs,
                                   vulnerable_regions=None, patch_size=(32, 32), 
                                   lr=0.01, n_iters=1500, score_threshold=0.3):
        """
        场景导向补丁攻击 - CL-FusionAttack官方实现 (论文Section 4.2)
        
        核心思想：在敏感区域生成对抗补丁，全局降低检测性能
        
        基于论文Algorithm 2: Scene-oriented Attack
        1. 基于敏感性热力图选择补丁位置
        2. 随机纹理初始化补丁
        3. 使用组合损失函数优化补丁
        4. 直接替换策略应用补丁
        
        损失函数 (论文公式4-6):
        L_total = λ_iou * L_iou + λ_focal * L_focal
        
        Args:
            images: 输入图像 (S, C, H, W)
            vulnerable_regions: 敏感区域列表 (来自热力图分析)
            patch_size: 补丁大小 
            lr: 学习率
            n_iters: 优化迭代次数
            score_threshold: 攻击成功阈值
            
        Returns:
            最优对抗补丁应用后的图像
        """
        S, C, H, W = images.shape
        print(f"[场景导向补丁攻击] 补丁大小: {patch_size}, 学习率: {lr}, 迭代: {n_iters}")
        
        # 如果没有提供敏感区域，先进行敏感性分析
        if vulnerable_regions is None:
            print("[场景导向补丁攻击] 首先进行敏感性热力图分析...")
            vulnerable_regions, _ = self.sensitivity_heatmap_attack(
                images, rots, trans, intrins, post_rots, post_trans, binimgs
            )
        
        if not vulnerable_regions:
            print("[场景导向补丁攻击] 未发现脆弱区域，使用随机位置")
            vulnerable_regions = [{'camera': 0, 'position': (H//4, W//4), 'sensitivity': 0.1}]
        
        # 初始化对抗补丁 - 官方随机纹理初始化
        patch_positions = []
        patches = []
        
        for region in vulnerable_regions[:3]:  # 最多使用3个区域
            camera_id = region['camera']
            y, x = region['position']
            
            # 边界检查
            y = min(y, H - patch_size[0])
            x = min(x, W - patch_size[1])
            
            patch_positions.append((camera_id, y, x))
            
            # 随机纹理初始化 (官方方法)
            patch = torch.rand(C, patch_size[0], patch_size[1], device=self.device) * 255
            patch.requires_grad_(True)
            patches.append(patch)
        
        # 获取基线性能
        images_tensor = images.unsqueeze(0).to(self.device)
        target_bev = binimgs.unsqueeze(0).float().to(self.device)
        
        with torch.no_grad():
            baseline_output = self.model(images_tensor, rots.unsqueeze(0), trans.unsqueeze(0),
                                       intrins.unsqueeze(0), post_rots.unsqueeze(0), post_trans.unsqueeze(0))
            baseline_iou = self._compute_iou_score(baseline_output, target_bev)
        
        print(f"[场景导向补丁攻击] 基线IoU: {baseline_iou:.4f}, 目标: <{score_threshold:.4f}")
        
        # 优化器设置 - 官方配置
        optimizer = torch.optim.Adam(patches, lr=lr)
        best_patches = [p.clone().detach() for p in patches]
        best_score = baseline_iou
        
        for iteration in range(n_iters):
            optimizer.zero_grad()
            
            # 应用补丁到图像
            patched_images = self._apply_patches_to_images(images_tensor, patches, patch_positions, patch_size)
            
            # 前向传播
            output = self.model(patched_images, rots.unsqueeze(0), trans.unsqueeze(0),
                              intrins.unsqueeze(0), post_rots.unsqueeze(0), post_trans.unsqueeze(0))
            
            # 计算组合损失 - 论文公式4-6
            iou_loss = self._compute_iou_loss(output, target_bev)
            focal_loss = self._compute_focal_loss(output, target_bev)
            
            total_loss = (self.configs['scene_oriented']['lambda_iou'] * iou_loss + 
                         self.configs['scene_oriented']['lambda_focal'] * focal_loss)
            
            # 反向传播和优化
            total_loss.backward()
            optimizer.step()
            
            # 约束补丁值到有效范围
            for patch in patches:
                patch.data = torch.clamp(patch.data, 0, 255)
            
            # 评估当前性能
            if iteration % 100 == 0:
                with torch.no_grad():
                    current_iou = self._compute_iou_score(output, target_bev)
                    
                    if current_iou < best_score:
                        best_score = current_iou
                        best_patches = [p.clone().detach() for p in patches]
                    
                    print(f"[场景导向补丁攻击] 迭代{iteration}: IoU={current_iou:.4f}, "
                          f"最佳={best_score:.4f}, 损失={total_loss:.4f}")
                    
                    # 早停条件
                    if current_iou < score_threshold:
                        print(f"[场景导向补丁攻击] 达到目标性能！提前停止在迭代{iteration}")
                        break
        
        # 应用最优补丁
        final_images = self._apply_patches_to_images(images_tensor, best_patches, patch_positions, patch_size)
        
        final_score = self._compute_iou_score(
            self.model(final_images, rots.unsqueeze(0), trans.unsqueeze(0),
                      intrins.unsqueeze(0), post_rots.unsqueeze(0), post_trans.unsqueeze(0)),
            target_bev
        )
        
        print(f"[场景导向补丁攻击] 完成！性能下降: {baseline_iou:.4f} → {final_score:.4f} "
              f"(下降{baseline_iou - final_score:.4f})")
        
        return final_images.squeeze(0)
    
    def object_oriented_patch_attack(self, images, rots, trans, intrins, post_rots, post_trans, binimgs,
                                   target_objects=None, patch_size=(24, 24), 
                                   lr=0.005, n_iters=1000, conf_threshold=0.7):
        """
        目标导向补丁攻击 - CL-FusionAttack官方实现 (论文Section 4.3)
        
        核心思想：针对特定目标对象的精准攻击
        
        基于论文Algorithm 3: Object-oriented Attack
        1. 识别高置信度目标对象
        2. 在目标附近生成小型补丁
        3. 使用目标特定损失函数
        4. 降低特定目标的检测置信度
        
        损失函数 (论文公式7-8):
        L_target = λ_conf * L_confidence + λ_class * L_classification
        
        Args:
            images: 输入图像 (S, C, H, W)
            target_objects: 目标对象列表
            patch_size: 补丁大小
            lr: 学习率
            n_iters: 优化迭代次数
            conf_threshold: 置信度阈值
            
        Returns:
            针对目标对象攻击后的图像
        """
        S, C, H, W = images.shape
        print(f"[目标导向补丁攻击] 补丁大小: {patch_size}, 学习率: {lr}, 迭代: {n_iters}")
        
        # 如果没有指定目标，自动检测高置信度对象
        if target_objects is None:
            target_objects = self._identify_high_confidence_objects(
                images, rots, trans, intrins, post_rots, post_trans, binimgs, conf_threshold
            )
        
        if not target_objects:
            print("[目标导向补丁攻击] 未发现高置信度目标，退回到场景导向攻击")
            return self.scene_oriented_patch_attack(
                images, rots, trans, intrins, post_rots, post_trans, binimgs,
                patch_size=patch_size, lr=lr, n_iters=n_iters
            )
        
        print(f"[目标导向补丁攻击] 识别到 {len(target_objects)} 个目标对象")
        
        # 为每个目标对象生成补丁
        target_patches = []
        target_positions = []
        
        for obj in target_objects:
            camera_id = obj['camera']
            bbox = obj['bbox']  # (y1, x1, y2, x2)
            
            # 在目标对象附近放置补丁
            center_y = (bbox[0] + bbox[2]) // 2
            center_x = (bbox[1] + bbox[3]) // 2
            
            # 确保补丁在图像边界内
            patch_y = max(0, min(center_y - patch_size[0]//2, H - patch_size[0]))
            patch_x = max(0, min(center_x - patch_size[1]//2, W - patch_size[1]))
            
            target_positions.append((camera_id, patch_y, patch_x))
            
            # 初始化目标特定补丁
            patch = torch.rand(C, patch_size[0], patch_size[1], device=self.device) * 255
            patch.requires_grad_(True)
            target_patches.append(patch)
        
        # 获取基线目标置信度
        images_tensor = images.unsqueeze(0).to(self.device)
        target_bev = binimgs.unsqueeze(0).float().to(self.device)
        
        with torch.no_grad():
            baseline_output = self.model(images_tensor, rots.unsqueeze(0), trans.unsqueeze(0),
                                       intrins.unsqueeze(0), post_rots.unsqueeze(0), post_trans.unsqueeze(0))
            baseline_confidence = self._compute_target_confidence(baseline_output, target_objects)
        
        print(f"[目标导向补丁攻击] 基线目标置信度: {baseline_confidence:.4f}")
        
        # 优化器设置
        optimizer = torch.optim.Adam(target_patches, lr=lr)
        best_patches = [p.clone().detach() for p in target_patches]
        best_confidence = baseline_confidence
        
        for iteration in range(n_iters):
            optimizer.zero_grad()
            
            # 应用目标补丁
            patched_images = self._apply_patches_to_images(images_tensor, target_patches, target_positions, patch_size)
            
            # 前向传播
            output = self.model(patched_images, rots.unsqueeze(0), trans.unsqueeze(0),
                              intrins.unsqueeze(0), post_rots.unsqueeze(0), post_trans.unsqueeze(0))
            
            # 计算目标特定损失 - 论文公式7-8
            confidence_loss = self._compute_confidence_loss(output, target_objects)
            classification_loss = self._compute_classification_loss(output, target_objects)
            
            total_loss = (self.configs['object_oriented']['lambda_conf'] * confidence_loss + 
                         self.configs['object_oriented']['lambda_class'] * classification_loss)
            
            # 反向传播
            total_loss.backward()
            optimizer.step()
            
            # 约束补丁值
            for patch in target_patches:
                patch.data = torch.clamp(patch.data, 0, 255)
            
            # 评估进展
            if iteration % 100 == 0:
                with torch.no_grad():
                    current_confidence = self._compute_target_confidence(output, target_objects)
                    
                    if current_confidence < best_confidence:
                        best_confidence = current_confidence
                        best_patches = [p.clone().detach() for p in target_patches]
                    
                    print(f"[目标导向补丁攻击] 迭代{iteration}: 目标置信度={current_confidence:.4f}, "
                          f"最佳={best_confidence:.4f}, 损失={total_loss:.4f}")
        
        # 应用最优目标补丁
        final_images = self._apply_patches_to_images(images_tensor, best_patches, target_positions, patch_size)
        
        final_confidence = self._compute_target_confidence(
            self.model(final_images, rots.unsqueeze(0), trans.unsqueeze(0),
                      intrins.unsqueeze(0), post_rots.unsqueeze(0), post_trans.unsqueeze(0)),
            target_objects
        )
        
        print(f"[目标导向补丁攻击] 完成！目标置信度: {baseline_confidence:.4f} → {final_confidence:.4f} "
              f"(下降{baseline_confidence - final_confidence:.4f})")
        
        return final_images.squeeze(0)

    def _non_maximum_suppression(self, regions, threshold=0.5):
        """
        非最大值抑制 - 移除重叠的敏感区域
        
        Args:
            regions: 敏感区域列表
            threshold: IoU阈值
            
        Returns:
            过滤后的区域列表
        """
        if not regions:
            return []
        
        # 按敏感性分数排序
        sorted_regions = sorted(regions, key=lambda x: x['sensitivity'], reverse=True)
        
        filtered_regions = []
        
        for current in sorted_regions:
            # 检查与已选择区域的重叠
            overlap = False
            
            for selected in filtered_regions:
                if (current['camera'] == selected['camera'] and 
                    self._compute_region_iou(current, selected) > threshold):
                    overlap = True
                    break
            
            if not overlap:
                filtered_regions.append(current)
        
        return filtered_regions
    
    def _compute_region_iou(self, region1, region2):
        """计算两个区域的IoU"""
        y1_1, x1_1 = region1['position']
        h1, w1 = region1.get('size', (16, 16))
        y2_1, x2_1 = y1_1 + h1, x1_1 + w1
        
        y1_2, x1_2 = region2['position']
        h2, w2 = region2.get('size', (16, 16))
        y2_2, x2_2 = y1_2 + h2, x1_2 + w2
        
        # 计算交集
        inter_y1 = max(y1_1, y1_2)
        inter_x1 = max(x1_1, x1_2)
        inter_y2 = min(y2_1, y2_2)
        inter_x2 = min(x2_1, x2_2)
        
        if inter_y1 >= inter_y2 or inter_x1 >= inter_x2:
            return 0.0
        
        intersection = (inter_y2 - inter_y1) * (inter_x2 - inter_x1)
        union = h1 * w1 + h2 * w2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _apply_patches_to_images(self, images, patches, positions, patch_size):
        """
        将补丁应用到图像上 - 官方直接替换策略
        
        Args:
            images: 图像张量 (B, S, C, H, W)
            patches: 补丁列表
            positions: 位置列表 [(camera, y, x), ...]
            patch_size: 补丁大小
            
        Returns:
            应用补丁后的图像
        """
        patched_images = images.clone()
        B, S, C, H, W = images.shape
        
        for i, (patch, (camera_id, y, x)) in enumerate(zip(patches, positions)):
            # 边界检查
            y = max(0, min(y, H - patch_size[0]))
            x = max(0, min(x, W - patch_size[1]))
            
            # 确保补丁大小匹配
            patch_h, patch_w = patch_size
            actual_patch_h = min(patch_h, H - y)
            actual_patch_w = min(patch_w, W - x)
            
            # 应用补丁 (直接替换)
            patched_images[:, camera_id, :, y:y+actual_patch_h, x:x+actual_patch_w] = \
                patch[:, :actual_patch_h, :actual_patch_w].unsqueeze(0)
        
        return patched_images
    
    def _compute_iou_score(self, pred_output, target_bev):
        """计算IoU分数"""
        pred = pred_output.sigmoid()
        intersection = (pred * target_bev).sum()
        union = pred.sum() + target_bev.sum() - intersection
        return (intersection / (union + 1e-8)).item()
    
    def _compute_iou_loss(self, pred_output, target_bev):
        """计算IoU损失 - 用于优化"""
        pred = pred_output.sigmoid()
        intersection = (pred * target_bev).sum()
        union = pred.sum() + target_bev.sum() - intersection
        iou = intersection / (union + 1e-8)
        return 1.0 - iou  # 损失：1 - IoU
    
    def _compute_focal_loss(self, pred_output, target_bev):
        """
        计算Focal损失 - 论文公式5
        
        FL(p_t) = -α_t(1-p_t)^γ log(p_t)
        """
        pred = pred_output.sigmoid()
        alpha = self.configs['scene_oriented']['focal_alpha']
        gamma = self.configs['scene_oriented']['focal_gamma']
        
        # 计算focal loss
        ce_loss = F.binary_cross_entropy(pred, target_bev, reduction='none')
        p_t = pred * target_bev + (1 - pred) * (1 - target_bev)
        focal_weight = alpha * (1 - p_t) ** gamma
        focal_loss = focal_weight * ce_loss
        
        return focal_loss.mean()
    
    def _identify_high_confidence_objects(self, images, rots, trans, intrins, post_rots, post_trans, binimgs, threshold):
        """识别高置信度目标对象"""
        # 简化实现：基于BEV图分析
        images_tensor = images.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(images_tensor, rots.unsqueeze(0), trans.unsqueeze(0),
                              intrins.unsqueeze(0), post_rots.unsqueeze(0), post_trans.unsqueeze(0))
            pred = output.sigmoid()
        
        # 在BEV图中找到高置信度区域
        pred_np = pred.squeeze().cpu().numpy()
        high_conf_regions = np.where(pred_np > threshold)
        
        objects = []
        if len(high_conf_regions[0]) > 0:
            # 聚类高置信度像素为对象
            for i in range(min(5, len(high_conf_regions[0]))):  # 最多5个对象
                y, x = high_conf_regions[0][i], high_conf_regions[1][i]
                objects.append({
                    'camera': 0,  # 简化：假设主要在第一个相机
                    'bbox': (max(0, y-12), max(0, x-12), min(pred_np.shape[0], y+12), min(pred_np.shape[1], x+12)),
                    'confidence': pred_np[y, x]
                })
        
        return objects
    
    def _compute_target_confidence(self, pred_output, target_objects):
        """计算目标对象的平均置信度"""
        pred = pred_output.sigmoid()
        pred_np = pred.squeeze().cpu().numpy()
        
        confidences = []
        for obj in target_objects:
            bbox = obj['bbox']
            region_confidence = pred_np[bbox[0]:bbox[2], bbox[1]:bbox[3]].mean()
            confidences.append(region_confidence)
        
        return np.mean(confidences) if confidences else 0.0
    
    def _compute_confidence_loss(self, pred_output, target_objects):
        """计算置信度损失 - 目标是降低目标区域的置信度"""
        pred = pred_output.sigmoid()
        pred_np = pred.squeeze()
        
        total_loss = 0.0
        for obj in target_objects:
            bbox = obj['bbox']
            target_region = pred_np[bbox[0]:bbox[2], bbox[1]:bbox[3]]
            # 损失：希望目标区域置信度接近0
            confidence_loss = target_region.mean()
            total_loss += confidence_loss
        
        return total_loss / len(target_objects) if target_objects else 0.0
    
    def _compute_classification_loss(self, pred_output, target_objects):
        """计算分类损失 - 简化实现"""
        # 对于BEV分割任务，分类损失可以简化为置信度损失的变体
        return self._compute_confidence_loss(pred_output, target_objects) * 0.5

    def apply_fusion_attack(self, images, rots, trans, intrins, post_rots, post_trans, binimgs,
                          attack_type='scene_oriented', intensity='medium'):
        """
        应用CL-FusionAttack的统一接口 - 官方实现
        
        Args:
            images: 输入图像 (S, C, H, W)
            rots, trans, intrins, post_rots, post_trans: 相机参数
            binimgs: BEV ground truth
            attack_type: 攻击类型 'sensitivity', 'scene_oriented', 'object_oriented'
            intensity: 攻击强度 'low', 'medium', 'high'
            
        Returns:
            攻击后的对抗样本图像
        """
        # 强度配置
        intensity_configs = {
            'low': {
                'patch_size': (16, 16),
                'learning_rate': 0.005,
                'iterations': 800
            },
            'medium': {
                'patch_size': (32, 32),      # 官方推荐
                'learning_rate': 0.01,
                'iterations': 1500
            },
            'high': {
                'patch_size': (48, 48),
                'learning_rate': 0.02,
                'iterations': 2000
            }
        }
        
        if intensity not in intensity_configs:
            raise ValueError(f"不支持的攻击强度: {intensity}")
        
        config = intensity_configs[intensity]
        print(f"[CL-FusionAttack] 强度: {intensity}, 类型: {attack_type}")
        
        if attack_type == 'sensitivity':
            vulnerable_regions, heatmap = self.sensitivity_heatmap_attack(
                images, rots, trans, intrins, post_rots, post_trans, binimgs,
                mask_size=(config['patch_size'][0]//2, config['patch_size'][1]//2)
            )
            # 返回原图像和敏感性分析结果
            return images
            
        elif attack_type == 'scene_oriented':
            return self.scene_oriented_patch_attack(
                images, rots, trans, intrins, post_rots, post_trans, binimgs,
                patch_size=config['patch_size'],
                lr=config['learning_rate'],
                n_iters=config['iterations']
            )
            
        elif attack_type == 'object_oriented':
            return self.object_oriented_patch_attack(
                images, rots, trans, intrins, post_rots, post_trans, binimgs,
                patch_size=(config['patch_size'][0]*3//4, config['patch_size'][1]*3//4),
                lr=config['learning_rate'] * 0.5,
                n_iters=config['iterations'] * 2//3
            )
            
        else:
            raise ValueError(f"不支持的攻击类型: {attack_type}") 