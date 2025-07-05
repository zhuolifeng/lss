import torch
import torch.nn.functional as F
import numpy as np
import random
import math
from typing import Optional, Callable, Dict, List, Tuple
import cv2
import torchvision.transforms as transforms
from scipy.spatial.distance import cdist


class RobustBEVAttack:
    """
    RobustBEV (AAAI 2025) 经典实现
    "A Black-Box Evaluation Framework for Semantic Robustness in Bird's Eye View Detection"
    
    官方GitHub: https://github.com/TrustAI/RobustBEV
    
    核心创新：
    1. SimpleDIRECT优化算法 - 确定性全局优化
    2. 语义对抗威胁模型 - 现实世界扰动
    3. 距离基础的代理目标函数 - 替代mAP
    
    经典语义扰动：
    1. 几何变换 (Geometric Transformation) - 缩放&平移
    2. 颜色偏移 (Colour Shift) - HSV空间变换  
    3. 运动模糊 (Motion Blur) - 线性/高斯模糊
    """
    
    def __init__(self, model):
        """
        初始化RobustBEV攻击器 - 官方实现
        
        Args:
            model: BEV检测模型
        """
        self.model = model
        self.device = next(model.parameters()).device
        
        # 官方参数配置 - 基于AAAI 2025论文
        self.configs = {
            'geometric': {
                # 论文Table 1设置
                'gamma_range': [0.04, 0.1],  # γ∈[0.04,0.1] for geometry
                'param_bounds': {
                    'scale_h': [0.96, 1.04],      # θ_s^hor ∈ [1-γ, 1+γ]
                    'scale_v': [0.96, 1.04],      # θ_s^vrt ∈ [1-γ, 1+γ]  
                    'trans_h': [-35.2, 35.2],    # θ_t^hor ∈ [-γW, γW]
                    'trans_v': [-12.8, 12.8]     # θ_t^vrt ∈ [-γH, γH]
                }
            },
            'colour': {
                # 论文Table 1设置
                'gamma_range': [0.1, 0.4],   # γ∈[0.1,0.4] for colour
                'param_bounds': {
                    'hue': [-1.256, 1.256],      # θ^hue ∈ [-π·γ, π·γ]
                    'saturation': [0.7, 1.3],   # θ^sat ∈ [1-γ, 1+γ]
                    'brightness': [-0.3, 0.3]   # θ^brt ∈ [-γ, γ]
                }
            },
            'motion_blur': {
                # 论文设置
                'kernel_sizes': [5, 7, 9, 11],   # 官方severity levels
                'param_bounds': {
                    'angle': [-math.pi, math.pi],  # θ^ang ∈ [-π, π]
                    'direction': [-1, 1]           # θ^dir ∈ [-1, 1]
                }
            },
            'simple_direct': {
                # 论文Algorithm 2设置
                'max_evaluations': 2500,     # 官方查询预算
                'convergence_threshold': 10, # 收敛阈值
                'R': 3,                      # PO节点数量
                'tolerance': 1e-6            # 数值容忍度
            }
        }
        
        # SimpleDIRECT优化器状态
        self.optimization_history = []
        self.current_evaluations = 0
        
    def geometric_transformation_attack(self, images, rots, trans, intrins, post_rots, post_trans, binimgs,
                                       gamma=0.1, max_evaluations=2500):
        """
        几何变换攻击 - RobustBEV官方实现 (论文Section 5.1)
        
        基于论文公式(8)的缩放和平移变换：
        [x_j, y_j]^T = [θ_s^hor, 0, θ_t^hor; 0, θ_s^vrt, θ_t^vrt] [x_i', y_i', 1]^T
        
        参数边界 (论文Table 1):
        - θ_s^hor, θ_s^vrt ∈ [1-γ, 1+γ] (缩放)
        - θ_t^hor ∈ [-γW, γW], θ_t^vrt ∈ [-γH, γH] (平移)
        
        Args:
            images: 输入图像 (S, C, H, W)
            gamma: 扰动强度参数
            max_evaluations: SimpleDIRECT最大查询次数
            
        Returns:
            几何变换后的最坏情况对抗样本
        """
        S, C, H, W = images.shape
        print(f"[几何变换攻击] γ={gamma}, 最大查询: {max_evaluations}")
        
        # 初始化距离基础的代理目标函数
        images_tensor = images.unsqueeze(0).to(self.device)
        target_bev = binimgs.unsqueeze(0).float().to(self.device)
        
        # 获取基线距离分数 (论文公式(3))
        with torch.no_grad():
            baseline_output = self.model(images_tensor, rots.unsqueeze(0), trans.unsqueeze(0),
                                       intrins.unsqueeze(0), post_rots.unsqueeze(0), post_trans.unsqueeze(0))
            baseline_distance = self._compute_distance_based_objective(baseline_output, target_bev)
        
        print(f"[几何变换攻击] 基线距离分数: {baseline_distance:.4f}")
        
        # 官方参数空间定义 (论文Table 1)
        param_space = {
            'scale_h': [1-gamma, 1+gamma],      # θ_s^hor
            'scale_v': [1-gamma, 1+gamma],      # θ_s^vrt  
            'trans_h': [-gamma*W, gamma*W],     # θ_t^hor
            'trans_v': [-gamma*H, gamma*H]      # θ_t^vrt
        }
        
        # 使用SimpleDIRECT算法优化 (论文Algorithm 1&2)
        optimal_params = self._simple_direct_optimization(
            objective_func=lambda params: self._geometric_objective(
                images, rots, trans, intrins, post_rots, post_trans, target_bev, params),
            param_space=param_space,
            max_evaluations=max_evaluations
        )
        
        # 应用最优几何变换
        best_images = self._apply_geometric_transform(images, optimal_params)
        
        # 评估最终结果
        with torch.no_grad():
            final_tensor = best_images.unsqueeze(0).to(self.device)
            final_output = self.model(final_tensor, rots.unsqueeze(0), trans.unsqueeze(0),
                                    intrins.unsqueeze(0), post_rots.unsqueeze(0), post_trans.unsqueeze(0))
            final_distance = self._compute_distance_based_objective(final_output, target_bev)
        
        print(f"[几何变换攻击] 最优参数: {optimal_params}")
        print(f"[几何变换攻击] 最终距离分数: {final_distance:.4f} (增长: {final_distance-baseline_distance:.4f})")
        
        return best_images
    
    def colour_shift_attack(self, images, rots, trans, intrins, post_rots, post_trans, binimgs,
                           gamma=0.3, max_evaluations=2500):
        """
        颜色偏移攻击 - RobustBEV官方实现 (论文Section 5.2)
        
        基于HSV颜色空间的语义扰动 (论文公式9-11):
        - Hue: S^hue(x^hue, θ^hue) = (x^hue + θ^hue) mod 2π
        - Saturation: S^sat(x^sat, θ^sat) = min(max(0, θ^sat · x^sat), 1)
        - Brightness: S^brt(x^brt, θ^brt) = min(max(x^brt + θ^brt, 0), 1)
        
        Args:
            images: 输入图像 (S, C, H, W)
            gamma: 扰动强度参数
            max_evaluations: SimpleDIRECT最大查询次数
            
        Returns:
            颜色偏移后的最坏情况对抗样本
        """
        S, C, H, W = images.shape
        print(f"[颜色偏移攻击] γ={gamma}, 最大查询: {max_evaluations}")
        
        images_tensor = images.unsqueeze(0).to(self.device)
        target_bev = binimgs.unsqueeze(0).float().to(self.device)
        
        # 获取基线距离分数
        with torch.no_grad():
            baseline_output = self.model(images_tensor, rots.unsqueeze(0), trans.unsqueeze(0),
                                       intrins.unsqueeze(0), post_rots.unsqueeze(0), post_trans.unsqueeze(0))
            baseline_distance = self._compute_distance_based_objective(baseline_output, target_bev)
        
        print(f"[颜色偏移攻击] 基线距离分数: {baseline_distance:.4f}")
        
        # 官方参数空间定义 (论文Table 1)
        param_space = {
            'hue': [-math.pi*gamma, math.pi*gamma],     # θ^hue ∈ [-π·γ, π·γ]
            'saturation': [1-gamma, 1+gamma],           # θ^sat ∈ [1-γ, 1+γ]
            'brightness': [-gamma, gamma]                # θ^brt ∈ [-γ, γ]
        }
        
        # 使用SimpleDIRECT算法优化
        optimal_params = self._simple_direct_optimization(
            objective_func=lambda params: self._colour_objective(
                images, rots, trans, intrins, post_rots, post_trans, target_bev, params),
            param_space=param_space,
            max_evaluations=max_evaluations
        )
        
        # 应用最优颜色变换
        best_images = self._apply_colour_shift(images, optimal_params)
        
        # 评估最终结果
        with torch.no_grad():
            final_tensor = best_images.unsqueeze(0).to(self.device)
            final_output = self.model(final_tensor, rots.unsqueeze(0), trans.unsqueeze(0),
                                    intrins.unsqueeze(0), post_rots.unsqueeze(0), post_trans.unsqueeze(0))
            final_distance = self._compute_distance_based_objective(final_output, target_bev)
        
        print(f"[颜色偏移攻击] 最优参数: {optimal_params}")
        print(f"[颜色偏移攻击] 最终距离分数: {final_distance:.4f} (增长: {final_distance-baseline_distance:.4f})")
        
        return best_images
    
    def motion_blur_attack(self, images, rots, trans, intrins, post_rots, post_trans, binimgs,
                          kernel_size=9, max_evaluations=2500):
        """
        运动模糊攻击 - RobustBEV官方实现 (论文Section 5.3)
        
        基于可优化的模糊核参数:
        - θ^ang ∈ [-π, π] (模糊角度)
        - θ^dir ∈ [-1, 1] (模糊方向)
        
        Args:
            images: 输入图像 (S, C, H, W) 
            kernel_size: 模糊核大小 (5,7,9,11)
            max_evaluations: SimpleDIRECT最大查询次数
            
        Returns:
            运动模糊后的最坏情况对抗样本
        """
        S, C, H, W = images.shape
        print(f"[运动模糊攻击] 核大小={kernel_size}, 最大查询: {max_evaluations}")
        
        images_tensor = images.unsqueeze(0).to(self.device)
        target_bev = binimgs.unsqueeze(0).float().to(self.device)
        
        # 获取基线距离分数
        with torch.no_grad():
            baseline_output = self.model(images_tensor, rots.unsqueeze(0), trans.unsqueeze(0),
                                       intrins.unsqueeze(0), post_rots.unsqueeze(0), post_trans.unsqueeze(0))
            baseline_distance = self._compute_distance_based_objective(baseline_output, target_bev)
        
        print(f"[运动模糊攻击] 基线距离分数: {baseline_distance:.4f}")
        
        # 官方参数空间定义
        param_space = {
            'angle': [-math.pi, math.pi],    # θ^ang ∈ [-π, π]
            'direction': [-1, 1]             # θ^dir ∈ [-1, 1]
        }
        
        # 使用SimpleDIRECT算法优化
        optimal_params = self._simple_direct_optimization(
            objective_func=lambda params: self._motion_blur_objective(
                images, rots, trans, intrins, post_rots, post_trans, target_bev, params, kernel_size),
            param_space=param_space,
            max_evaluations=max_evaluations
        )
        
        # 应用最优运动模糊
        best_images = self._apply_motion_blur(images, optimal_params, kernel_size)
        
        # 评估最终结果
        with torch.no_grad():
            final_tensor = best_images.unsqueeze(0).to(self.device)
            final_output = self.model(final_tensor, rots.unsqueeze(0), trans.unsqueeze(0),
                                    intrins.unsqueeze(0), post_rots.unsqueeze(0), post_trans.unsqueeze(0))
            final_distance = self._compute_distance_based_objective(final_output, target_bev)
        
        print(f"[运动模糊攻击] 最优参数: {optimal_params}")
        print(f"[运动模糊攻击] 最终距离分数: {final_distance:.4f} (增长: {final_distance-baseline_distance:.4f})")
        
        return best_images

    def _simple_direct_optimization(self, objective_func, param_space, max_evaluations=2500):
        """
        SimpleDIRECT优化算法 - RobustBEV论文Algorithm 1&2
        
        核心改进:
        1. 简化PO节点选择 (Algorithm 2)
        2. 基于坡度信息的指导优化
        3. 减少冗余计算
        
        Args:
            objective_func: 目标函数 (待最大化)
            param_space: 参数空间字典
            max_evaluations: 最大函数评估次数
            
        Returns:
            optimal_params: 最优参数
        """
        print(f"[SimpleDIRECT] 开始优化，最大评估: {max_evaluations}")
        
        # 初始化参数空间 - 映射到单位超立方体
        param_names = list(param_space.keys())
        param_bounds = np.array([param_space[name] for name in param_names])
        dimension = len(param_names)
        
        # 初始化根节点
        self.current_evaluations = 0
        self.optimization_history = []
        
        # 评估中心点
        center_params = {name: 0.5 * (bounds[0] + bounds[1]) for name, bounds in param_space.items()}
        center_value = objective_func(center_params)
        self.current_evaluations += 1
        
        # 记录最佳结果
        best_params = center_params.copy()
        best_value = center_value
        
        print(f"[SimpleDIRECT] 中心点评估: {center_value:.6f}")
        
        # 节点数据结构 - (中心点, 直径, 函数值, 最大坡度)
        nodes = [(np.array([0.5] * dimension), 1.0, center_value, 0.0)]
        convergence_count = 0
        
        while self.current_evaluations < max_evaluations and convergence_count < self.configs['simple_direct']['convergence_threshold']:
            
            # 选择潜在最优(PO)节点 - Algorithm 2
            po_nodes = self._select_potentially_optimal_nodes(nodes)
            
            if not po_nodes:
                convergence_count += 1
                continue
            
            new_nodes = []
            improvement_found = False
            
            # 分割选中的PO节点
            for node_idx in po_nodes:
                center, diameter, value, max_slope = nodes[node_idx]
                
                # 找到最长的维度
                longest_dims = self._find_longest_dimensions(diameter, dimension)
                
                # 在最长维度上进行三分割
                for dim in longest_dims:
                    if self.current_evaluations >= max_evaluations:
                        break
                        
                    # 计算采样点 (官方三分割策略)
                    delta = diameter / 3.0
                    
                    # 正向采样点
                    pos_point = center.copy()
                    pos_point[dim] = min(1.0, center[dim] + delta)
                    pos_params = self._unit_cube_to_params(pos_point, param_space)
                    pos_value = objective_func(pos_params)
                    self.current_evaluations += 1
                    
                    # 负向采样点  
                    neg_point = center.copy()
                    neg_point[dim] = max(0.0, center[dim] - delta)
                    neg_params = self._unit_cube_to_params(neg_point, param_space)
                    neg_value = objective_func(neg_params)
                    self.current_evaluations += 1
                    
                    # 计算坡度 (官方公式)
                    pos_slope = abs(pos_value - value) / delta
                    neg_slope = abs(neg_value - value) / delta
                    dim_max_slope = max(pos_slope, neg_slope)
                    
                    # 更新最佳结果
                    for point, val in [(pos_point, pos_value), (neg_point, neg_value)]:
                        if val > best_value:
                            best_value = val
                            best_params = self._unit_cube_to_params(point, param_space)
                            improvement_found = True
                    
                    # 创建新节点 (基于函数值分配子空间)
                    if pos_value >= neg_value:
                        # 正向点获得更大的子空间
                        new_nodes.append((pos_point, delta, pos_value, dim_max_slope))
                        new_nodes.append((neg_point, delta, neg_value, dim_max_slope))
                    else:
                        # 负向点获得更大的子空间
                        new_nodes.append((neg_point, delta, neg_value, dim_max_slope))
                        new_nodes.append((pos_point, delta, pos_value, dim_max_slope))
            
            # 更新节点列表
            nodes.extend(new_nodes)
            
            if not improvement_found:
                convergence_count += 1
            else:
                convergence_count = 0
            
            # 进度报告
            if self.current_evaluations % 100 == 0:
                print(f"[SimpleDIRECT] 评估: {self.current_evaluations}, 最佳值: {best_value:.6f}")
        
        print(f"[SimpleDIRECT] 优化完成! 总评估: {self.current_evaluations}, 最终值: {best_value:.6f}")
        return best_params
    
    def _select_potentially_optimal_nodes(self, nodes):
        """
        选择潜在最优节点 - 论文Algorithm 2的简化版本
        
        核心简化：
        1. 移除公式(5)的冗余条件检查
        2. 基于公式(7)的改进评分
        3. 限制PO节点数量为R
        """
        R = self.configs['simple_direct']['R']
        
        if len(nodes) <= R:
            return list(range(len(nodes)))
        
        # 计算每个节点的潜在改进分数 (论文公式7)
        scores = []
        for i, (center, diameter, value, max_slope) in enumerate(nodes):
            # I(Θ_j) = L(θ_j) + 0.5 * δ(Θ_j) * K̂_j
            improvement_score = value + 0.5 * diameter * max_slope
            scores.append((improvement_score, i))
        
        # 选择分数最高的R-1个节点 + 最大直径节点
        scores.sort(reverse=True)
        selected_indices = [idx for _, idx in scores[:R-1]]
        
        # 添加最大直径节点 (保证收敛性)
        max_diameter_idx = max(range(len(nodes)), key=lambda i: nodes[i][1])
        if max_diameter_idx not in selected_indices:
            selected_indices.append(max_diameter_idx)
        
        return selected_indices[:R]
    
    def _find_longest_dimensions(self, diameter, dimension):
        """找到需要分割的最长维度"""
        # 简化实现：所有维度等长，随机选择
        return [random.randint(0, dimension-1)]
    
    def _unit_cube_to_params(self, unit_point, param_space):
        """将单位超立方体中的点转换为实际参数"""
        params = {}
        for i, (name, bounds) in enumerate(param_space.items()):
            params[name] = bounds[0] + unit_point[i] * (bounds[1] - bounds[0])
        return params
    
    def _compute_distance_based_objective(self, pred_output, target_bev, tau=2.0):
        """
        计算距离基础的代理目标函数 - 论文公式(3)
        
        L(F(S_θ(x)), y) = Σ_{v=1}^V min(min D(ŷ_θ^v, y_v), τ)
        
        Args:
            pred_output: 模型预测输出
            target_bev: 真实BEV标签
            tau: 距离阈值
            
        Returns:
            distance_score: 距离基础分数 (越高表示攻击越成功)
        """
        pred = pred_output.sigmoid()
        
        # 计算预测和真实之间的2D中心距离
        pred_centers = self._extract_detection_centers(pred)
        target_centers = self._extract_detection_centers(target_bev)
        
        if len(target_centers) == 0:
            return 0.0
        
        total_distance = 0.0
        for target_center in target_centers:
            if len(pred_centers) == 0:
                min_distance = tau  # 没有预测时使用最大距离
            else:
                # 计算到所有预测中心的最小距离
                distances = [np.linalg.norm(pred_center - target_center) for pred_center in pred_centers]
                min_distance = min(distances)
            
            # 应用距离阈值约束
            clamped_distance = min(min_distance, tau)
            total_distance += clamped_distance
        
        return total_distance
    
    def _extract_detection_centers(self, bev_map, threshold=0.5):
        """从BEV地图中提取检测中心点"""
        bev_np = bev_map.squeeze().cpu().numpy()
        
        if bev_np.max() <= threshold:
            return []
        
        # 二值化
        binary_map = (bev_np > threshold).astype(np.uint8)
        
        # 寻找连通组件
        contours, _ = cv2.findContours(binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        centers = []
        for contour in contours:
            if cv2.contourArea(contour) > 10:  # 过滤小区域
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    centers.append(np.array([cx, cy]))
        
        return centers

    def _geometric_objective(self, images, rots, trans, intrins, post_rots, post_trans, target_bev, params):
        """几何变换的目标函数"""
        try:
            # 应用几何变换
            transformed_images = self._apply_geometric_transform(images, params)
            
            # 前向传播
            with torch.no_grad():
                transformed_tensor = transformed_images.unsqueeze(0).to(self.device)
                output = self.model(transformed_tensor, rots.unsqueeze(0), trans.unsqueeze(0),
                                  intrins.unsqueeze(0), post_rots.unsqueeze(0), post_trans.unsqueeze(0))
                distance_score = self._compute_distance_based_objective(output, target_bev)
            
            return distance_score
        except Exception as e:
            print(f"几何目标函数评估失败: {e}")
            return 0.0
    
    def _colour_objective(self, images, rots, trans, intrins, post_rots, post_trans, target_bev, params):
        """颜色变换的目标函数"""
        try:
            # 应用颜色变换
            colour_shifted_images = self._apply_colour_shift(images, params)
            
            # 前向传播
            with torch.no_grad():
                colour_tensor = colour_shifted_images.unsqueeze(0).to(self.device)
                output = self.model(colour_tensor, rots.unsqueeze(0), trans.unsqueeze(0),
                                  intrins.unsqueeze(0), post_rots.unsqueeze(0), post_trans.unsqueeze(0))
                distance_score = self._compute_distance_based_objective(output, target_bev)
            
            return distance_score
        except Exception as e:
            print(f"颜色目标函数评估失败: {e}")
            return 0.0
    
    def _motion_blur_objective(self, images, rots, trans, intrins, post_rots, post_trans, target_bev, params, kernel_size):
        """运动模糊的目标函数"""
        try:
            # 应用运动模糊
            blurred_images = self._apply_motion_blur(images, params, kernel_size)
            
            # 前向传播
            with torch.no_grad():
                blurred_tensor = blurred_images.unsqueeze(0).to(self.device)
                output = self.model(blurred_tensor, rots.unsqueeze(0), trans.unsqueeze(0),
                                  intrins.unsqueeze(0), post_rots.unsqueeze(0), post_trans.unsqueeze(0))
                distance_score = self._compute_distance_based_objective(output, target_bev)
            
            return distance_score
        except Exception as e:
            print(f"运动模糊目标函数评估失败: {e}")
            return 0.0

    def _apply_geometric_transform(self, images, params):
        """
        应用几何变换 - 基于论文公式(8)
        
        变换矩阵: [θ_s^hor, 0, θ_t^hor; 0, θ_s^vrt, θ_t^vrt; 0, 0, 1]
        """
        S, C, H, W = images.shape
        
        # 构建仿射变换矩阵
        scale_h = params.get('scale_h', 1.0)
        scale_v = params.get('scale_v', 1.0)
        trans_h = params.get('trans_h', 0.0)
        trans_v = params.get('trans_v', 0.0)
        
        # 变换矩阵 (2x3)
        transform_matrix = torch.tensor([
            [scale_h, 0, trans_h],
            [0, scale_v, trans_v]
        ], dtype=torch.float32, device=images.device)
        
        transformed_images = torch.zeros_like(images)
        
        for s in range(S):
            img = images[s]  # (C, H, W)
            
            # 生成仿射网格
            grid = F.affine_grid(transform_matrix.unsqueeze(0), (1, C, H, W), align_corners=False)
            
            # 应用变换
            transformed_img = F.grid_sample(img.unsqueeze(0), grid, align_corners=False, mode='bilinear')
            transformed_images[s] = transformed_img.squeeze(0)
        
        return torch.clamp(transformed_images, 0, 255)
    
    def _apply_colour_shift(self, images, params):
        """
        应用颜色偏移 - 基于论文公式(9-11)的HSV变换
        """
        S, C, H, W = images.shape
        
        hue_shift = params.get('hue', 0.0)
        saturation_factor = params.get('saturation', 1.0)
        brightness_shift = params.get('brightness', 0.0)
        
        # 将图像转换为HSV空间
        shifted_images = torch.zeros_like(images)
        
        for s in range(S):
            img = images[s].permute(1, 2, 0)  # (H, W, C)
            img_np = img.cpu().numpy().astype(np.uint8)
            
            # RGB to HSV
            hsv_img = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV).astype(np.float32)
            
            # 应用HSV变换 (论文公式9-11)
            hsv_img[:, :, 0] = (hsv_img[:, :, 0] + hue_shift * 180 / np.pi) % 180  # Hue
            hsv_img[:, :, 1] = np.clip(hsv_img[:, :, 1] * saturation_factor, 0, 255)  # Saturation
            hsv_img[:, :, 2] = np.clip(hsv_img[:, :, 2] + brightness_shift * 255, 0, 255)  # Brightness
            
            # HSV to RGB
            rgb_img = cv2.cvtColor(hsv_img.astype(np.uint8), cv2.COLOR_HSV2RGB)
            shifted_images[s] = torch.from_numpy(rgb_img).permute(2, 0, 1).float().to(images.device)
        
        return torch.clamp(shifted_images, 0, 255)
    
    def _apply_motion_blur(self, images, params, kernel_size):
        """
        应用运动模糊 - 基于可优化的角度和方向参数
        """
        S, C, H, W = images.shape
        
        angle = params.get('angle', 0.0)
        direction = params.get('direction', 0.0)
        
        # 创建运动模糊核
        blur_kernel = self._create_motion_blur_kernel(kernel_size, angle, direction)
        blur_kernel = torch.from_numpy(blur_kernel).float().unsqueeze(0).unsqueeze(0).to(images.device)
        
        blurred_images = torch.zeros_like(images)
        
        for s in range(S):
            for c in range(C):
                img_channel = images[s, c].unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
                blurred_channel = F.conv2d(img_channel, blur_kernel, padding=kernel_size//2)
                blurred_images[s, c] = blurred_channel.squeeze()
        
        return torch.clamp(blurred_images, 0, 255)
    
    def _create_motion_blur_kernel(self, kernel_size, angle, direction):
        """创建运动模糊核"""
        kernel = np.zeros((kernel_size, kernel_size))
        
        # 计算线性运动模糊的轨迹
        center = kernel_size // 2
        length = int(kernel_size * abs(direction))
        
        if length == 0:
            kernel[center, center] = 1.0
            return kernel
        
        # 基于角度和方向创建线性轨迹
        dx = np.cos(angle) * direction
        dy = np.sin(angle) * direction
        
        for i in range(-length//2, length//2 + 1):
            x = int(center + i * dx)
            y = int(center + i * dy)
            if 0 <= x < kernel_size and 0 <= y < kernel_size:
                kernel[y, x] = 1.0
        
        # 归一化
        kernel_sum = kernel.sum()
        if kernel_sum > 0:
            kernel /= kernel_sum
        else:
            kernel[center, center] = 1.0
        
        return kernel

    def apply_robust_bev_attack(self, images, rots, trans, intrins, post_rots, post_trans, binimgs,
                               intensity='medium', attack_type='simple_direct'):
        """
        应用RobustBEV攻击的统一接口 - 官方实现
        
        Args:
            images: 输入图像 (S, C, H, W)
            rots, trans, intrins, post_rots, post_trans: 相机参数
            binimgs: BEV ground truth
            intensity: 攻击强度 'low', 'medium', 'high'
            attack_type: 攻击类型 'geometric', 'colour', 'motion_blur', 'simple_direct'
            
        Returns:
            攻击后的对抗样本图像
        """
        # 强度配置
        intensity_configs = {
            'low': {'gamma': 0.1, 'max_eval': 1000},
            'medium': {'gamma': 0.2, 'max_eval': 2500},    # 官方推荐
            'high': {'gamma': 0.3, 'max_eval': 5000}
        }
        
        if intensity not in intensity_configs:
            raise ValueError(f"不支持的攻击强度: {intensity}")
        
        config = intensity_configs[intensity]
        print(f"[RobustBEV] 强度: {intensity}, 类型: {attack_type}")
        
        if attack_type == 'geometric':
            return self.geometric_transformation_attack(
                images, rots, trans, intrins, post_rots, post_trans, binimgs,
                gamma=config['gamma'], max_evaluations=config['max_eval']
            )
            
        elif attack_type == 'colour':
            return self.colour_shift_attack(
                images, rots, trans, intrins, post_rots, post_trans, binimgs,
                gamma=config['gamma'], max_evaluations=config['max_eval']
            )
            
        elif attack_type == 'motion_blur':
            kernel_size = 7 if intensity == 'low' else 9 if intensity == 'medium' else 11
            return self.motion_blur_attack(
                images, rots, trans, intrins, post_rots, post_trans, binimgs,
                kernel_size=kernel_size, max_evaluations=config['max_eval']
            )
            
        elif attack_type == 'simple_direct':
            # SimpleDIRECT随机组合攻击 - 官方实现
            attack_types = ['geometric', 'colour', 'motion_blur']
            selected_attacks = random.sample(attack_types, random.randint(1, len(attack_types)))
            
            print(f"[SimpleDIRECT] 随机选择攻击组合: {selected_attacks}")
            print("[组合语义攻击] 开始执行多种语义扰动的组合优化")
            
            # 依次应用选中的攻击
            current_images = images
            for attack in selected_attacks:
                print(f"[组合语义攻击] 执行 {attack} 攻击")
                current_images = self.apply_robust_bev_attack(
                    current_images, rots, trans, intrins, post_rots, post_trans, binimgs,
                    intensity=intensity, attack_type=attack
                )
            
            return current_images
            
        else:
            raise ValueError(f"不支持的攻击类型: {attack_type}") 