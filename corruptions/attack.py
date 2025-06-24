import torch
import torch.nn.functional as F
import random
from typing import Optional, Callable


class AdversarialAttack:
    def __init__(self, model):
        self.model = model
        # 常见模型（如 ImageNet）使用的标准化参数
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)

    def normalize(self, images: torch.Tensor) -> torch.Tensor:
        """将 [0,255] 图像归一化到模型所需的分布"""
        images = images.float() / 255.0
        return (images - self.mean.to(images.device)) / self.std.to(images.device)

    def denormalize(self, images: torch.Tensor) -> torch.Tensor:
        """从模型分布反归一化回 [0,1] 再转 0~255"""
        images = images * self.std.to(images.device) + self.mean.to(images.device)
        images = torch.clamp(images, 0, 1)
        return (images * 255.0).round().byte()

    def denormalize_image(self, image: torch.Tensor) -> torch.Tensor:
        """将[0,1]范围的图像转换回[0,255]整数格式"""
        return torch.clamp(image * 255, 0, 255).byte()

    def post_process_for_imperceptibility(self, adv_images: torch.Tensor, orig_images: torch.Tensor, 
                                        method='adaptive_smooth') -> torch.Tensor:
        """
        攻击后处理：减少扰动的可见性同时保持攻击效果
        """
        if method == 'adaptive_smooth':
            return self._adaptive_smoothing(adv_images, orig_images)
        elif method == 'quantization':
            return self._smart_quantization(adv_images, orig_images)
        elif method == 'frequency_filter':
            return self._frequency_filtering(adv_images, orig_images)
        elif method == 'gradient_based':
            return self._gradient_based_refinement(adv_images, orig_images)
        else:
            return adv_images

    def _adaptive_smoothing(self, adv_images: torch.Tensor, orig_images: torch.Tensor) -> torch.Tensor:
        """自适应平滑：基于感知重要性的智能平滑"""
        # 计算扰动强度分布和感知重要性
        perturbation = (adv_images - orig_images).abs()
        
        # 计算感知重要性：结合对比度和纹理
        S, C, H, W = adv_images.shape
        smoothed_images = adv_images.clone()
        
        for s in range(S):
            for c in range(C):
                img = adv_images[s, c]
                orig = orig_images[s, c]
                pert = perturbation[s, c]
                
                # 计算局部对比度
                laplacian_kernel = torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], 
                                              dtype=torch.float32, device=img.device).unsqueeze(0).unsqueeze(0)
                contrast = F.conv2d(orig.unsqueeze(0).unsqueeze(0), laplacian_kernel, padding=1).abs().squeeze()
                
                # 计算局部标准差
                mean_kernel = torch.ones(5, 5, device=img.device) / 25.0
                local_mean = F.conv2d(orig.unsqueeze(0).unsqueeze(0), 
                                    mean_kernel.unsqueeze(0).unsqueeze(0), padding=2).squeeze()
                local_std = F.conv2d((orig.unsqueeze(0).unsqueeze(0) - local_mean.unsqueeze(0).unsqueeze(0))**2,
                                   mean_kernel.unsqueeze(0).unsqueeze(0), padding=2).squeeze().sqrt()
                
                # 计算感知重要性：低对比度和低纹理区域更重要（更容易被感知到扰动）
                perceptual_importance = 1.0 / (contrast + local_std + 1e-8)
                perceptual_importance = perceptual_importance / perceptual_importance.max()
                
                # 自适应平滑强度：感知重要性高的区域平滑更多
                smooth_strength = perceptual_importance * 0.7  # 最多70%平滑
                
                # 应用高斯模糊
                gaussian_kernel = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]], 
                                             device=img.device, dtype=torch.float32) / 16.0
                smoothed = F.conv2d(img.unsqueeze(0).unsqueeze(0), 
                                  gaussian_kernel.unsqueeze(0).unsqueeze(0), padding=1).squeeze()
                
                # 混合原图和平滑图
                smoothed_images[s, c] = img * (1 - smooth_strength) + smoothed * smooth_strength
        
        return smoothed_images

    def _smart_quantization(self, adv_images: torch.Tensor, orig_images: torch.Tensor) -> torch.Tensor:
        """智能量化：基于感知模型的精确量化"""
        # 将图像转换到[0,1]空间
        orig_01 = orig_images * self.std.to(orig_images.device) + self.mean.to(orig_images.device)
        adv_01 = adv_images * self.std.to(adv_images.device) + self.mean.to(adv_images.device)
        
        perturbation = adv_01 - orig_01
        
        # 基于Just Noticeable Difference (JND)模型的量化
        S, C, H, W = perturbation.shape
        quantized_perturbation = torch.zeros_like(perturbation)
        
        for s in range(S):
            for c in range(C):
                pert = perturbation[s, c]
                orig = orig_01[s, c]
                
                # 计算Weber对比度：在不同亮度区域有不同的感知阈值
                # 明亮区域能容忍更大的绝对变化，暗区域对小变化更敏感
                weber_threshold = torch.clamp(orig * 0.02 + 0.005, 0.005, 0.03)  # 2%-3%的Weber阈值
                
                # 基于Weber阈值的自适应量化
                normalized_pert = pert / weber_threshold
                
                # 不同区域使用不同的量化精度
                high_precision_mask = normalized_pert.abs() > 2.0  # 超过2倍阈值需要高精度
                medium_precision_mask = (normalized_pert.abs() > 1.0) & (~high_precision_mask)
                
                # 高精度区域：256级量化
                quantized_perturbation[s, c][high_precision_mask] = (
                    torch.round(pert[high_precision_mask] * 256) / 256
                )
                
                # 中精度区域：128级量化
                quantized_perturbation[s, c][medium_precision_mask] = (
                    torch.round(pert[medium_precision_mask] * 128) / 128
                )
                
                # 低精度区域：64级量化
                low_precision_mask = ~(high_precision_mask | medium_precision_mask)
                quantized_perturbation[s, c][low_precision_mask] = (
                    torch.round(pert[low_precision_mask] * 64) / 64
                )
        
        # 重新组合
        quantized_adv_01 = orig_01 + quantized_perturbation
        quantized_adv_01 = torch.clamp(quantized_adv_01, 0, 1)
        
        # 转换回归一化空间
        quantized_adv = (quantized_adv_01 - self.mean.to(adv_images.device)) / self.std.to(adv_images.device)
        
        return quantized_adv

    def _frequency_filtering(self, adv_images: torch.Tensor, orig_images: torch.Tensor) -> torch.Tensor:
        """频域滤波：基于人眼视觉敏感度的频域处理"""
        filtered_images = adv_images.clone()
        S, C, H, W = adv_images.shape
        
        for s in range(S):
            for c in range(C):
                adv_img = adv_images[s, c]
                orig_img = orig_images[s, c]
                
                # 计算扰动的FFT
                perturbation = adv_img - orig_img
                
                # 转换到频域
                fft_pert = torch.fft.fft2(perturbation)
                fft_magnitude = fft_pert.abs()
                fft_phase = torch.angle(fft_pert)
                
                # 创建基于人眼敏感度的滤波器
                # 人眼对中频最敏感，对高频和极低频相对不敏感
                h, w = H, W
                u = torch.arange(h, device=adv_img.device).float() - h // 2
                v = torch.arange(w, device=adv_img.device).float() - w // 2
                U, V = torch.meshgrid(u, v, indexing='ij')
                D = torch.sqrt(U**2 + V**2)
                D = torch.fft.fftshift(D)  # 移到正确位置
                
                # 人眼对比敏感函数的近似
                # 在中频(约0.1-0.3 cycles/pixel)最敏感
                normalized_freq = D / (min(h, w) / 2)
                csf_weight = torch.exp(-((normalized_freq - 0.2) / 0.15)**2)  # 高斯形状的CSF
                
                # 在人眼敏感的频率处进行更强的滤波
                filter_strength = 0.3 * csf_weight  # 最多30%的滤波
                filtered_magnitude = fft_magnitude * (1 - filter_strength)
                
                # 重构信号
                filtered_fft = filtered_magnitude * torch.exp(1j * fft_phase)
                filtered_perturbation = torch.fft.ifft2(filtered_fft).real
                
                filtered_images[s, c] = orig_img + filtered_perturbation
        
        return filtered_images

    def _gradient_based_refinement(self, adv_images: torch.Tensor, orig_images: torch.Tensor) -> torch.Tensor:
        """基于梯度的细化：保持攻击关键区域，精细化非关键区域"""
        S, C, H, W = adv_images.shape
        refined_images = adv_images.clone()
        
        # 计算多尺度梯度信息
        def compute_multiscale_gradient(img):
            gradients = []
            for kernel_size in [3, 5, 7]:
                # Sobel算子
                sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                                     dtype=torch.float32, device=img.device).unsqueeze(0).unsqueeze(0)
                sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                                     dtype=torch.float32, device=img.device).unsqueeze(0).unsqueeze(0)
                
                if kernel_size > 3:
                    # 对大核进行归一化
                    sobel_x = F.interpolate(sobel_x, size=(kernel_size, kernel_size), mode='bilinear')
                    sobel_y = F.interpolate(sobel_y, size=(kernel_size, kernel_size), mode='bilinear')
                    pad = kernel_size // 2
                else:
                    pad = 1
                
                grad_x = F.conv2d(img.unsqueeze(0).unsqueeze(0), sobel_x, padding=pad).abs().squeeze()
                grad_y = F.conv2d(img.unsqueeze(0).unsqueeze(0), sobel_y, padding=pad).abs().squeeze()
                gradient_magnitude = torch.sqrt(grad_x**2 + grad_y**2)
                gradients.append(gradient_magnitude)
            
            # 组合多尺度梯度
            combined_gradient = torch.stack(gradients).mean(dim=0)
            return combined_gradient / (combined_gradient.max() + 1e-8)
        
        for s in range(S):
            for c in range(C):
                orig_img = orig_images[s, c]
                adv_img = adv_images[s, c]
                
                # 计算原图和扰动的梯度重要性
                orig_gradient = compute_multiscale_gradient(orig_img)
                perturbation = adv_img - orig_img
                pert_gradient = compute_multiscale_gradient(perturbation.abs())
                
                # 计算综合重要性：原图梯度高或扰动梯度高的区域都重要
                importance = torch.max(orig_gradient, pert_gradient * 0.7)
                
                # 在非重要区域进行细化
                # 使用sigmoid函数创建平滑的重要性掩码
                smooth_importance = torch.sigmoid((importance - 0.5) * 10)  # 陡峭的sigmoid
                
                # 对非重要区域应用约束
                constraint_strength = (1 - smooth_importance) * 0.4  # 最多40%约束
                
                refined_perturbation = perturbation * (1 - constraint_strength)
                refined_images[s, c] = orig_img + refined_perturbation
        
        return refined_images

    def fgsm_attack(self, images, rots, trans, intrins, post_rots, post_trans, binimgs,
                    epsilon=0.025, post_process='adaptive_smooth'):
        """
        FGSM攻击 + 强化后处理实现真正imperceptible效果
        """
        S, C, H, W = images.shape
        selected_indices = list(range(S))  # 攻击所有图像以最大化效果
        print(f"[Imperceptible FGSM] 攻击图像索引: {selected_indices}, epsilon: {epsilon:.6f}, 后处理: {post_process}")

        # 输入images已经是归一化后的张量，直接使用
        images_tensor = images.unsqueeze(0).clone().detach().requires_grad_(True)

        # 前向传播计算损失
        output = self.model(images_tensor, rots.unsqueeze(0), trans.unsqueeze(0),
                            intrins.unsqueeze(0), post_rots.unsqueeze(0), post_trans.unsqueeze(0))
        
        # 使用智能攻击损失 - 重点攻击但控制可见性
        loss_ce = F.binary_cross_entropy_with_logits(output, binimgs.unsqueeze(0).float())
        vehicle_mask = binimgs.unsqueeze(0).float()
        
        # 1. 车辆区域智能抑制攻击
        loss_vehicle_suppress = (output.sigmoid() * vehicle_mask).mean()
        
        # 2. 控制的假阳性攻击 - 减少强度以避免可见变化
        non_vehicle_mask = 1.0 - vehicle_mask
        loss_false_positive = -(output.sigmoid() * non_vehicle_mask * 0.3).mean()  # 降低到0.3
        
        # 3. 温和的预测混乱
        prob = output.sigmoid()
        # 让预测向0.4-0.6范围混乱，而不是强制0.5
        target_confusion = 0.4 + 0.2 * torch.rand_like(prob)  # 直接使用rand_like生成0.4-0.6范围的随机值
        loss_confusion = -0.5 * F.mse_loss(prob, target_confusion)  # 减少权重
        
        # 4. 精确的边界攻击 - 只攻击真正的边界
        if vehicle_mask.sum() > 0:
            # 使用更精确的边界检测
            kernel = torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], 
                                dtype=torch.float32, device=vehicle_mask.device).unsqueeze(0).unsqueeze(0)
            edge_mask = F.conv2d(vehicle_mask, kernel, padding=1).abs()
            edge_mask = (edge_mask > 0.2).float()  # 更严格的边界定义
            loss_edge_attack = (output.sigmoid() * edge_mask).mean()
        else:
            loss_edge_attack = torch.tensor(0.0, device=output.device)
        
        # 组合损失，平衡攻击效果和可见性
        total_loss = (2.0 * loss_ce + 
                     4.0 * loss_vehicle_suppress + 
                     2.0 * loss_false_positive + 
                     1.0 * loss_confusion + 
                     3.0 * loss_edge_attack)
        total_loss.backward()

        # 计算扰动并应用感知约束
        raw_perturbation = epsilon * images_tensor.grad.sign()
        
        # 应用感知加权：在纹理复杂区域允许更大扰动
        texture_weights = self._compute_texture_weights(images)
        perturbation = raw_perturbation * texture_weights.unsqueeze(0)

        # 生成初步对抗样本
        raw_adv_images = images_tensor + perturbation
        raw_adv_images = raw_adv_images.squeeze(0)

        # 应用强化的后处理
        processed_adv_images = self.post_process_for_imperceptibility(
            raw_adv_images, images, method=post_process)

        # 进一步的感知优化
        final_adv_images = self._perceptual_refinement(processed_adv_images, images)

        # 计算扰动统计
        diff = (final_adv_images - images).abs()
        max_diff = diff.max()
        mean_diff = diff.mean()
        
        # 转换到像素级差异
        orig_img_01 = images * self.std.to(images.device) + self.mean.to(images.device)
        adv_img_01 = final_adv_images * self.std.to(final_adv_images.device) + self.mean.to(final_adv_images.device)
        pixel_diff = (adv_img_01 - orig_img_01).abs() * 255
        max_pixel_diff = pixel_diff.max()
        mean_pixel_diff = pixel_diff.mean()
        
        print(f"[Imperceptible FGSM] 归一化空间差异 - 最大: {max_diff:.6f}, 平均: {mean_diff:.6f}")
        print(f"[Imperceptible FGSM] 像素级差异 - 最大: {max_pixel_diff:.2f}/255, 平均: {mean_pixel_diff:.2f}/255")
        
        return final_adv_images

    def _compute_texture_weights(self, images: torch.Tensor) -> torch.Tensor:
        """计算纹理复杂度权重，在纹理复杂区域允许更大扰动"""
        S, C, H, W = images.shape
        weights = torch.ones((S, C, H, W), device=images.device)
        
        for s in range(S):
            for c in range(C):
                img = images[s, c]
                
                # 计算局部方差作为纹理复杂度指标
                kernel = torch.ones(5, 5, device=img.device) / 25.0
                local_mean = F.conv2d(img.unsqueeze(0).unsqueeze(0), 
                                    kernel.unsqueeze(0).unsqueeze(0), padding=2).squeeze()
                local_variance = F.conv2d((img.unsqueeze(0).unsqueeze(0) - local_mean.unsqueeze(0).unsqueeze(0))**2,
                                        kernel.unsqueeze(0).unsqueeze(0), padding=2).squeeze()
                
                # 归一化方差到[0.5, 1.5]范围
                normalized_variance = local_variance / (local_variance.max() + 1e-8)
                texture_weight = 0.5 + normalized_variance
                weights[s, c] = texture_weight
        
        return weights

    def _perceptual_refinement(self, adv_images: torch.Tensor, orig_images: torch.Tensor) -> torch.Tensor:
        """感知优化：进一步减少可见差异"""
        S, C, H, W = adv_images.shape
        refined_images = adv_images.clone()
        
        for s in range(S):
            for c in range(C):
                adv_img = adv_images[s, c]
                orig_img = orig_images[s, c]
                
                # 计算局部对比度
                laplacian_kernel = torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], 
                                              dtype=torch.float32, device=adv_img.device).unsqueeze(0).unsqueeze(0)
                
                orig_contrast = F.conv2d(orig_img.unsqueeze(0).unsqueeze(0), laplacian_kernel, padding=1).abs().squeeze()
                adv_contrast = F.conv2d(adv_img.unsqueeze(0).unsqueeze(0), laplacian_kernel, padding=1).abs().squeeze()
                
                # 在低对比度区域减少扰动
                low_contrast_mask = orig_contrast < orig_contrast.quantile(0.3)
                diff = adv_img - orig_img
                
                # 对低对比度区域应用更强的约束
                reduction_factor = torch.where(low_contrast_mask, 0.3, 0.0)  # 在低对比度区域减少30%扰动
                refined_diff = diff * (1 - reduction_factor)
                refined_images[s, c] = orig_img + refined_diff
        
        return refined_images

    def iou_targeted_attack(self, images, rots, trans, intrins, post_rots, post_trans, binimgs,
                           epsilon=0.03, alpha=0.005, num_steps=20, post_process='gradient_based'):
        """
        专门针对IoU的攻击方法 - 智能版本实现imperceptible + 有效攻击
        """
        S, C, H, W = images.shape
        selected_indices = list(range(S))  # 攻击所有图像
        print(f"[Imperceptible IoU攻击] 攻击图像索引: {selected_indices}, epsilon: {epsilon:.6f}")

        images_tensor = images.unsqueeze(0).clone().detach()
        
        # 攻击所有图像
        mask = torch.ones_like(images_tensor)

        # 初始化随机扰动
        delta = (torch.rand_like(images_tensor) * 2 * epsilon - epsilon) * mask
        delta = delta.detach()

        best_delta = delta.clone()
        best_iou_loss = float('-inf')
        prev_grad_sign = None

        # 计算感知权重
        texture_weights = self._compute_texture_weights(images).unsqueeze(0)

        for step in range(num_steps):
            delta.requires_grad_(True)
            
            output = self.model(images_tensor + delta, rots.unsqueeze(0), trans.unsqueeze(0),
                                intrins.unsqueeze(0), post_rots.unsqueeze(0), post_trans.unsqueeze(0))
            
            pred = output.sigmoid()
            target = binimgs.unsqueeze(0).float()
            
            # 计算IoU损失
            intersection = (pred * target).sum(dim=(2, 3))
            union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) - intersection
            iou = intersection / (union + 1e-8)
            iou_loss = -iou.mean()
            
            # 智能的攻击损失项
            # 1. 控制的边界攻击
            if target.sum() > 0:
                kernel = torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], 
                                    dtype=torch.float32, device=target.device).unsqueeze(0).unsqueeze(0)
                edge_mask = F.conv2d(target, kernel, padding=1).abs()
                edge_mask = (edge_mask > 0.1).float()
                edge_confusion_loss = -(pred * edge_mask).mean()
            else:
                edge_confusion_loss = torch.tensor(0.0, device=output.device)
            
            # 2. 温和的形状攻击
            if target.sum() > 0:
                eroded_target = F.max_pool2d(-target, 3, stride=1, padding=1)
                eroded_target = -eroded_target
                dilated_target = F.max_pool2d(target, 3, stride=1, padding=1)
                
                center_suppress_loss = (pred * eroded_target).mean()
                periphery_enhance_loss = -(pred * (dilated_target - target) * 0.5).mean()  # 减少强度
                shape_loss = center_suppress_loss + periphery_enhance_loss
            else:
                shape_loss = torch.tensor(0.0, device=output.device)
            
            # 组合损失 - 平衡效果和可见性
            total_loss = (3.0 * iou_loss + 
                         1.5 * edge_confusion_loss + 
                         2.0 * shape_loss)
            
            total_loss.backward()

            if delta.grad is not None:
                # 应用感知约束的梯度更新
                momentum = 0.9 if step > 0 else 0.0
                adaptive_alpha = alpha * (1.0 + step * 0.02)
                
                # 结合纹理权重和梯度
                weighted_grad = delta.grad.sign() * texture_weights
                
                if step > 0 and prev_grad_sign is not None:
                    delta = delta + adaptive_alpha * (weighted_grad + momentum * prev_grad_sign) * mask
                else:
                    delta = delta + adaptive_alpha * weighted_grad * mask
                
                prev_grad_sign = weighted_grad.clone()
                delta = torch.clamp(delta, -epsilon, epsilon)
                
                # 记录最佳结果
                current_iou_loss = iou_loss.item()
                if current_iou_loss > best_iou_loss:
                    best_iou_loss = current_iou_loss
                    best_delta = delta.clone().detach()
            
            delta = delta.detach()
            self.model.zero_grad()

        # 使用最佳扰动
        print(f"[Imperceptible IoU攻击] 最佳IoU损失: {best_iou_loss:.6f}")
        
        # 生成初步对抗样本
        raw_adv_images = images_tensor + best_delta
        raw_adv_images = raw_adv_images.squeeze(0)

        # 应用强化的后处理
        processed_adv_images = self.post_process_for_imperceptibility(
            raw_adv_images, images, method=post_process)

        # 进一步的感知优化
        final_adv_images = self._perceptual_refinement(processed_adv_images, images)

        # 计算统计信息
        diff = (final_adv_images - images).abs()
        max_diff = diff.max()
        mean_diff = diff.mean()
        
        orig_img_01 = images * self.std.to(images.device) + self.mean.to(images.device)
        adv_img_01 = final_adv_images * self.std.to(final_adv_images.device) + self.mean.to(final_adv_images.device)
        pixel_diff = (adv_img_01 - orig_img_01).abs() * 255
        max_pixel_diff = pixel_diff.max()
        mean_pixel_diff = pixel_diff.mean()
        
        print(f"[Imperceptible IoU攻击] 归一化空间差异 - 最大: {max_diff:.6f}, 平均: {mean_diff:.6f}")
        print(f"[Imperceptible IoU攻击] 像素级差异 - 最大: {max_pixel_diff:.2f}/255, 平均: {mean_pixel_diff:.2f}/255")
        
        return final_adv_images

    def apply_random_attack(self, images, rots, trans, intrins, post_rots, post_trans, binimgs,
                             intensity) -> torch.Tensor:
        """
        应用随机攻击，平衡版本以实现imperceptible + 有效的IoU下降
        """
        # 调整参数，平衡攻击效果和可见性
        configs = {
            'low': {
                'method': 'fgsm',
                'epsilon': 0.015,  # 适度扰动
                'post_process': 'adaptive_smooth'
            },
            'medium': {
                'method': 'iou_targeted',  # 使用IoU专门攻击
                'epsilon': 0.025,  # 平衡的扰动
                'alpha': 0.005,    # 温和的步长
                'steps': 15,       # 适度迭代
                'post_process': 'gradient_based'
            },
            'high': {
                'method': 'iou_targeted',  # 使用IoU专门攻击
                'epsilon': 0.035,  # 较大扰动但有控制
                'alpha': 0.007,    # 较大步长
                'steps': 20,       # 更多迭代
                'post_process': 'quantization'
            }
        }

        if intensity not in configs:
            raise ValueError(f"不支持的强度 '{intensity}'")

        cfg = configs[intensity]
        print(f"[Imperceptible攻击配置] 强度: {intensity}, 方法: {cfg['method']}, 后处理: {cfg['post_process']}")

        if cfg['method'] == 'fgsm':
            return self.fgsm_attack(images, rots, trans, intrins, post_rots, post_trans, binimgs, 
                                  epsilon=cfg['epsilon'], post_process=cfg['post_process'])
        elif cfg['method'] == 'iou_targeted':
            return self.iou_targeted_attack(images, rots, trans, intrins, post_rots, post_trans, binimgs,
                                          epsilon=cfg['epsilon'], alpha=cfg['alpha'], num_steps=cfg['steps'],
                                          post_process=cfg['post_process'])
        else:  # pgd
            return self.pgd_attack(images, rots, trans, intrins, post_rots, post_trans, binimgs,
                                 epsilon=cfg['epsilon'], alpha=cfg['alpha'], num_steps=cfg['steps'],
                                 post_process=cfg['post_process'])

    def pgd_attack(self, images, rots, trans, intrins, post_rots, post_trans, binimgs,
                   epsilon=0.025, alpha=0.005, num_steps=15, post_process='gradient_based'):
        """
        PGD攻击 + 后处理 - 增强版本
        """
        S, C, H, W = images.shape
        selected_indices = random.sample(range(S), min(4, S))  # 攻击更多图像
        print(f"[PGD+后处理] 攻击图像索引: {selected_indices}, epsilon: {epsilon:.6f}, 后处理: {post_process}")

        images_tensor = images.unsqueeze(0).clone().detach()
        
        # 创建掩码
        mask = torch.zeros((1, S, 1, 1, 1), device=images.device)
        for idx in selected_indices:
            mask[:, idx] = 1.0
        mask = mask.expand_as(images_tensor)

        # 初始化随机扰动
        delta = (torch.rand_like(images_tensor) * 2 * epsilon - epsilon) * mask
        delta = delta.detach()

        for step in range(num_steps):
            delta.requires_grad_(True)
            
            output = self.model(images_tensor + delta, rots.unsqueeze(0), trans.unsqueeze(0),
                                intrins.unsqueeze(0), post_rots.unsqueeze(0), post_trans.unsqueeze(0))
            
            # 使用相同的强化损失函数
            loss_ce = F.binary_cross_entropy_with_logits(output, binimgs.unsqueeze(0).float())
            vehicle_mask = binimgs.unsqueeze(0).float()
            
            loss_vehicle_suppress = (output.sigmoid() * vehicle_mask).mean()
            non_vehicle_mask = 1.0 - vehicle_mask
            loss_false_positive = -(output.sigmoid() * non_vehicle_mask * 0.3).mean()
            
            prob = output.sigmoid()
            target_confusion = torch.full_like(prob, 0.5)
            loss_confusion = -F.mse_loss(prob, target_confusion)
            
            # 边界攻击
            if vehicle_mask.sum() > 0:
                kernel = torch.tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], 
                                    dtype=torch.float32, device=vehicle_mask.device).unsqueeze(0).unsqueeze(0)
                edge_mask = F.conv2d(vehicle_mask, kernel, padding=1).abs()
                edge_mask = (edge_mask > 0.1).float()
                loss_edge_attack = (output.sigmoid() * edge_mask).mean()
            else:
                loss_edge_attack = torch.tensor(0.0, device=output.device)
            
            # 组合损失 - 逐步增强攻击强度
            intensity_multiplier = 1.0 + step * 0.1  # 随步数增加强度
            total_loss = intensity_multiplier * (
                2.0 * loss_ce + 
                6.0 * loss_vehicle_suppress + 
                4.0 * loss_false_positive + 
                3.0 * loss_confusion + 
                5.0 * loss_edge_attack
            )
            total_loss.backward()

            if delta.grad is not None:
                # 使用adaptive步长
                adaptive_alpha = alpha * (1.0 + step * 0.05)  # 逐步增加步长
                delta = delta + adaptive_alpha * delta.grad.sign() * mask
                delta = torch.clamp(delta, -epsilon, epsilon)
            
            delta = delta.detach()
            self.model.zero_grad()

        # 生成初步对抗样本
        raw_adv_images = images_tensor + delta
        raw_adv_images = raw_adv_images.squeeze(0)

        # 应用后处理
        processed_adv_images = self.post_process_for_imperceptibility(
            raw_adv_images, images, method=post_process)

        # 计算统计信息
        diff = (processed_adv_images - images).abs()
        max_diff = diff.max()
        mean_diff = diff.mean()
        
        orig_img_01 = images * self.std.to(images.device) + self.mean.to(images.device)
        adv_img_01 = processed_adv_images * self.std.to(processed_adv_images.device) + self.mean.to(processed_adv_images.device)
        pixel_diff = (adv_img_01 - orig_img_01).abs() * 255
        max_pixel_diff = pixel_diff.max()
        mean_pixel_diff = pixel_diff.mean()
        
        print(f"[PGD+后处理] 归一化空间差异 - 最大: {max_diff:.6f}, 平均: {mean_diff:.6f}")
        print(f"[PGD+后处理] 像素级差异 - 最大: {max_pixel_diff:.2f}/255 ({max_pixel_diff/255*100:.2f}%), 平均: {mean_pixel_diff:.2f}/255")
        
        return processed_adv_images
