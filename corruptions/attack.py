import torch
import torch.nn.functional as F
import random
from typing import Optional, Callable


class AdversarialAttack:
    def __init__(self, model):
        self.model = model

    def fgsm_attack(self, images, rots, trans, intrins, post_rots, post_trans, binimgs,
                    epsilon=0.025):
        """
        简化的FGSM攻击,直接在原始图像空间操作
        """
        S, C, H, W = images.shape
        selected_indices = list(range(S))  # 攻击所有图像
        print(f"[FGSM攻击] 攻击图像索引: {selected_indices}, epsilon: {epsilon:.6f}")

        # 直接使用输入图像，需要梯度
        images_tensor = images.unsqueeze(0).clone().detach().requires_grad_(True)

        # 前向传播计算损失
        output = self.model(images_tensor, rots.unsqueeze(0), trans.unsqueeze(0),
                            intrins.unsqueeze(0), post_rots.unsqueeze(0), post_trans.unsqueeze(0))
        
        # 计算攻击损失
        loss_ce = F.binary_cross_entropy_with_logits(output, binimgs.unsqueeze(0).float())
        vehicle_mask = binimgs.unsqueeze(0).float()
        
        # 车辆区域抑制攻击
        loss_vehicle_suppress = (output.sigmoid() * vehicle_mask).mean()
        
        # 非车辆区域假阳性攻击
        non_vehicle_mask = 1.0 - vehicle_mask
        loss_false_positive = -(output.sigmoid() * non_vehicle_mask * 0.3).mean()
        
        # 预测混乱
        prob = output.sigmoid()
        target_confusion = 0.4 + 0.2 * torch.rand_like(prob)
        loss_confusion = -0.5 * F.mse_loss(prob, target_confusion)
        
        # 边界攻击
        if vehicle_mask.sum() > 0:
            kernel = torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], 
                                dtype=torch.float32, device=vehicle_mask.device).unsqueeze(0).unsqueeze(0)
            edge_mask = F.conv2d(vehicle_mask, kernel, padding=1).abs()
            edge_mask = (edge_mask > 0.2).float()
            loss_edge_attack = (output.sigmoid() * edge_mask).mean()
        else:
            loss_edge_attack = torch.tensor(0.0, device=output.device)
        
        # 组合损失
        total_loss = (2.0 * loss_ce + 
                     4.0 * loss_vehicle_suppress + 
                     2.0 * loss_false_positive + 
                     1.0 * loss_confusion + 
                     3.0 * loss_edge_attack)
        total_loss.backward()

        # 计算扰动
        perturbation = epsilon * images_tensor.grad.sign()

        # 生成对抗样本
        adv_images = images_tensor + perturbation
        adv_images = adv_images.squeeze(0)

        # 计算扰动统计
        diff = (adv_images - images).abs()
        max_diff = diff.max()
        mean_diff = diff.mean()
        
        print(f"[FGSM攻击] 扰动统计 - 最大: {max_diff:.6f}, 平均: {mean_diff:.6f}")
        
        return adv_images

    def iou_targeted_attack(self, images, rots, trans, intrins, post_rots, post_trans, binimgs,
                           epsilon=0.03, alpha=0.005, num_steps=20):
        """
        专门针对IoU的攻击方法，直接在原始图像空间操作
        """
        S, C, H, W = images.shape
        selected_indices = list(range(S))
        print(f"[IoU攻击] 攻击图像索引: {selected_indices}, epsilon: {epsilon:.6f}")

        images_tensor = images.unsqueeze(0).clone().detach()
        
        # 攻击所有图像
        mask = torch.ones_like(images_tensor)

        # 初始化随机扰动
        delta = (torch.rand_like(images_tensor) * 2 * epsilon - epsilon) * mask
        delta = delta.detach()

        best_delta = delta.clone()
        best_iou_loss = float('-inf')

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
            
            # 边界攻击
            if target.sum() > 0:
                kernel = torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], 
                                    dtype=torch.float32, device=target.device).unsqueeze(0).unsqueeze(0)
                edge_mask = F.conv2d(target, kernel, padding=1).abs()
                edge_mask = (edge_mask > 0.1).float()
                edge_confusion_loss = -(pred * edge_mask).mean()
            else:
                edge_confusion_loss = torch.tensor(0.0, device=output.device)
            
            # 形状攻击
            if target.sum() > 0:
                eroded_target = F.max_pool2d(-target, 3, stride=1, padding=1)
                eroded_target = -eroded_target
                dilated_target = F.max_pool2d(target, 3, stride=1, padding=1)
                
                center_suppress_loss = (pred * eroded_target).mean()
                periphery_enhance_loss = -(pred * (dilated_target - target) * 0.5).mean()
                shape_loss = center_suppress_loss + periphery_enhance_loss
            else:
                shape_loss = torch.tensor(0.0, device=output.device)
            
            # 组合损失
            total_loss = (3.0 * iou_loss + 
                         1.5 * edge_confusion_loss + 
                         2.0 * shape_loss)
            
            total_loss.backward()

            if delta.grad is not None:
                adaptive_alpha = alpha * (1.0 + step * 0.02)
                delta = delta + adaptive_alpha * delta.grad.sign() * mask
                delta = torch.clamp(delta, -epsilon, epsilon)
                
                # 记录最佳结果
                current_iou_loss = iou_loss.item()
                if current_iou_loss > best_iou_loss:
                    best_iou_loss = current_iou_loss
                    best_delta = delta.clone().detach()
            
            delta = delta.detach()
            self.model.zero_grad()

        # 使用最佳扰动
        print(f"[IoU攻击] 最佳IoU损失: {best_iou_loss:.6f}")
        
        final_adv_images = images_tensor + best_delta
        final_adv_images = final_adv_images.squeeze(0)

        # 计算统计信息
        diff = (final_adv_images - images).abs()
        max_diff = diff.max()
        mean_diff = diff.mean()
        
        print(f"[IoU攻击] 扰动统计 - 最大: {max_diff:.6f}, 平均: {mean_diff:.6f}")
        
        return final_adv_images

    def apply_random_attack(self, images, rots, trans, intrins, post_rots, post_trans, binimgs,
                             intensity) -> torch.Tensor:
        """
        应用随机攻击，简化版本
        """
        configs = {
            'low': {
                'method': 'fgsm',
                'epsilon': 0.015,
            },
            'medium': {
                'method': 'iou_targeted',
                'epsilon': 0.025,
                'alpha': 0.005,
                'steps': 15,
            },
            'high': {
                'method': 'iou_targeted',
                'epsilon': 0.035,
                'alpha': 0.007,
                'steps': 20,
            }
        }

        if intensity not in configs:
            raise ValueError(f"不支持的强度 '{intensity}'")

        cfg = configs[intensity]
        print(f"[攻击配置] 强度: {intensity}, 方法: {cfg['method']}")

        if cfg['method'] == 'fgsm':
            return self.fgsm_attack(images, rots, trans, intrins, post_rots, post_trans, binimgs, 
                                  epsilon=cfg['epsilon'])
        elif cfg['method'] == 'iou_targeted':
            return self.iou_targeted_attack(images, rots, trans, intrins, post_rots, post_trans, binimgs,
                                          epsilon=cfg['epsilon'], alpha=cfg['alpha'], num_steps=cfg['steps'])
        else:
            return self.pgd_attack(images, rots, trans, intrins, post_rots, post_trans, binimgs,
                                 epsilon=cfg['epsilon'], alpha=cfg['alpha'], num_steps=cfg['steps'])

    def pgd_attack(self, images, rots, trans, intrins, post_rots, post_trans, binimgs,
                   epsilon=0.025, alpha=0.005, num_steps=15):
        """
        简化的PGD攻击
        """
        S, C, H, W = images.shape
        selected_indices = random.sample(range(S), min(4, S))
        print(f"[PGD攻击] 攻击图像索引: {selected_indices}, epsilon: {epsilon:.6f}")

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
            
            # 使用相同的损失函数
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
            
            # 组合损失
            intensity_multiplier = 1.0 + step * 0.1
            total_loss = intensity_multiplier * (
                2.0 * loss_ce + 
                6.0 * loss_vehicle_suppress + 
                4.0 * loss_false_positive + 
                3.0 * loss_confusion + 
                5.0 * loss_edge_attack
            )
            total_loss.backward()

            if delta.grad is not None:
                adaptive_alpha = alpha * (1.0 + step * 0.05)
                delta = delta + adaptive_alpha * delta.grad.sign() * mask
                delta = torch.clamp(delta, -epsilon, epsilon)
            
            delta = delta.detach()
            self.model.zero_grad()

        # 生成最终对抗样本
        final_adv_images = images_tensor + delta
        final_adv_images = final_adv_images.squeeze(0)

        # 计算统计信息
        diff = (final_adv_images - images).abs()
        max_diff = diff.max()
        mean_diff = diff.mean()
        
        print(f"[PGD攻击] 扰动统计 - 最大: {max_diff:.6f}, 平均: {mean_diff:.6f}")
        
        return final_adv_images
