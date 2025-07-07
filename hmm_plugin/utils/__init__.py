"""
HMM插件工具模块
包含各种辅助功能和工具类

主要组件：
1. ActionEncoder: 动作编码器
2. EgoMotionExtractor: 自车运动提取器
3. 其他辅助工具
"""

from .action_encoder import ActionEncoder, EgoMotionExtractor

__all__ = [
    'ActionEncoder',
    'EgoMotionExtractor'
] 