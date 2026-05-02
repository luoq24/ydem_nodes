"""ydem_nodes - 自定义节点实现"""

import torch
import json

class YDemPoseFilter:
    """POSE_KEYPOINT人物过滤节点"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pose_keypoint": ("POSE_KEYPOINT",),
                "input_pose_order": (
                    ["left-right", "right-left", "top-bottom", "bottom-top", "small-large", "large-small"], 
                    {"default": "large-small"}
                ),
                "input_pose_index": ("STRING", {"default": "0"}),
            },
        }
    
    RETURN_TYPES = ("POSE_KEYPOINT",)
    RETURN_NAMES = ("pose_keypoint",)
    FUNCTION = "process"
    CATEGORY = "ydem_nodes/pose"
    
    def process(self, pose_keypoint, input_pose_order, input_pose_index):
        """过滤POSE_KEYPOINT，仅保留指定人物"""
        # 动态导入核心模块，支持开发阶段热更新
        import importlib
        import sys
        
        # 移除已加载的模块，强制重新导入
        module_name = 'ydem_nodes.pose_filter_core'
        if module_name in sys.modules:
            del sys.modules[module_name]
        if 'pose_filter_core' in sys.modules:
            del sys.modules['pose_filter_core']
        
        # 导入核心模块
        from . import pose_filter_core
        importlib.reload(pose_filter_core)
        
        # 打印调试信息
        print("=== YDemPoseFilter 热更新测试 ===")
        print(f"模块重新加载成功: {pose_filter_core.__name__}")
        print(f"模块文件路径: {pose_filter_core.__file__}")
        
        # 创建过滤器实例并执行过滤
        filter_core = pose_filter_core.PoseFilterCore()
        result = filter_core.filter_pose(pose_keypoint, input_pose_order, input_pose_index)
        
        return (result,)

# 节点映射
NODE_CLASS_MAPPINGS = {
    "YDemPoseFilter": YDemPoseFilter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "YDemPoseFilter": "YDem Pose Filter",
}
