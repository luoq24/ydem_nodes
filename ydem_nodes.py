"""ydem_nodes - 自定义节点实现"""

import torch
import json
import numpy as np

# 导入comfyui_controlnet_aux的底层渲染逻辑
try:
    from comfyui_controlnet_aux.src.custom_controlnet_aux.dwpose import draw_poses, decode_json_as_poses
except ImportError:
    draw_poses = None
    decode_json_as_poses = None

class YDemPoseRenderer:
    """支持多帧输入的POSE_KEYPOINT渲染节点"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "kps": ("POSE_KEYPOINT",),
                "render_body": ("BOOLEAN", {"default": True}),
                "render_hand": ("BOOLEAN", {"default": True}),
                "render_face": ("BOOLEAN", {"default": True}),
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "render"
    CATEGORY = "ydem_nodes/pose"
    
    def render(self, kps, render_body, render_hand, render_face):
        """渲染POSE_KEYPOINT数据为图像"""
        if draw_poses is None or decode_json_as_poses is None:
            raise ImportError("comfyui_controlnet_aux 模块未安装或版本不兼容")
        
        # 如果输入是列表（多帧），遍历所有帧进行渲染
        if isinstance(kps, list):
            np_images = []
            for frame in kps:
                poses, _, height, width = decode_json_as_poses(frame)
                np_image = draw_poses(
                    poses,
                    height,
                    width,
                    render_body,
                    render_hand,
                    render_face,
                )
                np_images.append(np_image)
            # 堆叠所有帧 [B, H, W, C]
            np_images = np.stack(np_images, axis=0)
            return (torch.from_numpy(np_images.astype(np.float32) / 255),)
        else:
            poses, _, height, width = decode_json_as_poses(kps)
            np_image = draw_poses(
                poses,
                height,
                width,
                render_body,
                render_hand,
                render_face,
            )
            # 添加batch维度 [1, H, W, C]
            return (torch.from_numpy(np_image.astype(np.float32) / 255).unsqueeze(0),)


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
        
        # 创建过滤器实例并执行过滤
        filter_core = pose_filter_core.PoseFilterCore()
        result = filter_core.filter_pose(pose_keypoint, input_pose_order, input_pose_index)
        
        return (result,)

# 节点映射
NODE_CLASS_MAPPINGS = {
    "YDemPoseFilter": YDemPoseFilter,
    "YDemPoseRenderer": YDemPoseRenderer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "YDemPoseFilter": "YDem Pose Filter",
    "YDemPoseRenderer": "YDem Pose Renderer",
}
