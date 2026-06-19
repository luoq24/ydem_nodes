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
                "input_pose_index": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
                "force_reset_frames": ("STRING", {"default": ""}),
            },
        }
    
    RETURN_TYPES = ("POSE_KEYPOINT",)
    RETURN_NAMES = ("pose_keypoint",)
    FUNCTION = "process"
    CATEGORY = "ydem_nodes/pose"
    
    def process(self, pose_keypoint, input_pose_order, input_pose_index, force_reset_frames=""):
        """过滤POSE_KEYPOINT，仅保留指定人物"""
        from . import pose_filter_core
        
        filter_core = pose_filter_core.PoseFilterCore()
        result = filter_core.filter_pose(pose_keypoint, input_pose_order, input_pose_index, force_reset_frames)
        
        return (result,)


class YDemPoseRemoveHand:
    """移除POSE_KEYPOINT指定帧的手部关键点"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pose_keypoint": ("POSE_KEYPOINT",),
                "left_hand": ("BOOLEAN", {"default": False}),
                "right_hand": ("BOOLEAN", {"default": False}),
                "frames": ("STRING", {"default": ""}),
            },
        }
    
    RETURN_TYPES = ("POSE_KEYPOINT",)
    RETURN_NAMES = ("pose_keypoint",)
    FUNCTION = "process"
    CATEGORY = "ydem_nodes/pose"
    
    def process(self, pose_keypoint, left_hand, right_hand, frames):
        """移除指定帧的手部关键点"""
        from . import pose_filter_core
        
        remove_core = pose_filter_core.PoseHandRemoveCore()
        result = remove_core.remove_hands(pose_keypoint, left_hand, right_hand, frames)
        
        return (result,)

class YDemLoadTextFile:
    """读取本地文本文件，输出 STRING 和 ANY（可连接 QwenEditOutputExtractor）"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file_path": ("STRING", {"default": "", "multiline": False}),
            },
        }

    RETURN_TYPES = ("STRING", "ANY")
    RETURN_NAMES = ("text_content", "data")
    FUNCTION = "load_file"
    CATEGORY = "ydem_nodes/io"

    def load_file(self, file_path=""):
        if not file_path:
            raise ValueError("file_path 不能为空")

        with open(file_path, "r", encoding="utf-8") as f:
            text_content = f.read()

        # 尝试按 JSON 解析，失败则原样返回
        try:
            data = json.loads(text_content)
        except (json.JSONDecodeError, ValueError):
            data = text_content

        return (text_content, data)


# 节点映射
NODE_CLASS_MAPPINGS = {
    "YDemPoseFilter": YDemPoseFilter,
    "YDemPoseRenderer": YDemPoseRenderer,
    "YDemPoseRemoveHand": YDemPoseRemoveHand,
    "YDemLoadTextFile": YDemLoadTextFile,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "YDemPoseFilter": "YDem Pose Filter",
    "YDemPoseRenderer": "YDem Pose Renderer",
    "YDemPoseRemoveHand": "YDem Pose Remove Hand",
    "YDemLoadTextFile": "YDem Load Text File",
}
