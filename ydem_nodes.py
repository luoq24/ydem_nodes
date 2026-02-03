"""ydem_nodes - 自定义节点实现"""

import torch
import comfy.sd
from nodes import MAX_RESOLUTION
import numpy as np
import math

class YDemExampleNode:
    """示例节点"""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": "Hello World"}),
                "number": ("INT", {"default": 1, "min": 1, "max": 100}),
            },
            "optional": {
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("output_text", "output_number")
    FUNCTION = "process"
    CATEGORY = "ydem_nodes"

    def process(self, text, number, image=None):
        """处理输入"""
        output_text = f"{text} - Processed"
        output_number = number * 2
        return (output_text, output_number)

class YDemImageProcessor:
    """图像处理节点"""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "scale_factor": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 4.0, "step": 0.1}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = "ydem_nodes/image"

    def process(self, image, scale_factor):
        """处理图像"""
        # 简单的图像缩放示例
        import torchvision.transforms as transforms
        from torchvision.transforms import functional as F

        transformed_images = []
        for img in image:
            # 转换为PIL图像
            pil_img = F.to_pil_image(img)
            # 计算新尺寸
            new_size = (int(pil_img.width * scale_factor), int(pil_img.height * scale_factor))
            # 缩放
            resized_img = pil_img.resize(new_size)
            # 转换回张量
            transformed_img = F.to_tensor(resized_img)
            transformed_images.append(transformed_img)

        return (torch.stack(transformed_images),)

class YDemFaceDirection:
    """面部朝向检测节点"""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "faceanalysis": ("FACEANALYSIS",),
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("FLOAT", "STRING")
    RETURN_NAMES = ("angle", "direction")
    FUNCTION = "process"
    CATEGORY = "ydem_nodes/face"

    def process(self, faceanalysis, image):
        """检测面部朝向"""
        # 动态导入核心模块，支持热更新
        import importlib
        import sys
        
        # 移除已加载的模块，强制重新导入
        # 注意：使用完整的模块路径
        module_name = 'ydem_nodes.face_direction_core'
        if module_name in sys.modules:
            del sys.modules[module_name]
        if 'face_direction_core' in sys.modules:
            del sys.modules['face_direction_core']
        
        # 导入核心模块
        from . import face_direction_core
        importlib.reload(face_direction_core)
        
        # 打印调试信息
        print("=== YDemFaceDirection 热更新测试 ===")
        print(f"模块重新加载成功: {face_direction_core.__name__}")
        print(f"模块文件路径: {face_direction_core.__file__}")
        
        # 创建检测器实例并执行检测
        detector = face_direction_core.FaceDirectionDetector()
        result = detector.detect_direction(faceanalysis, image)
        print(f"检测结果: angle={result[0]}, direction={result[1]}")
        print("====================================")
        return result

# 节点映射
NODE_CLASS_MAPPINGS = {
    "YDemExampleNode": YDemExampleNode,
    "YDemImageProcessor": YDemImageProcessor,
    "YDemFaceDirection": YDemFaceDirection,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "YDemExampleNode": "YDem Example Node",
    "YDemImageProcessor": "YDem Image Processor",
    "YDemFaceDirection": "YDem Face Direction",
}
