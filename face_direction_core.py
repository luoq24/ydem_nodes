"""面部朝向检测核心逻辑"""

import numpy as np
import math

class FaceDirectionDetector:
    """面部朝向检测器"""
    
    def detect_direction(self, faceanalysis, image):
        print('aaaa')
        """检测面部朝向"""
        # 将IMAGE张量转换为numpy数组
        def tensor_to_image(tensor):
            """将张量转换为图像"""
            if len(tensor.shape) == 4:
                tensor = tensor[0]
            tensor = tensor * 255
            tensor = tensor.clamp(0, 255)
            return tensor.cpu().numpy().astype(np.uint8)

        # 处理图像
        face_img = tensor_to_image(image)
        
        # 使用insightface检测面部
        faceanalysis.det_model.input_size = (640, 640)
        faces = faceanalysis.get(face_img)
        
        if not faces:
            return (0.0, "No face detected")
        
        # 获取第一个面部
        face = faces[0]
        
        # 获取面部关键点
        kps = face['kps']
        
        # 关键点索引：0=左眼，1=右眼，2=鼻子，3=左嘴角，4=右嘴角
        left_eye = kps[0]
        right_eye = kps[1]
        nose = kps[2]
        
        # 计算两眼之间的向量
        eye_vector = np.array(right_eye) - np.array(left_eye)
        # 计算鼻子到两眼中点的向量
        eye_midpoint = (np.array(left_eye) + np.array(right_eye)) / 2
        nose_vector = np.array(nose) - eye_midpoint
        
        # 计算面部宽度和高度
        face_width = np.linalg.norm(eye_vector)
        face_height = np.linalg.norm(nose_vector)
        
        # 分析面部关键点，确定可见面部面积
        # 关键点索引：0=左眼，1=右眼，2=鼻子，3=左嘴角，4=右嘴角
        left_eye = np.array(kps[0])
        right_eye = np.array(kps[1])
        nose = np.array(kps[2])
        left_mouth = np.array(kps[3])
        right_mouth = np.array(kps[4])
        
        # 计算面部中心点
        face_center = (left_eye + right_eye + nose + left_mouth + right_mouth) / 5
        
        # 计算面部边界框
        face_points = np.array([left_eye, right_eye, nose, left_mouth, right_mouth])
        min_x = np.min(face_points[:, 0])
        max_x = np.max(face_points[:, 0])
        face_width_actual = max_x - min_x
        
        # 计算左右面部的可见程度
        # 基于面部关键点相对于面部中心的分布
        left_face_points = [left_eye, left_mouth]
        right_face_points = [right_eye, right_mouth]
        
        # 计算左侧面部点到中心的平均距离
        left_distances = [np.linalg.norm(p - face_center) for p in left_face_points]
        avg_left_distance = np.mean(left_distances)
        
        # 计算右侧面部点到中心的平均距离
        right_distances = [np.linalg.norm(p - face_center) for p in right_face_points]
        avg_right_distance = np.mean(right_distances)
        
        # 计算面部方向：基于左右面部可见程度
        # 左脸可见程度 > 右脸可见程度 → 负值（左侧脸）
        # 右脸可见程度 > 左脸可见程度 → 正值（右侧脸）
        visibility_ratio = (avg_right_distance - avg_left_distance) / (max(avg_left_distance, avg_right_distance) + 1e-6)
        
        # 计算面部的宽高比，用于判断正面还是侧面
        width_height_ratio = face_width / (face_height + 1e-6)
        
        # 计算侧面程度：采用更直接的方法确保侧脸角度更大
        # 基于宽高比和可见度比例的综合判断
        
        # 方法1：基于宽高比的角度计算（主要因素）
        # 宽高比越小，角度越大
        if width_height_ratio > 1.5:
            # 正面
            angle_from_ratio = 0.0
        elif width_height_ratio < 0.9:
            # 侧脸（大幅降低阈值）
            angle_from_ratio = 85.0
        else:
            # 中间状态
            # 从1.5到0.9，角度从0到85快速增长
            angle_from_ratio = (1.5 - width_height_ratio) / (1.5 - 0.9) * 85.0
        
        # 方法2：基于可见度比例的角度增强
        # 可见度差异越大，角度越大
        visibility_strength = min(abs(visibility_ratio) * 3.0, 1.0)
        angle_from_visibility = 85.0 * visibility_strength
        
        # 综合两种方法，偏向于较大的角度
        # 使用最大值
        base_angle = max(angle_from_ratio, angle_from_visibility)
        
        # 强制增强：对于明显的侧脸，确保角度足够大
        if width_height_ratio < 1.1 or abs(visibility_ratio) > 0.3:
            base_angle = max(base_angle, 75.0)
        
        # 计算最终角度
        if base_angle < 5.0 and abs(visibility_ratio) < 0.1:
            # 接近正面
            direction_angle = 0.0
        else:
            # 使用计算得到的基础角度
            
            # 根据可见度比例确定方向和角度大小
            # 左脸更多 → 负值
            # 右脸更多 → 正值
            if visibility_ratio > 0:
                # 右脸更多
                direction_angle = base_angle
            elif visibility_ratio < 0:
                # 左脸更多
                direction_angle = -base_angle
            else:
                # 左右脸相当
                direction_angle = 0.0
        
        # 确定方向描述
        if abs(direction_angle) < 10:
            direction = "Front"
        elif direction_angle > 45:
            direction = "Right Side"
        elif direction_angle < -45:
            direction = "Left Side"
        elif direction_angle > 0:
            direction = "Right Quarter"
        else:
            direction = "Left Quarter"
        
        return (direction_angle, direction)
