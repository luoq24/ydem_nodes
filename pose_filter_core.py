"""POSE_KEYPOINT人物过滤核心逻辑"""

import json

class PoseFilterCore:
    """骨骼数据过滤核心类"""
    
    def __init__(self):
        self.last_target_bbox = None
    
    def get_bbox_center(self, bbox):
        """计算bounding box中心点"""
        if len(bbox) >= 4:
            x1, y1, x2, y2 = bbox[:4]
            return ((x1 + x2) / 2, (y1 + y2) / 2)
        return (0, 0)
    
    def get_bbox_area(self, bbox):
        """计算bounding box面积"""
        if len(bbox) >= 4:
            x1, y1, x2, y2 = bbox[:4]
            return (x2 - x1) * (y2 - y1)
        return 0
    
    def calculate_iou(self, bbox1, bbox2):
        """计算两个bounding box的IoU"""
        if len(bbox1) < 4 or len(bbox2) < 4:
            return 0.0
        
        x1, y1, x2, y2 = bbox1[:4]
        x1_, y1_, x2_, y2_ = bbox2[:4]
        
        # 计算交集
        inter_x1 = max(x1, x1_)
        inter_y1 = max(y1, y1_)
        inter_x2 = min(x2, x2_)
        inter_y2 = min(y2, y2_)
        
        inter_width = max(0, inter_x2 - inter_x1)
        inter_height = max(0, inter_y2 - inter_y1)
        inter_area = inter_width * inter_height
        
        # 计算并集
        area1 = (x2 - x1) * (y2 - y1)
        area2 = (x2_ - x1_) * (y2_ - y1_)
        union_area = area1 + area2 - inter_area
        
        if union_area == 0:
            return 0.0
        
        return inter_area / union_area
    
    def sort_persons(self, persons, order):
        """按指定策略排序人物"""
        if not persons:
            return persons
        
        def get_sort_key(person):
            bbox = person.get('bbox', [0, 0, 0, 0])
            center_x, center_y = self.get_bbox_center(bbox)
            area = self.get_bbox_area(bbox)
            
            if order == 'left-right':
                return center_x
            elif order == 'right-left':
                return -center_x
            elif order == 'top-bottom':
                return center_y
            elif order == 'bottom-top':
                return -center_y
            elif order == 'small-large':
                return area
            elif order == 'large-small':
                return -area
            else:
                return -area  # 默认按面积降序
        
        return sorted(persons, key=get_sort_key)
    
    def filter_pose(self, pose_keypoint, input_pose_order, input_pose_index):
        """过滤POSE_KEYPOINT数据，仅保留指定人物"""
        # 解析输入参数
        try:
            target_index = int(input_pose_index.strip())
        except ValueError:
            target_index = 0
        
        # 解析POSE_KEYPOINT数据
        pose_data = json.loads(pose_keypoint)
        frames = pose_data.get('frames', [])
        
        if not frames:
            return json.dumps(pose_data)
        
        # 处理第1帧：确定目标人物
        first_frame = frames[0]
        persons = first_frame.get('persons', [])
        
        if not persons:
            return json.dumps(pose_data)
        
        # 按指定策略排序
        sorted_persons = self.sort_persons(persons, input_pose_order)
        
        # 确定目标人物索引
        if target_index >= len(sorted_persons):
            target_index = 0
        
        target_person = sorted_persons[target_index]
        self.last_target_bbox = target_person.get('bbox', [0, 0, 0, 0])
        
        print(f"=== PoseFilterCore 初始化 ===")
        print(f"排序策略: {input_pose_order}")
        print(f"目标索引: {target_index}")
        print(f"第1帧人数: {len(persons)}")
        print(f"目标人物bbox: {self.last_target_bbox}")
        
        # 遍历所有帧进行过滤
        for frame_idx, frame in enumerate(frames):
            frame_persons = frame.get('persons', [])
            
            if not frame_persons:
                frame['persons'] = []
                continue
            
            # 第1帧直接保留目标人物
            if frame_idx == 0:
                frame['persons'] = [target_person]
                continue
            
            # 后续帧：通过IoU匹配找到最相似的人物
            best_match = None
            best_iou = 0.3  # 阈值
            
            for person in frame_persons:
                person_bbox = person.get('bbox', [0, 0, 0, 0])
                iou = self.calculate_iou(self.last_target_bbox, person_bbox)
                
                if iou > best_iou:
                    best_iou = iou
                    best_match = person
            
            if best_match:
                frame['persons'] = [best_match]
                self.last_target_bbox = best_match.get('bbox', [0, 0, 0, 0])
                print(f"帧 {frame_idx}: 找到匹配人物, IoU={best_iou:.4f}")
            else:
                frame['persons'] = []
                print(f"帧 {frame_idx}: 未找到匹配人物")
        
        print("===========================")
        
        return json.dumps(pose_data)
