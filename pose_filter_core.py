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
    
    def get_bbox_from_keypoints(self, keypoints):
        """从关键点数据中计算bounding box"""
        if not keypoints or len(keypoints) < 3:
            return [0, 0, 0, 0]
        
        xs = []
        ys = []
        
        for i in range(0, len(keypoints), 3):
            x = keypoints[i]
            y = keypoints[i + 1]
            score = keypoints[i + 2] if i + 2 < len(keypoints) else 0
            
            if score > 0 and x > 0 and y > 0:
                xs.append(x)
                ys.append(y)
        
        if not xs or not ys:
            return [0, 0, 0, 0]
        
        x1 = min(xs)
        y1 = min(ys)
        x2 = max(xs)
        y2 = max(ys)
        
        return [x1, y1, x2, y2]
    
    def calculate_iou(self, bbox1, bbox2):
        """计算两个bounding box的IoU"""
        if len(bbox1) < 4 or len(bbox2) < 4:
            return 0.0
        
        x1, y1, x2, y2 = bbox1[:4]
        x1_, y1_, x2_, y2_ = bbox2[:4]
        
        inter_x1 = max(x1, x1_)
        inter_y1 = max(y1, y1_)
        inter_x2 = min(x2, x2_)
        inter_y2 = min(y2, y2_)
        
        inter_width = max(0, inter_x2 - inter_x1)
        inter_height = max(0, inter_y2 - inter_y1)
        inter_area = inter_width * inter_height
        
        area1 = (x2 - x1) * (y2 - y1)
        area2 = (x2_ - x1_) * (y2_ - y1_)
        union_area = area1 + area2 - inter_area
        
        if union_area == 0:
            return 0.0
        
        return inter_area / union_area
    
    def get_person_bbox(self, person):
        """获取人物的bounding box，支持从bbox字段或关键点计算"""
        bbox = person.get('bbox', [])
        if len(bbox) >= 4:
            return bbox
        
        bbox = person.get('bbox_xyxy', [])
        if len(bbox) >= 4:
            return bbox
        
        keypoints = person.get('pose_keypoints_2d', [])
        return self.get_bbox_from_keypoints(keypoints)
    
    def sort_persons(self, persons, order):
        """按指定策略排序人物（参考ReActor实现）"""
        if not persons:
            return persons
        
        def get_sort_key(person):
            bbox = self.get_person_bbox(person)
            x1, y1 = bbox[0], bbox[1]
            area = self.get_bbox_area(bbox)
            
            if order == 'left-right':
                return x1
            elif order == 'right-left':
                return -x1
            elif order == 'top-bottom':
                return y1
            elif order == 'bottom-top':
                return -y1
            elif order == 'small-large':
                return area
            elif order == 'large-small':
                return -area
            else:
                return -area
        
        return sorted(persons, key=get_sort_key)
    
    def filter_pose(self, pose_keypoint, input_pose_order, input_pose_index):
        """过滤POSE_KEYPOINT数据，仅保留指定人物"""
        is_list_input = isinstance(pose_keypoint, list)
        
        try:
            target_index = int(input_pose_index.strip())
        except ValueError:
            target_index = 0
        
        # 解析POSE_KEYPOINT数据
        if isinstance(pose_keypoint, str):
            pose_data = json.loads(pose_keypoint)
        elif isinstance(pose_keypoint, dict):
            pose_data = pose_keypoint.copy()
            if 'frames' in pose_data:
                pose_data['frames'] = [frame.copy() for frame in pose_data['frames']]
        elif isinstance(pose_keypoint, list):
            if pose_keypoint and isinstance(pose_keypoint[0], dict):
                first_item = pose_keypoint[0]
                canvas_height = first_item.get('canvas_height', 512)
                canvas_width = first_item.get('canvas_width', 512)
                pose_data = {
                    "version": "1.0",
                    "canvas_height": canvas_height,
                    "canvas_width": canvas_width,
                    "frames": [frame.copy() for frame in pose_keypoint]
                }
            else:
                pose_data = {"version": "1.0", "canvas_height": 512, "canvas_width": 512, "frames": []}
        else:
            pose_data = {"version": "1.0", "canvas_height": 512, "canvas_width": 512, "frames": []}
        
        frames = pose_data.get('frames', [])
        
        if not frames:
            return pose_keypoint if is_list_input else pose_data
        
        first_frame = frames[0]
        persons = first_frame.get('persons', [])
        if not persons:
            persons = first_frame.get('people', [])
        
        if not persons:
            return pose_keypoint if is_list_input else pose_data
        
        sorted_persons = self.sort_persons(persons, input_pose_order)
        
        if target_index >= len(sorted_persons):
            target_index = 0
        
        target_person = sorted_persons[target_index]
        self.last_target_bbox = self.get_person_bbox(target_person)
        
        # 遍历所有帧进行过滤
        for frame_idx, frame in enumerate(frames):
            frame_persons = frame.get('persons', [])
            if not frame_persons:
                frame_persons = frame.get('people', [])
            
            if not frame_persons:
                frame['persons'] = []
                frame['people'] = []
                continue
            
            if frame_idx == 0:
                frame['persons'] = [target_person]
                frame['people'] = [target_person]
                continue
            
            best_match = None
            best_iou = 0.3
            
            for person in frame_persons:
                person_bbox = self.get_person_bbox(person)
                iou = self.calculate_iou(self.last_target_bbox, person_bbox)
                
                if iou > best_iou:
                    best_iou = iou
                    best_match = person
            
            if best_match:
                frame['persons'] = [best_match]
                frame['people'] = [best_match]
                self.last_target_bbox = self.get_person_bbox(best_match)
            else:
                frame['persons'] = []
                frame['people'] = []
        
        # 返回与输入相同的格式
        if is_list_input:
            return frames
        return pose_data
