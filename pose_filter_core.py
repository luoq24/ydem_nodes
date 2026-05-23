"""POSE_KEYPOINT人物过滤核心逻辑"""

import copy
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
    
    def parse_force_reset_frames(self, force_reset_frames):
        """解析强制重设帧配置，返回帧号到(排序策略, 选人序号)的映射"""
        reset_config = {}
        
        if not force_reset_frames or not force_reset_frames.strip():
            return reset_config
        
        # 支持的排序策略简写映射
        valid_strategies = {
            'left': 'left-right',
            'right': 'right-left', 
            'top': 'top-bottom',
            'bottom': 'bottom-top',
            'small': 'small-large',
            'large': 'large-small',
            'empty': 'empty'  # 不输出pose数据
        }
        
        # 分割多组配置
        groups = force_reset_frames.split(';')
        
        for group in groups:
            group = group.strip()
            if not group:
                continue
            
            parts = group.split('_')
            if len(parts) != 3:
                raise ValueError(f"无效的强制重设配置格式: {group}，正确格式应为'帧序号_策略简写_选人序号'")
            
            try:
                frame_num = int(parts[0])
            except ValueError:
                raise ValueError(f"帧序号必须是整数: {parts[0]}")
            
            if frame_num <= 0:
                raise ValueError(f"帧序号必须大于0: {frame_num}")
            
            strategy_short = parts[1]
            if strategy_short not in valid_strategies:
                raise ValueError(f"无效的选人顺序策略简写: {strategy_short}，可选值: {list(valid_strategies.keys())}")
            
            try:
                pose_index = int(parts[2])
            except ValueError:
                raise ValueError(f"选人序号必须是整数: {parts[2]}")
            
            if pose_index < 0:
                raise ValueError(f"选人序号不能为负数: {pose_index}")
            
            reset_config[frame_num] = (valid_strategies[strategy_short], pose_index)
        
        return reset_config
    
    def filter_pose(self, pose_keypoint, input_pose_order, input_pose_index, force_reset_frames=""):
        """过滤POSE_KEYPOINT数据，仅保留指定人物"""
        is_list_input = isinstance(pose_keypoint, list)
        
        try:
            target_index = int(input_pose_index)
        except (ValueError, TypeError):
            target_index = 0
        
        # 解析强制重设配置
        reset_config = self.parse_force_reset_frames(force_reset_frames)
        
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
                
                # 检查输入列表是否是帧列表还是包含frames字段的对象列表
                if 'frames' in first_item:
                    # 输入是包含frames字段的对象列表，需要合并所有帧
                    all_frames = []
                    for item in pose_keypoint:
                        if isinstance(item, dict) and 'frames' in item:
                            all_frames.extend([f.copy() for f in item['frames']])
                    pose_data = {
                        "version": "1.0",
                        "canvas_height": canvas_height,
                        "canvas_width": canvas_width,
                        "frames": all_frames
                    }
                else:
                    # 输入本身就是帧列表
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
        
        # 获取第一帧的目标人物
        sorted_persons = self.sort_persons(persons, input_pose_order)
        if target_index >= len(sorted_persons):
            target_index = 0
        target_person = sorted_persons[target_index]
        self.last_target_bbox = self.get_person_bbox(target_person)
        
        # 当前使用的排序策略和选人序号
        current_order = input_pose_order
        current_index = target_index
        # 是否清空pose数据的标志
        empty_mode = False
        
        # 遍历所有帧进行过滤
        for frame_idx, frame in enumerate(frames):
            frame_persons = frame.get('persons', [])
            if not frame_persons:
                frame_persons = frame.get('people', [])
            
            if not frame_persons:
                frame['persons'] = []
                frame['people'] = []
                continue
            
            # 检查是否需要强制重设
            if frame_idx in reset_config:
                current_order, current_index = reset_config[frame_idx]
                
                # 检查是否是empty策略
                if current_order == 'empty':
                    empty_mode = True
                    frame['persons'] = []
                    frame['people'] = []
                    continue
                else:
                    empty_mode = False
                    sorted_frame_persons = self.sort_persons(frame_persons, current_order)
                    if current_index >= len(sorted_frame_persons):
                        current_index = 0
                    target_person = sorted_frame_persons[current_index]
                    frame['persons'] = [target_person]
                    frame['people'] = [target_person]
                    self.last_target_bbox = self.get_person_bbox(target_person)
                    continue
            
            # 如果处于empty模式，清空pose数据
            if empty_mode:
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


class PoseHandRemoveCore:
    """手部关键点移除核心类"""

    # COCO-WholeBody 133点格式中手部的索引范围 (每个关键点3个值: x, y, score)
    LEFT_HAND_START = 91 * 3   # 273
    LEFT_HAND_END = 112 * 3    # 336 (exclusive)
    RIGHT_HAND_START = 112 * 3 # 336
    RIGHT_HAND_END = 133 * 3   # 399 (exclusive)

    def parse_frames(self, frames_str):
        """解析帧范围字符串，如 '1_5;13_20'，返回0-based的帧索引集合"""
        frame_set = set()
        if not frames_str or not frames_str.strip():
            return frame_set

        segments = frames_str.split(';')
        for segment in segments:
            segment = segment.strip()
            if not segment:
                continue

            parts = segment.split('_')
            if len(parts) != 2:
                raise ValueError(f"无效的帧范围格式: {segment}，正确格式应为'起始帧_结束帧'")

            try:
                start = int(parts[0]) - 1  # 转为0-based索引
                end = int(parts[1]) - 1    # 转为0-based索引
            except ValueError:
                raise ValueError(f"帧序号必须是整数: {segment}")

            if start < 0 or end < 0:
                raise ValueError(f"帧序号必须大于0: {segment}")

            if start > end:
                start, end = end, start

            for i in range(start, end + 1):
                frame_set.add(i)

        return frame_set

    def remove_hands(self, pose_keypoint, left_hand, right_hand, frames_str):
        """移除指定帧的手部关键点"""
        if not left_hand and not right_hand:
            return pose_keypoint

        is_list_input = isinstance(pose_keypoint, list)
        target_frames = self.parse_frames(frames_str)

        # 如果没有指定帧范围，不处理
        if not target_frames:
            return pose_keypoint

        # 解析POSE_KEYPOINT数据
        if isinstance(pose_keypoint, str):
            pose_data = json.loads(pose_keypoint)
            if 'frames' in pose_data:
                pose_data['frames'] = copy.deepcopy(pose_data['frames'])
        elif isinstance(pose_keypoint, dict):
            pose_data = pose_keypoint.copy()
            if 'frames' in pose_data:
                pose_data['frames'] = copy.deepcopy(pose_data['frames'])
        elif isinstance(pose_keypoint, list):
            if pose_keypoint and isinstance(pose_keypoint[0], dict):
                first_item = pose_keypoint[0]
                canvas_height = first_item.get('canvas_height', 512)
                canvas_width = first_item.get('canvas_width', 512)

                if 'frames' in first_item:
                    all_frames = []
                    for item in pose_keypoint:
                        if isinstance(item, dict) and 'frames' in item:
                            all_frames.extend(copy.deepcopy(item['frames']))
                    pose_data = {
                        "version": "1.0",
                        "canvas_height": canvas_height,
                        "canvas_width": canvas_width,
                        "frames": all_frames
                    }
                else:
                    pose_data = {
                        "version": "1.0",
                        "canvas_height": canvas_height,
                        "canvas_width": canvas_width,
                        "frames": copy.deepcopy(pose_keypoint)
                    }
            else:
                return pose_keypoint
        else:
            return pose_keypoint

        frames = pose_data.get('frames', [])

        # 处理目标帧
        for frame_idx, frame in enumerate(frames):
            if frame_idx not in target_frames:
                continue

            persons = frame.get('persons', [])
            if not persons:
                persons = frame.get('people', [])

            for person in persons:
                # 处理分开存储的手部关键点（OpenPose格式）
                if left_hand:
                    if 'hand_left_keypoints_2d' in person:
                        person['hand_left_keypoints_2d'] = [0.0] * len(person['hand_left_keypoints_2d'])

                if right_hand:
                    if 'hand_right_keypoints_2d' in person:
                        person['hand_right_keypoints_2d'] = [0.0] * len(person['hand_right_keypoints_2d'])

                # 处理统一存储的关键点（COCO-WholeBody 133点格式）
                keypoints = person.get('pose_keypoints_2d', [])
                if len(keypoints) >= self.RIGHT_HAND_END:  # 133 * 3 = 399
                    if left_hand:
                        for i in range(self.LEFT_HAND_START, self.LEFT_HAND_END):
                            keypoints[i] = 0.0
                    if right_hand:
                        for i in range(self.RIGHT_HAND_START, self.RIGHT_HAND_END):
                            keypoints[i] = 0.0

        # 返回与输入相同的格式
        if is_list_input:
            return frames
        return pose_data
