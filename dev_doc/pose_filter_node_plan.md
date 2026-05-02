# POSE_KEYPOINT 人物过滤节点 - 开发计划

## 1. 需求分析

### 1.1 业务背景
本节点用于处理多人物骨骼动画数据，从多帧POSE_KEYPOINT数据中筛选出用户指定的单人骨骼数据，便于后续单人动画处理流程。

### 1.2 功能需求
| 需求编号 | 需求描述 | 来源参考 |
| :--- | :--- | :--- |
| REQ-001 | 输入POSE_KEYPOINT格式数据（JSON格式多人物骨骼） | comfyui_controlnet_aux/node_wrappers/dwpose.py |
| REQ-002 | 输出POSE_KEYPOINT格式数据（仅保留指定人物） | - |
| REQ-003 | 支持按多种顺序策略排序人物 | ComfyUI-ReActor/nodes.py#L1677-1678 |
| REQ-004 | 支持用户指定保留的人物索引 | ComfyUI-ReActor/nodes.py#L1680 |

### 1.3 输入输出规格
- **输入**: 
  - `pose_keypoint`: POSE_KEYPOINT类型，包含多帧、多人的骨骼关键点数据
- **输出**: 
  - `pose_keypoint`: POSE_KEYPOINT类型，仅保留指定人物的骨骼数据

### 1.4 参数设计（参考ReActor）
| 参数名 | 类型 | 默认值 | 可选值 | 说明 |
| :--- | :--- | :--- | :--- | :--- |
| input_pose_order | COMBO | "large-small" | ["left-right", "right-left", "top-bottom", "bottom-top", "small-large", "large-small"] | 人物排序策略 |
| input_pose_index | STRING | "0" | 单个数字，如 "0" | 要保留的人物索引（基于第1帧，仅支持单人） |

## 2. 技术方案

### 2.1 节点命名
- **类名**: `YDemPoseFilter`
- **显示名称**: `YDem Pose Filter`
- **分类**: `ydem_nodes/pose`

### 2.2 数据结构分析
POSE_KEYPOINT格式说明（待调研确认）：
```
{
  "version": "1.0",
  "frames": [
    {
      "frame_index": 0,
      "persons": [
        {
          "person_id": 0,
          "keypoints": [...],
          "bbox": [x1, y1, x2, y2],
          "score": 0.95
        },
        ...
      ]
    },
    ...
  ]
}
```

### 2.3 核心算法设计

#### 2.3.1 排序策略实现
| 排序类型 | 排序依据 | 实现逻辑 |
| :--- | :--- | :--- |
| left-right | 人物bounding box中心点x坐标 | 从小到大排序 |
| right-left | 人物bounding box中心点x坐标 | 从大到小排序 |
| top-bottom | 人物bounding box中心点y坐标 | 从小到大排序 |
| bottom-top | 人物bounding box中心点y坐标 | 从大到小排序 |
| small-large | 人物bounding box面积 | 从小到大排序 |
| large-small | 人物bounding box面积 | 从大到小排序 |

#### 2.3.2 人物追踪逻辑
1. 在第1帧中，按指定排序策略对人物进行排序
2. 根据input_pose_index确定要保留的人物索引（仅支持单人）
3. 记录该人物的特征（bbox、关键点位置等）用于后续帧匹配
4. 在后续帧中，通过简单的IoU匹配算法找到对应人物并保留
5. **设计说明**：采用简单的本地追踪算法，不保证100%正确；复杂场景下可能出现追踪错误，后续将添加用户介入修正手段（如指定第N帧选择其他序号人物）

#### 2.3.3 过滤流程
```
输入POSE_KEYPOINT → 解析第1帧人物 → 按策略排序 → 确定保留索引 → 
遍历所有帧 → 特征匹配追踪 → 仅保留目标人物 → 输出过滤后的POSE_KEYPOINT
```

### 2.4 错误处理
| 异常场景 | 处理策略 |
| :--- | :--- |
| 输入为空 | 返回空的POSE_KEYPOINT |
| 索引超出范围 | 使用最大有效索引 |
| 某帧无匹配人物 | 保留上一帧数据或置空 |

## 3. 开发计划

### 3.1 任务分解
| 任务编号 | 任务描述 | 预估工时 | 依赖 |
| :--- | :--- | :--- | :--- |
| TASK-001 | 调研POSE_KEYPOINT具体数据格式 | 2h | - |
| TASK-002 | 创建节点类框架 | 1h | TASK-001 |
| TASK-003 | 实现排序策略算法 | 3h | TASK-002 |
| TASK-004 | 实现人物追踪匹配算法 | 4h | TASK-003 |
| TASK-005 | 编写单元测试 | 2h | TASK-004 |
| TASK-006 | 集成到ydem_nodes.py | 1h | TASK-005 |

### 3.2 里程碑
| 里程碑 | 目标 | 预计完成时间 |
| :--- | :--- | :--- |
| M1 | 完成需求分析和技术方案设计 | 2026-05-02 |
| M2 | 完成核心算法实现 | 2026-05-05 |
| M3 | 完成测试和集成 | 2026-05-08 |

## 4. 代码结构

### 4.1 新增文件
- `ydem_nodes/pose_filter_core.py` - 核心算法实现（排序、追踪、过滤）

### 4.2 修改文件
- `ydem_nodes/ydem_nodes.py` - 添加YDemPoseFilter节点类

### 4.3 类结构设计
```python
class YDemPoseFilter:
    """POSE_KEYPOINT人物过滤节点"""
    
    @classmethod
    def INPUT_TYPES(cls):
        # 返回input_faces_order和input_faces_index参数定义
    
    RETURN_TYPES = ("POSE_KEYPOINT",)
    RETURN_NAMES = ("pose_keypoint",)
    FUNCTION = "process"
    CATEGORY = "ydem_nodes/pose"
    
    def process(self, pose_keypoint, input_pose_order, input_pose_index):
        # 实现过滤逻辑
```

## 5. 参考资源

### 5.1 参考节点
| 节点路径 | 用途 |
| :--- | :--- |
| `comfyui_controlnet_aux/node_wrappers/dwpose.py` | POSE_KEYPOINT数据格式参考 |
| `ComfyUI-ReActor/nodes.py#L1672-1687` | input_faces_order和input_faces_index参数设计参考 |

### 5.2 相关文档
- ComfyUI节点开发规范
- POSE_KEYPOINT数据格式规范（待确认）

## 6. 待确认事项

| 序号 | 待确认内容 | 确认状态 | 确认结果 |
| :--- | :--- | :--- | :--- |
| 1 | POSE_KEYPOINT的具体JSON结构 | 待调研 | - |
| 2 | 是否需要支持多人物保留（如input_pose_index="0,2"） | **已确认** | 不支持多人，仅保留单人 |
| 3 | 人物追踪的特征匹配算法选择（IoU/关键点距离） | **已确认** | 采用简单的IoU本地追踪算法，不保证100%正确；复杂场景会有追踪错误，后续添加用户介入修正手段 |
| 4 | 是否需要处理人物ID变化的情况 | 待确认 | - |
