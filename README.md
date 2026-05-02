# ydem_nodes

ComfyUI 自定义节点集合

## 节点列表

### 骨骼处理节点
- **YDem Pose Filter**: POSE_KEYPOINT人物过滤节点，从多人物骨骼数据中筛选出指定人物

## 安装

1. 将 `ydem_nodes` 文件夹复制到 ComfyUI 的 `custom_nodes` 目录
2. 确保已安装 `comfyui_controlnet_aux` 节点包（提供POSE_KEYPOINT数据）
3. 重启 ComfyUI

## 使用方法

1. 在 ComfyUI 中，节点会出现在对应的分类下：
   - 骨骼处理节点：`ydem_nodes/pose` 分类
2. 拖放节点到工作区并连接输入输出

## 示例

### YDem Pose Filter
- 输入：
  - `pose_keypoint`: 来自 DWPose 等节点的骨骼关键点数据（POSE_KEYPOINT类型）
  - `input_pose_order`: 人物排序策略（left-right, right-left, top-bottom, bottom-top, small-large, large-small）
  - `input_pose_index`: 要保留的人物索引（基于第1帧排序后的索引）
- 输出：
  - `pose_keypoint`: 仅保留指定人物的骨骼数据

## 版本

v2.0.0 - 重构为骨骼处理节点，添加POSE_KEYPOINT人物过滤功能
