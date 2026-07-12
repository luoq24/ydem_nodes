"""ydem_nodes - ComfyUI自定义节点"""

from .ydem_nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']
__version__ = "1.0.0"

# 让 ComfyUI 前端加载 ./js 下的 web 扩展（prompt-manager 桥接）
WEB_DIRECTORY = "./js"
