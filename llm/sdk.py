# 模型下载
from modelscope import snapshot_download
model_dir = snapshot_download('cyankiwi/Qwen3.6-27B-AWQ-INT4', cache_dir='/root/autodl-tmp/models', revision='master')

