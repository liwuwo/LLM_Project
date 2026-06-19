from langchain_openai import ChatOpenAI
from utils.constants import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL

# DeepSeek 云端模型
deepseek_llm = ChatOpenAI(
    model="deepseek-v4-flash",
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL,
    temperature=0.9
)

# 本地部署的 Qwen3.6 模型（vLLM 服务）
local_llm = ChatOpenAI(
    model="/root/autodl-tmp/models/cyankiwi/Qwen3___6-27B-AWQ-INT4",
    api_key="EMPTY",
    base_url="http://127.0.0.1:8000/v1",
    temperature=0.9
)