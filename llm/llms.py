from langchain_openai import ChatOpenAI
from utils.constants import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL

# DeepSeek 云端模型
# 新增:
#   - request_timeout=30 : 避免请求无限卡住，30 秒超时给出明确报错
#   - max_retries=3      : 遇网络抖动（如 RemoteProtocolError）自动重试
#     （langchain_openai 底层默认就会对 5xx/网络错误做若干次指数退避重试，
#      显式声明更可控）
deepseek_llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL,
    temperature=0.2,
    timeout=30,           # 单请求 30 秒超时（防止 RemoteProtocolError 卡死对话）
    max_retries=3,       # 自动重试 3 次（指数退避）
)

# 本地部署的 Qwen3.6 模型（vLLM 服务）
local_llm = ChatOpenAI(
    model="/root/autodl-tmp/models/cyankiwi/Qwen3___6-27B-AWQ-INT4",
    api_key="EMPTY",
    base_url="http://127.0.0.1:8000/v1",
    temperature=0.1,
    timeout=60,          # 本地模型首请求可能较长，给更多时间
    max_retries=2,
)