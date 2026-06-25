import os
from dotenv import load_dotenv

load_dotenv()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL")


# Text-to-SQL Agent Prompt 模板（{chat_history} 是对话历史占位符，在 query() 中展开）
DEFAULT_AGENT_PROMPT = "你是一名数据库助手，可以通过调用工具查询数据库。"
AGENT_PROMPT = os.getenv("AGENT_PROMPT", DEFAULT_AGENT_PROMPT)

# Text-to-SQL Agent 对话记忆配置
# AGENT_MEMORY_ENABLED   : 是否启用多轮对话记忆（True/False）
# AGENT_MAX_MEMORY_TURNS : 最多保留的历史对话轮数（1 轮 = 1 次用户提问 + 1 次助手回答）
def _parse_bool(val, default=False):
    if val is None:
        return default
    return str(val).strip().lower() in ("1", "true", "yes", "y", "on")

AGENT_MEMORY_ENABLED=_parse_bool(os.getenv("AGENT_MEMORY_ENABLED"), default=True)
AGENT_MAX_MEMORY_TURNS=int(os.getenv("AGENT_MAX_MEMORY_TURNS", "5"))
USE_LOCAL_LLM=_parse_bool(os.getenv("USE_LOCAL_LLM"), default=False)
