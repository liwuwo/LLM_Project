"""
对话记忆组件（零依赖的轻量实现）。

设计背景：
    项目原计划用 LangChain 的 ConversationBufferMemory（from langchain.memory），
    但线上运行环境的包结构不统一（langchain_classic / langchain_core / langchain_community
    版本差异导致无法用固定路径导入）。而 ConversationBufferMemory 的核心功能非常简单：
    维护一个消息列表 + 提供 "用户: xxx\n助手: xxx" 格式的字符串化方法。因此选择
    零依赖自实现，接口与 LangChain 标准保持一致，后续环境完备时可直接替换。

对外接口（与 LangChain ConversationBufferMemory 兼容）：
    - save_context(inputs, outputs)  : 写入一条"用户-助手"对话
    - load_memory_variables({})      : 返回 {"chat_history": "格式化字符串"}
    - clear()                         : 清空记忆
    - chat_memory.messages            : 原始消息列表（便于手动做轮数裁剪）

典型用法：
    from agents.conversation_memory import ConversationHistory
    mem = ConversationHistory(
        memory_key="chat_history",
        return_messages=False,   # 返回字符串而非消息对象
        human_prefix="用户",
        ai_prefix="助手",
    )
    mem.save_context({"input": "有哪几张表？"}, {"output": "users, orders, products"})
    history = mem.load_memory_variables({})["chat_history"]
    # history == "用户: 有哪几张表？\n助手: users, orders, products"
"""

from __future__ import annotations

__all__ = ["ConversationHistory"]


class _ChatMessage:
    """一条内部消息对象。不对外暴露。

    Args:
        message_type: 'human'（用户）或 'ai'（助手）
        content: 消息文本内容
    """
    __slots__ = ("type", "content")

    def __init__(self, message_type: str, content: str):
        self.type = message_type
        self.content = str(content)


class _ChatMemoryStore:
    """轻量消息列表容器。保留 `messages` 属性名以与 LangChain 调用习惯兼容。

    之所以单独抽一个类，是为了保持 `chat_memory.messages` 的访问形式与
    LangChain 的 ConversationBufferMemory 一致：下游裁剪逻辑可以直接用
    `mem.chat_memory.messages.pop(0)`，无需关心底层数据结构变化。
    """
    __slots__ = ("messages",)

    def __init__(self):
        self.messages: list = []

    def append(self, message: _ChatMessage):
        self.messages.append(message)

    def clear(self):
        self.messages.clear()


class ConversationHistory:
    """
    零依赖的对话历史记录器。接口与 LangChain ConversationBufferMemory 保持一致。

    Args:
        memory_key:       load_memory_variables() 返回字典的 key，默认 "chat_history"。
                          与 .env / AGENT_PROMPT 中的 {chat_history} 占位符保持一致。
        return_messages:  False 时返回格式化字符串（默认）；
                          True  时返回原始 _ChatMessage 列表（罕用，但保留便于调试）。
        human_prefix:     用户发言的前缀，默认 "用户"。用于格式化输出。
        ai_prefix:        助手发言的前缀，默认 "助手"。
    """

    def __init__(
        self,
        memory_key: str = "chat_history",
        return_messages: bool = False,
        human_prefix: str = "用户",
        ai_prefix: str = "助手",
    ):
        self.memory_key = memory_key
        self.return_messages = return_messages
        self.human_prefix = human_prefix
        self.ai_prefix = ai_prefix
        self.chat_memory = _ChatMemoryStore()

    # ──────────────── 写入 ────────────────
    def save_context(self, inputs: dict, outputs: dict) -> None:
        """把一条"用户提问-助手回答"写入记忆。

        故意用 "取第一个字符串值" 的策略，以兼容多种调用方式：
            mem.save_context({"input": "问题"}, {"output": "回答"})
            mem.save_context({"question": "问题"}, {"answer": "回答"})
            mem.save_context({"input": "问题", "extra": "...", }, {"output": "回答"})
        这些都能正确工作。
        """
        human_content = self._first_string_value(inputs)
        ai_content = self._first_string_value(outputs)
        if human_content is not None:
            self.chat_memory.append(_ChatMessage("human", human_content))
        if ai_content is not None:
            self.chat_memory.append(_ChatMessage("ai", ai_content))

    @staticmethod
    def _first_string_value(d: dict) -> str | None:
        """从字典中取第一个字符串值。
        找不到字符串就取第一个非 None 值转字符串，否则返回 None。
        """
        if not d:
            return None
        for v in d.values():
            if isinstance(v, str):
                return v
            if v is not None:
                return str(v)
        return None

    # ──────────────── 读取 ────────────────
    def load_memory_variables(self, inputs: dict | None = None) -> dict:
        """按 memory_key 返回格式化后的记忆内容。

        与 LangChain ConversationBufferMemory 的返回格式保持完全一致：
            {"chat_history": "用户: xxx\n助手: xxx\n用户: yyy\n助手: yyy"}
        """
        if self.return_messages:
            # 返回消息对象列表（罕用，但保留接口便于调试/未来扩展）
            return {self.memory_key: list(self.chat_memory.messages)}

        # 默认：返回 "用户: xxx\n助手: xxx" 格式的字符串
        lines = []
        for msg in self.chat_memory.messages:
            prefix = self.human_prefix if msg.type == "human" else self.ai_prefix
            lines.append(f"{prefix}: {msg.content}")
        return {self.memory_key: "\n".join(lines)}

    # ──────────────── 清空 ────────────────
    def clear(self) -> None:
        """清空所有记忆消息。"""
        self.chat_memory.clear()

    # ──────────────── 辅助信息（便于调试/日志） ────────────────
    def __bool__(self) -> bool:
        """
        避免 Python 经典陷阱：有 __len__ 但无 __bool__ 时，
        Python 会用 __len__() == 0 来判断 False，
        导致 "if not self.memory: return" 这种前置检查误判为"无记忆对象"，
        从而永远写不进去记忆。这里显式返回 True，
        表示"存在一个有效的记忆容器"（不论里面当前有没有内容）。
        """
        return True

    def __len__(self) -> int:
        """返回当前对话轮数（一条 human + 一条 ai = 1 轮）。"""
        return len(self.chat_memory.messages) // 2

    def __repr__(self) -> str:
        return (
            f"ConversationHistory(memory_key={self.memory_key!r}, "
            f"turns={len(self)}, human_prefix={self.human_prefix!r}, "
            f"ai_prefix={self.ai_prefix!r})"
        )