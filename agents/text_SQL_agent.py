from langchain_classic.agents import initialize_agent, AgentExecutor, AgentType
from agents.conversation_memory import ConversationHistory

from tools.text_SQL_Tools import queryDBTables, queryTablesStructure, validateSQl, executeSQL
from db.mysql_utils import MysqlDataBaseManager
from db.config import DATABASE_URL
from utils.constants import AGENT_PROMPT, AGENT_MEMORY_ENABLED, AGENT_MAX_MEMORY_TURNS
from llm.llms import deepseek_llm, local_llm
from utils.logUtils import logger


class TextSQLAgent:
    """
    文本转SQL智能体
    能够理解用户的自然语言问题，自动转换为SQL查询并返回结果。

    典型的工具调用流程：
        1) 调用 queryDBTables        -> 了解数据库中有哪些表
        2) 调用 queryTablesStructure -> 了解相关表的字段与结构
        3) 调用 validateSQl          -> 检查生成的 SQL 合法性
        4) 调用 executeSQL           -> 执行 SQL 获取结果
    """

    def __init__(
            self,
            use_local_llm: bool = False,
            max_iterations: int = 20,
            verbose: bool = True,
    ):
        """
        初始化 Text-to-SQL Agent
        :param use_local_llm:   是否使用本地 LLM 模型，默认使用 DeepSeek 云端模型
        :param max_iterations:  Agent 最大迭代次数（防止死循环）
        :param verbose:         是否打印每一步的详细执行过程
        """
        # 1. 选择 LLM 模型
        self.llm = local_llm if use_local_llm else deepseek_llm
        # 打印精确配置：避免"以为用了本地模型但其实连的是云端"这类排查陷阱
        base_url = getattr(self.llm, 'base_url', None)
        model = getattr(self.llm, 'model_name', None) or getattr(self.llm, 'model', None)
        logger.info(
            f"使用 LLM 模型: {'本地模型' if use_local_llm else 'DeepSeek 云端模型'} "
            f"(model={model!r}, base_url={base_url!r})"
        )

        # 2. 初始化数据库管理器（失败会抛出异常）
        self.db_manager = MysqlDataBaseManager(DATABASE_URL)
        logger.info("数据库连接初始化完成")

        # 3. 创建工具列表（按工具调用顺序排列，帮助 LLM 理解）
        self.tools = [
            queryDBTables(db=self.db_manager),
            queryTablesStructure(db=self.db_manager),
            validateSQl(db=self.db_manager),
            executeSQL(db=self.db_manager),
        ]
        logger.info(f"注册工具: {[t.name for t in self.tools]}")

        # 4. 初始化对话记忆
        #    采用零依赖的 ConversationHistory（自实现，接口与 LangChain
        #    ConversationBufferMemory 完全兼容，避免因 langchain.memory 缺失而降级）
        self.memory_enabled = AGENT_MEMORY_ENABLED
        self.max_memory_turns = max(1, int(AGENT_MAX_MEMORY_TURNS))
        if self.memory_enabled:
            self.memory = ConversationHistory(
                memory_key="chat_history",
                return_messages=False,  # 返回字符串（"用户: xxx\n助手: xxx"）而非消息对象
                human_prefix="用户",  # 自定义中文前缀，让 Prompt 更自然
                ai_prefix="助手",
            )
            logger.info(f"对话记忆已启用（最多保留 {self.max_memory_turns} 轮）")
        else:
            self.memory = None
            logger.info("对话记忆已禁用（每次查询独立，配置 AGENT_MEMORY_ENABLED=False）")

        # 5. 创建 Agent
        self.agent_executor = self._create_agent(
            max_iterations=max_iterations,
            verbose=verbose,
        )
        logger.info("TextSQLAgent 初始化完成")

    def _create_agent(self, max_iterations: int = 20, verbose: bool = True) -> AgentExecutor:
        """
        创建 ReAct 模式的 Agent。
        - max_iterations: 最大迭代次数（防止无限循环）
        - verbose: 是否打印每一步的详细信息
        """

        def _handle_parsing_error(error: Exception) -> str:
            """
            自定义解析错误处理器。当 LLM 输出的 Action Input 不是合法 JSON 时：
            尝试从原始文本中提取参数内容，作为提示反馈给 LLM，让它下次正确输出。
            """
            err_str = str(error)
            # 提取 LLM 的原始输出（通常包含在错误信息中）
            hint = "请严格按以下格式输出，不要省略 JSON 括号和引号："
            hint += "\n  Action: executeSQL"
            hint += '\n  Action Input: {"sql": "SELECT ... FROM ..."}'
            hint += "\n\n切记：Action Input 必须是合法的 JSON 对象，包含 'sql' 字段。"
            return hint

        agent_executor = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=verbose,
            max_iterations=max_iterations,
            max_execution_time=120,
            early_stopping_method="generate",
            handle_parsing_errors=_handle_parsing_error,
            return_intermediate_steps=True,
        )
        return agent_executor

    @staticmethod
    def _summarize_from_steps(question: str, intermediate_steps: list) -> str:
        """
        当 Agent 因为迭代超限被终止时，从已有中间步骤中提取信息生成回答。
        这样用户不会只看到 "Agent stopped due to iteration limit or time limit."
        """
        if not intermediate_steps:
            return "未能在规定步骤内得到完整答案，请尝试换一种更简单的提问方式。"

        lines = [f"问题：{question}"]
        lines.append("（Agent 在规定步骤内未能给出完整回答，以下是已收集到的信息：）")
        lines.append("")

        sql_results = []
        table_info = []

        for step_idx, (action, observation) in enumerate(intermediate_steps, 1):
            tool_name = getattr(action, 'tool', 'unknown')
            tool_input = getattr(action, 'tool_input', '')

            lines.append(f"步骤 {step_idx}：调用工具 {tool_name}")
            if tool_input:
                lines.append(f"  输入参数：{tool_input}")
            lines.append(f"  返回结果：{str(observation)[:300]}")
            lines.append("")

            # 收集关键信息
            if tool_name == 'executeSQL':
                sql_results.append((tool_input, str(observation)))
            elif tool_name == 'queryDBTables':
                table_info.append(str(observation))

        lines.append("")
        lines.append("建议：你可以直接基于上述 executeSQL 返回的数据来获取你想要的信息。")
        return "\n".join(lines)

    def _generate_clean_answer(self, question: str, intermediate_steps: list) -> str:
        """
        兜底逻辑：当 Agent 没有正确给出 Final Answer 时，
        从 intermediate_steps 中找到最后一次成功的 executeSQL 结果，
        直接让 LLM 基于该数据生成简洁回答。
        """
        if not intermediate_steps:
            return "未能获取到查询结果，请尝试换一种提问方式。"

        # 反向查找：收集所有 executeSQL 的执行结果（优先取最后一个）
        sql_executions = []
        for action, observation in intermediate_steps:
            tool_name = getattr(action, 'tool', '')
            tool_input = getattr(action, 'tool_input', '')
            if tool_name == 'executeSQL' and observation:
                sql_executions.append((str(tool_input), str(observation)))

        if not sql_executions:
            # 没有任何 SQL 执行结果，返回通用提示
            return self._summarize_from_steps(question, intermediate_steps)

        # 取最后一次（通常是最终的）SQL 执行结果
        last_sql, last_result = sql_executions[-1]

        # 直接调用 LLM 基于数据生成简洁回答（绕过 ReAct 循环）
        fallback_prompt = (
            "以下是对数据库的查询结果。请基于这些数据用中文回答用户问题。\n\n"
            f"【用户问题】{question}\n\n"
            f"【SQL 查询】{last_sql}\n\n"
            f"【查询结果】\n{last_result}\n\n"
            "请直接给出简洁明了的中文回答，以'根据查询结果：'开头，直接陈述数据和结论，不要调用任何工具。"
        )

        try:
            summary = self.llm.invoke(fallback_prompt)
            # ChatOpenAI 返回 AIMessage 对象，需要提取 content
            if hasattr(summary, 'content'):
                return summary.content
            return str(summary)
        except Exception as e:
            # LLM 调用失败，直接返回格式化后的执行结果
            lines = [f"根据数据库查询结果：\n"]
            lines.append(f"执行的 SQL：{last_sql}")
            lines.append("")
            lines.append(last_result[:1500])
            return "\n".join(lines)

    def _clean_react_output(self, text: str) -> str | None:
        """
        清理答案中的 ReAct 格式标记（Thought/Action/Action Input 等），仅保留有意义的内容。
        """
        if not text:
            return text

        import re

        # 如果已经是干净的 Final Answer，直接提取
        if 'Final Answer:' in text:
            idx = text.rfind('Final Answer:')
            cleaned = text[idx + len('Final Answer:'):].strip()
            return cleaned

        # 检测是否包含未完成的 ReAct 输出
        has_thought = bool(re.search(r'(?im)^\s*Thought\s*[:：]', text))
        has_action = bool(re.search(r'(?im)^\s*Action\s*[:：]', text))

        if has_thought or has_action:
            # 答案中夹杂 ReAct 格式，标记为需要兜底处理
            return None  # 返回 None 表示需要触发兜底逻辑

        return text.strip()

    def _print_sql_summary(self, question: str):
        """
        从 db_manager 取出并清空本次请求的 SQL 执行历史，
        按步骤在控制台打印清晰的执行总结。
        """
        try:
            history = self.db_manager.get_and_clear_sql_history()
        except Exception as e:
            logger.warning(f"获取 SQL 执行历史失败: {e}")
            return

        print()
        print("=" * 78)
        print(f"【SQL 执行总结】问题: {question}")
        print("=" * 78)

        if not history:
            print("  （本次未执行任何 SQL / 校验操作）")
            print("=" * 78)
            print()
            return

        total = len(history)
        success_count = 0
        for idx, record in enumerate(history, 1):
            raw_input = record.get("raw_input", "")
            sql = record.get("sql", "") or "(未能解析出有效 SQL)"
            kind = record.get("kind", "execute")
            success = record.get("success", False)
            rows = record.get("rows")
            error = record.get("error")

            if success:
                success_count += 1
                status = "✓ 成功"
            else:
                status = "✗ 失败"

            print(f"  步骤 {idx}/{total}：[{kind.upper()}]  {status}")

            # 如果原始输入看起来就已经是干净 SQL，就不额外打印 raw_input
            raw_str = str(raw_input).strip()
            if raw_str and raw_str != sql and not raw_str.upper().startswith(("SELECT", "WITH", "SHOW")):
                # 原始输入可能是 LLM 的嵌套 JSON，有排查价值
                print(f"    · LLM 原始输入: {raw_str}")

            # 多行 SQL 做缩进处理
            sql_lines = sql.splitlines() if sql else [""]
            if len(sql_lines) == 1:
                print(f"    · SQL: {sql_lines[0]}")
            else:
                print("    · SQL:")
                for line in sql_lines:
                    print(f"        {line}")

            if rows is not None:
                print(f"    · 返回条数: {rows}")
            if error:
                # 错误信息通常较长，换行显示
                err_lines = str(error).splitlines()
                print("    · 错误信息:")
                for line in err_lines:
                    print(f"        {line}")
            print("-" * 78)

        print(
            f"  共执行 {total} 条 SQL 相关操作，"
            f"成功 {success_count} 条，失败 {total - success_count} 条。"
        )
        print("=" * 78)
        print()

    def _save_to_memory(self, question: str, answer: str):
        """
        将一次问答写入对话记忆。
        - 会自动做长度截断，防止写入过长的 SQL 结果摘要污染后续对话
        - 写入后调用 _trim_memory() 确保不超过最大轮数
        - 如果 memory 未启用，直接返回不做任何操作
        """
        # 用显式 is None 检查而非隐式 bool 判断：
        # 因为记忆对象本身实现了 __len__，空状态不代表对象不存在
        if not self.memory_enabled or self.memory is None:
            return

        # 防止超长答案污染历史：超过 500 字符就截断，保留开头语义
        safe_answer = str(answer)
        if len(safe_answer) > 500:
            safe_answer = safe_answer[:500] + "..."

        try:
            self.memory.save_context({"input": str(question)}, {"output": safe_answer})
            # 写入后做轮数裁剪
            self._trim_memory()
            logger.info(f"已写入对话记忆，当前共 {self.get_memory_turns()} 轮")
        except Exception as e:
            logger.warning(f"写入对话记忆失败（不影响本次查询）: {e}")

    def _trim_memory(self):
        """
        确保记忆不超过 AGENT_MAX_MEMORY_TURNS 轮。
        ConversationBufferMemory 不自带轮数限制，这里手动实现：
        每条"用户提问 + 助手回答"算 1 轮，对应 2 条消息。
        超过限制时，丢弃最早的一轮。
        """
        if self.memory is None:
            return
        max_messages = self.max_memory_turns * 2  # 每轮 = Human + AI = 2 条消息
        messages = self.memory.chat_memory.messages
        while len(messages) > max_messages:
            messages.pop(0)
            messages.pop(0)
            logger.info("对话记忆已满，丢弃最早 1 轮对话以控制上下文长度")

    def _format_chat_history_for_prompt(self) -> str:
        """
        从 memory 读取历史，格式化为可直接拼入 Prompt 的文本块。
        返回值示例（不为空时）：
            用户: 帮我查一下会员表
            助手: 根据查询结果，会员表有 xxx 条记录...
            用户: 那再帮我看一下最近一周的消费
            助手: ...
        如果没有历史对话或记忆未启用，返回空字符串。
        """
        if not self.memory_enabled or self.memory is None:
            return ""

        history_str = self.memory.load_memory_variables({}).get("chat_history", "")
        history_str = str(history_str).strip()
        if not history_str:
            return ""

        # 格式化为 Prompt 中的一个清晰段落
        return (
            "以下是你与用户之前的对话内容，请作为上下文参考。"
            "如果用户的问题与之前的对话相关（例如指代'刚才的结果''那些会员'等），"
            "请基于这段历史理解用户的真实意图，再决定如何查询数据库。\n"
            f"{history_str}\n"
            "（以上是对话历史，接下来请处理用户的新问题）"
        )

    def clear_memory(self):
        """
        清空对话记忆。适用于：用户切换话题、主动要求"重新开始"、或检测到上下文混乱时。
        """
        if self.memory:
            self.memory.clear()
            logger.info("对话记忆已清空")

    def get_memory_turns(self) -> int:
        """
        返回当前记忆中的对话轮数。用于 UI 显示状态或调试。
        1 轮 = 1 次用户提问 + 1 次助手回答。
        ConversationHistory 已实现 __len__，可直接返回 len(self.memory)。
        """
        if self.memory is None:
            return 0
        return len(self.memory)

    def query(self, question: str) -> dict:
        """
        执行自然语言查询。
        :param question: 用户的自然语言问题
        :return: 包含查询结果、中间步骤、成功标记的字典
        """
        if not question or not question.strip():
            # 空问题也要打印一次（虽然通常没有 SQL）
            self._print_sql_summary(question)
            return {
                'success': False,
                'answer': '查询问题为空，请提供有效的自然语言问题。',
                'intermediate_steps': [],
            }

        # 每次新提问前先清空 SQL 执行历史，避免污染下次的总结
        self.db_manager.get_and_clear_sql_history()

        try:
            logger.info(f"收到查询请求: {question}")

            # ── 第一步：从 LangChain Memory 读取历史并拼入 Prompt ──
            prompt_template = AGENT_PROMPT.replace("\\n", "\n")
            chat_history_block = self._format_chat_history_for_prompt()

            # 如果 chat_history_block 为空，把 {chat_history} 占位符所在的整段也清理掉
            # 避免 Prompt 中出现空的"【对话历史】"标题
            if chat_history_block:
                final_prompt = prompt_template.replace("{chat_history}", chat_history_block)
            else:
                # 没有历史：移除 "【对话历史】\n{chat_history}\n" 及其前后的空行
                import re as _re
                final_prompt = _re.sub(
                    r'\n*【对话历史】\n\{chat_history\}\n*',
                    '\n',
                    prompt_template
                )

            # 末尾拼接当前用户问题（保持与旧版本一致的位置）
            prompt_prefix = f"{final_prompt}\n\n用户问题：{question}"

            result = self.agent_executor.invoke({'input': prompt_prefix})

            raw_output = result.get('output', '')
            intermediate_steps = result.get('intermediate_steps', [])

            # 检测是否因迭代/时间超限被终止
            is_iteration_limit = (
                    'iteration limit' in str(raw_output).lower()
                    or 'time limit' in str(raw_output).lower()
                    or 'stopped due to' in str(raw_output).lower()
            )

            # 清理答案：尝试从输出中提取 Final Answer
            answer = self._clean_react_output(str(raw_output))

            # 情况 1：迭代超限 → 用兜底逻辑生成答案
            if is_iteration_limit:
                logger.warning(
                    f"Agent 达到迭代/时间上限（已执行 {len(intermediate_steps)} 步），"
                    "触发兜底逻辑生成答案。"
                )
                answer = self._generate_clean_answer(question, intermediate_steps)
                # ── 写入记忆（迭代超限也要记录，下次可做上下文衔接）──
                self._save_to_memory(question, answer)
                self._print_sql_summary(question)
                return {
                    'success': True,
                    'answer': answer,
                    'partial_answer': True,
                    'intermediate_steps': intermediate_steps,
                    'memory_turns': self.get_memory_turns(),
                }

            # 情况 2：Agent 返回了包含 ReAct 格式的混乱输出 → 触发兜底
            if answer is None:
                logger.warning(
                    f"Agent 输出包含 ReAct 格式，没有正确给出 Final Answer，"
                    f"将基于 executeSQL 结果重新生成答案。"
                )
                answer = self._generate_clean_answer(question, intermediate_steps)
                # ── 写入记忆 ──
                self._save_to_memory(question, answer)
                self._print_sql_summary(question)
                return {
                    'success': True,
                    'answer': answer,
                    'partial_answer': True,
                    'intermediate_steps': intermediate_steps,
                    'memory_turns': self.get_memory_turns(),
                }

            # 情况 3：正常返回了 Final Answer（被 _clean_react_output 已提取）
            logger.info("查询执行完成")
            # ── 写入记忆 ──
            self._save_to_memory(question, answer)
            self._print_sql_summary(question)
            return {
                'success': True,
                'answer': answer,
                'intermediate_steps': intermediate_steps,
                'memory_turns': self.get_memory_turns(),
            }

        except Exception as e:
            # ── 按异常类型给出更有针对性的中文提示 ──
            err_str = str(e)
            err_lower = err_str.lower()

            if ("connection error" in err_lower
                    or "remote protocol error" in err_lower
                    or "server disconnected" in err_lower
                    or "temporary failure in name resolution" in err_lower
                    or "connecttimeout" in err_lower):
                answer = (
                    "❌ 无法连接到 LLM 服务（网络连接异常）。\n"
                    "   请检查：\n"
                    "     1. 当前机器是否能访问 api.deepseek.com（可在终端用 curl 测试）\n"
                    "     2. 是否需要使用代理访问外网\n"
                    "     3. 尝试使用本地模型：创建 Agent 时传 use_local_llm=True"
                )
            elif ("401" in err_lower
                  or "invalid api key" in err_lower
                  or "unauthorized" in err_lower):
                answer = "❌ API Key 无效或已过期，请在 .env 中更新 DEEPSEEK_API_KEY。"
            elif ("429" in err_lower
                  or "rate limit" in err_lower
                  or "insufficient quota" in err_lower):
                answer = "❌ API 调用频次超限 / 额度不足，请稍后再试或更换 API Key。"
            elif "timeout" in err_lower:
                answer = "❌ 模型响应超时，请稍后再试；如持续出现请换用本地模型（use_local_llm=True）。"
            else:
                answer = f"查询执行失败: {err_str}"

            # 原始异常写日志（便于排查）
            logger.exception(f"查询执行失败: {err_str}")

            # 简短错误写入记忆，避免污染上下文
            self._save_to_memory(question, f"查询失败：{answer[:80]}")
            self._print_sql_summary(question)
            return {
                'success': False,
                'error': err_str,
                'answer': answer,
                'intermediate_steps': [],
                'memory_turns': self.get_memory_turns(),
            }

    def query_simple(self, question: str) -> str:
        """
        简化版查询，只返回最终答案字符串。
        """
        result = self.query(question)
        return result.get('answer', '查询失败')


# 使用示例
if __name__ == '__main__':
    # agent = TextSQLAgent(use_local_llm=False)
    #
    # test_questions = [
    #
    #     "查询哪些会员购买了商品，具体的商品单价和总消费金额是多少"
    # ]
    #
    # for question in test_questions:
    #     print('\n' + '=' * 80)
    #     print(f"问题: {question}")
    #     print('=' * 80)
    #     answer = agent.query_simple(question)
    #     print(f"\n答案:\n{answer}")
    #     print('\n')
    print(AGENT_PROMPT)
