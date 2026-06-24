from langchain_classic.agents import initialize_agent, AgentExecutor, AgentType

from tools.text_SQL_Tools import queryDBTables, queryTablesStructure, validateSQl, executeSQL
from db.mysql_utils import MysqlDataBaseManager
from db.config import DATABASE_URL
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
        logger.info(f"使用 LLM 模型: {'本地模型' if use_local_llm else 'DeepSeek 云端模型'}")

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

        # 4. 创建 Agent
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

    def _clean_react_output(self, text: str) -> str:
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

            # 给 LLM 一个更清晰的工作目标、格式示例和停止条件
            prompt_prefix = (
                "你是一名数据库助手，可以通过调用工具查询数据库。\n"
                "\n【工具列表】"
                "\n  - queryDBTables：查看数据库中有哪些表（第一次查询时强烈建议先调用此工具）"
                "\n  - queryTablesStructure：查看某个或某些表的字段与结构"
                "\n  - validateSQl：验证 SQL 是否合法（仅在你对 SQL 不确定时调用，通常可跳过）"
                "\n  - executeSQL：执行 SELECT 查询，直接返回真实数据"
                "\n\n【重要：输出格式与流程】"
                "\n  你必须严格按以下格式输出，每轮只输出一个 Action（不要多轮写在一起）："
                "\n  "
                "\n  步骤 1 — 思考："
                "\n    Thought: <用中文写一行简短的推理，说明你想做什么、打算查什么>"
                "\n  "
                "\n  步骤 2 — 调用工具（必须严格用 JSON 格式写 Action Input，引号和括号绝对不能省略）："
                "\n    Action: <工具名，只能是 queryDBTables / queryTablesStructure / validateSQl / executeSQL 四选一>"
                "\n    Action Input: {\"参数名\": \"参数值\"}"
                "\n  "
                "\n  【格式示例】"
                '\n    示例 1（查询表结构）：'
                '\n    Thought: 我需要先查看 orders 表的结构，了解订单表的字段。'
                '\n    Action: queryTablesStructure'
                '\n    Action Input: {"table_name": "orders"}'
                '\n  '
                '\n    示例 2（执行 SQL）：'
                '\n    Thought: 现在我有了表结构，可以执行 SQL 查询数据。'
                '\n    Action: executeSQL'
                '\n    Action Input: {"sql": "SELECT * FROM orders LIMIT 10"}'
                "\n  "
                "\n  步骤 3 — 拿到 Observation（工具返回结果）后："
                "\n    - 如果 Observation 已经包含你需要的数据，**立即** 输出 Final Answer，不要再产生任何 Thought/Action。"
                "\n    - 如果数据不够，才能继续下一轮 Thought → Action → Action Input。"
                "\n  "
                "\n  步骤 4 — 给出最终答案："
                "\n    Final Answer: <用中文总结 Observation 的数据，直接回答用户问题，不要使用任何工具调用格式>"
                "\n\n【绝对规则】"
                "\n  1) Action Input 必须是合法 JSON 对象，必须用双引号。例如：{\"sql\": \"SELECT ...\"}"
                "\n  2) 通过 executeSQL 拿到真实数据后，**必须立即输出 Final Answer**，绝对不要再思考或再调用工具。"
                "\n  3) Final Answer 是最终答案，不要再写 Thought/Action/Action Input 等字样。"
                "\n  4) Final Answer 要用中文。"
                f"\n\n用户问题：{question}"
            )

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
                self._print_sql_summary(question)
                return {
                    'success': True,
                    'answer': answer,
                    'partial_answer': True,
                    'intermediate_steps': intermediate_steps,
                }

            # 情况 2：Agent 返回了包含 ReAct 格式的混乱输出 → 触发兜底
            if answer is None:
                logger.warning(
                    f"Agent 输出包含 ReAct 格式，没有正确给出 Final Answer，"
                    f"将基于 executeSQL 结果重新生成答案。"
                )
                answer = self._generate_clean_answer(question, intermediate_steps)
                self._print_sql_summary(question)
                return {
                    'success': True,
                    'answer': answer,
                    'partial_answer': True,
                    'intermediate_steps': intermediate_steps,
                }

            # 情况 3：正常返回了 Final Answer（被 _clean_react_output 已提取）
            logger.info("查询执行完成")
            self._print_sql_summary(question)
            return {
                'success': True,
                'answer': answer,
                'intermediate_steps': intermediate_steps,
            }

        except Exception as e:
            logger.exception(f"查询执行失败: {e}")
            self._print_sql_summary(question)
            return {
                'success': False,
                'error': str(e),
                'answer': f"查询执行失败: {str(e)}",
                'intermediate_steps': [],
            }

    def query_simple(self, question: str) -> str:
        """
        简化版查询，只返回最终答案字符串。
        """
        result = self.query(question)
        return result.get('answer', '查询失败')


# 使用示例
if __name__ == '__main__':
    agent = TextSQLAgent(use_local_llm=False)

    test_questions = [

        "查询哪些会员购买了商品，具体的商品单价和总消费金额是多少"
    ]

    for question in test_questions:
        print('\n' + '=' * 80)
        print(f"问题: {question}")
        print('=' * 80)
        answer = agent.query_simple(question)
        print(f"\n答案:\n{answer}")
        print('\n')