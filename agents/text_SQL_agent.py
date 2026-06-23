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
        max_iterations: int = 10,
        verbose: bool = True,
    ):
        """
        初始化 Text-to-SQL Agent
        :param use_local_llm:   是否使用本地 LLM 模型，默认使用 DeepSeek 云端模型
        :param max_iterations:  Agent 最大迭代次数（防止死循环）
        :param verbose:         是否打印详细执行过程
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

    def _create_agent(self, max_iterations: int = 10, verbose: bool = True) -> AgentExecutor:
        """
        创建 ReAct 模式的 Agent。
        - max_iterations: 最大迭代次数（防止无限循环）
        - verbose: 是否打印每一步的详细信息
        """
        agent_executor = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=verbose,
            max_iterations=max_iterations,
            max_execution_time=60,
            early_stopping_method="generate",
            handle_parsing_errors=(
                "你的输出格式不正确。请严格按以下格式输出：\n"
                "Thought: <你的思考>\n"
                "Action: <工具名，必须是 queryDBTables / queryTablesStructure / validateSQl / executeSQL 之一>\n"
                "Action Input: <合法的 JSON 字符串>\n"
                "Observation: <工具返回结果>\n"
                "...\n"
                "当你有足够信息回答用户问题时，输出：\n"
                "Final Answer: <用中文回答用户的问题>"
            ),
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

    def query(self, question: str) -> dict:
        """
        执行自然语言查询。
        :param question: 用户的自然语言问题
        :return: 包含查询结果、中间步骤、成功标记的字典
        """
        if not question or not question.strip():
            return {
                'success': False,
                'answer': '查询问题为空，请提供有效的自然语言问题。',
                'intermediate_steps': [],
            }

        try:
            logger.info(f"收到查询请求: {question}")

            # 给 LLM 一个更清晰的工作目标和停止条件
            prompt_prefix = (
                "你是一名数据库助手，可以通过调用下面列出的工具来查询数据库。\n"
                "\n【工具列表】"
                "\n  - queryDBTables：查看数据库中有哪些表（第一次查询时强烈建议先调用）"
                "\n  - queryTablesStructure：查看某个或某些表的字段与结构"
                "\n  - validateSQl：验证 SQL 是否合法（仅在你对 SQL 不确定时调用）"
                "\n  - executeSQL：执行 SELECT 查询，直接返回真实数据"
                "\n\n【工作原则】"
                "\n  1) 调用工具之前，先在 Thought 中想清楚：'我想知道什么？需要调用哪个工具？'"
                "\n  2) 当你通过 executeSQL 拿到了查询结果后，立即用中文给出 Final Answer，不要再调用其他工具。"
                "\n  3) 不要反复调用 validateSQl 和 executeSQL 做同样的查询。"
                "\n  4) 当你已经知道答案时，直接输出 Final Answer，不要继续调用工具。"
                "\n\n【输出格式】"
                "\n  Thought: <你的推理>"
                "\n  Action: <工具名>"
                "\n  Action Input: <JSON 参数>"
                "\n  Observation: <工具返回内容>"
                "\n  ..."
                "\n  Final Answer: <用中文回答用户问题>"
                f"\n\n用户问题：{question}"
            )

            result = self.agent_executor.invoke({'input': prompt_prefix})

            raw_output = result.get('output', '')
            intermediate_steps = result.get('intermediate_steps', [])

            # 关键：检测是否因迭代超限被终止
            is_iteration_limit = (
                'iteration limit' in str(raw_output).lower()
                or 'time limit' in str(raw_output).lower()
                or 'stopped due to' in str(raw_output).lower()
            )

            if is_iteration_limit:
                logger.warning(
                    f"Agent 达到迭代/时间上限（已执行 {len(intermediate_steps)} 步），"
                    "将用已收集信息生成回答。"
                )
                answer = self._summarize_from_steps(question, intermediate_steps)
                return {
                    'success': True,
                    'answer': answer,
                    'partial_answer': True,
                    'intermediate_steps': intermediate_steps,
                }

            logger.info("查询执行完成")
            return {
                'success': True,
                'answer': raw_output,
                'intermediate_steps': intermediate_steps,
            }

        except Exception as e:
            logger.exception(f"查询执行失败: {e}")
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