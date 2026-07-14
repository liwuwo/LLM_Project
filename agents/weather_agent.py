from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware, ToolRetryMiddleware
from langgraph.graph.state import CompiledStateGraph

from llm.llms import deepseek_llm, local_llm
from utils.constants import WEATHER_PROMPT
from utils.logUtils import logger
from tools.weather_tools import queryWeather


class WeatherAgent:
    """
    天气查询智能体
    能够通过理解用户对天气方面的问题，调用天气查询工具返回结果。
    """

    def __init__(
            self,
            use_local_llm: bool = False,
            max_iterations: int = 20):
        """
        :param use_local_llm:   是否使用本地 LLM 模型，默认使用 DeepSeek 云端模型
        :param max_iterations:  Agent 最大迭代次数（防止死循环）
        """
        self.llm = local_llm if use_local_llm else deepseek_llm
        base_url = getattr(self.llm, 'base_url', None)
        model = getattr(self.llm, 'model_name', None) or getattr(self.llm, 'model', None)
        logger.info(
            f"使用 LLM 模型: {'本地模型' if use_local_llm else 'DeepSeek 云端模型'} "
            f"(model={model!r}, base_url={base_url!r})"
        )
        self.tools = [queryWeather()]
        self.agent_executor = self._create_agent(max_iterations)

    def _create_agent(self, max_iterations: int = 20) -> CompiledStateGraph:
        """创建 ReAct 模式的 Agent。"""
        agent_executor = create_agent(
            model=self.llm,
            tools=self.tools,
            system_prompt=WEATHER_PROMPT.replace("\\n", "\n"),
            response_format=None,
            middleware=[
                ModelCallLimitMiddleware(run_limit=max_iterations),
                ToolRetryMiddleware(max_retries=3)
            ]
        )
        return agent_executor

    def _extract_city_by_llm(self, question: str) -> str:
        """通过 LLM 分析用户问题，提取城市名称"""
        prompt = f"""
        请分析用户的问题，提取其中提到的城市名称。
        
        用户问题：{question}
        
        要求：
        1. 如果问题中明确提到了城市名称（如北京、上海、南京等），请直接返回该城市名称
        2. 如果问题中没有提到城市名称，返回空字符串
        3. 只返回城市名称，不要返回其他内容
        
        示例：
        用户问题："北京今天天气怎么样？" → 返回：北京
        用户问题："明天会下雨吗？" → 返回：（空）
        """

        response = self.llm.invoke(prompt)
        city = str(response.content).strip() if hasattr(response, 'content') else str(response).strip()

        logger.info(f"LLM 提取城市结果: '{city}'")
        return city

    def analyze_weather_question(self, question: str) -> dict:
        """
        通过 LLM 分析用户问题，判断是否为天气相关问题，并提取城市、区县和天气指标

        :param question: 用户问题
        :return: 包含 is_weather、city、district、weather_indices 的字典
        """
        prompt = f"""
        请分析用户的问题，并按要求输出结果。
        用户问题：{question}
        要求：
        1. 首先判断该问题是否与天气相关
        2. 如果是天气相关问题：
           - 提取城市名称（如北京、上海、南京等）
           - 提取区县名称（如果提到的话）
           - 提取用户想查询的天气指标（如温度、湿度、天气状况、风向风力、降水概率等）
        3. 如果不是天气相关问题，请明确说明
        输出格式（JSON）：
        {{
            "is_weather": true/false,
            "city": "城市名称",
            "district": "区县名称（未提及则为空字符串）",
            "weather_indices": ["指标1", "指标2", ...],
            "reason": "判断理由"
        }}
        示例：
        用户问题："北京朝阳区今天天气怎么样？"
        输出：{{"is_weather": true, "city": "北京", "district": "朝阳区", "weather_indices": ["天气状况", "温度"], "reason": "用户询问北京朝阳区的天气情况"}}
        用户问题："明天会下雨吗？"
        输出：{{"is_weather": true, "city": "", "district": "", "weather_indices": ["降水"], "reason": "用户询问是否会下雨"}}
        用户问题："今天吃什么？"
        输出：{{"is_weather": false, "city": "", "district": "", "weather_indices": [], "reason": "用户询问的是饮食问题，与天气无关"}}
        """

        response = self.llm.invoke(prompt)
        content = str(response.content).strip() if hasattr(response, 'content') else str(response).strip()

        logger.info(f"LLM 分析天气问题结果: {content}")

        # 解析 JSON 结果
        import json
        try:
            result = json.loads(content)
        except:
            # 如果解析失败，返回默认值
            result = {
                "is_weather": False,
                "city": "",
                "district": "",
                "weather_indices": [],
                "reason": "解析失败"
            }

        return result

    def query(self, question: dict) -> dict:
        """
        执行天气查询（保持原有 Agent 框架调用逻辑）。
        :param question: 用户的天气查询问题
        :return: 包含查询结果、中间步骤、成功标记的字典
        """
        if not question:
            return {
                'success': False,
                'answer': '查询问题为空，请提供有效的天气查询问题。',
                'intermediate_steps': [],
            }
        messages: str = str(question)
        try:
            logger.info(f"收到天气查询请求: {messages}")
            # 使用 Agent 框架执行查询（保持原有逻辑）
            result = self.agent_executor.invoke({'messages': messages})

            logger.info(f"Agent 原始响应类型: {type(result)}")
            logger.info(f"Agent 原始响应内容: {result}")

            # 提取响应内容
            raw_output = ""
            intermediate_steps = []

            if isinstance(result, dict):
                if 'messages' in result and len(result['messages']) > 0:
                    last_msg = result['messages'][-1]
                    if hasattr(last_msg, 'content'):
                        raw_output = getattr(last_msg, 'content', '')
                    elif isinstance(last_msg, dict):
                        raw_output = last_msg.get('content', '')
                    else:
                        raw_output = str(last_msg)
                elif 'output' in result:
                    raw_output = str(result['output'])
                elif 'agent_outcome' in result:
                    outcome = result['agent_outcome']
                    if isinstance(outcome, dict):
                        raw_output = outcome.get('return_values', {}).get('output', '')
                        if not raw_output:
                            raw_output = outcome.get('summary', '')
                    else:
                        raw_output = str(outcome)
                elif 'response' in result:
                    raw_output = str(result['response'])

                if 'intermediate_steps' in result:
                    intermediate_steps = result['intermediate_steps']
                elif 'steps' in result:
                    intermediate_steps = result['steps']

            if not raw_output:
                raw_output = str(result)

            logger.info(f"提取到的响应内容: '{raw_output}'")
            logger.info(f"中间步骤数量: {len(intermediate_steps)}")

            is_iteration_limit = (
                    'iteration limit' in str(raw_output).lower()
                    or 'time limit' in str(raw_output).lower()
                    or 'stopped due to' in str(raw_output).lower()
            )

            if is_iteration_limit:
                answer = f"查询超时或达到最大迭代次数。当前结果：{raw_output}"
                success = False
            elif not raw_output or raw_output.strip() == '':
                answer = "未获取到有效的天气信息，请尝试重新提问。"
                success = False
            else:
                answer = raw_output.strip()
                success = True

            logger.info(f"天气查询完成，成功: {success}")

            return {
                'success': success,
                'answer': answer,
                'intermediate_steps': intermediate_steps,
            }

        except Exception as e:
            logger.exception(f"天气查询执行失败: {e}")
            return {
                'success': False,
                'answer': f"查询执行失败：{str(e)}",
                'intermediate_steps': [],
            }
