from langgraph.runtime import Runtime
from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import ModelCallLimitMiddleware, ToolRetryMiddleware, SummarizationMiddleware, \
    PIIMiddleware, before_model, before_agent, wrap_tool_call
from langchain.messages import RemoveMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.graph.state import CompiledStateGraph

from db.connection_manager import get_weather_db_manager
from llm.llms import deepseek_llm, local_llm
from middleware.weather_middleware import WeatherMiddleware
from utils.constants import WEATHER_PROMPT
from utils.logUtils import logger
from tools.weather_tools import QueryRealTimeWeather, QueryFuture7dWeather


@before_model
def trim_messages(state: AgentState, runtime: Runtime) -> dict[str, list] | None:
    messages = state["messages"]

    return {"messages": [
        RemoveMessage(id=REMOVE_ALL_MESSAGES),
        *messages[-4:]
    ]}


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
        self.tools = [
            QueryRealTimeWeather(),
            QueryFuture7dWeather(),
        ]
        # 初始化 checkpoint（保持连接打开）
        self._checkpoint_ctx = None
        self._create_checkpoint()
        self.agent_executor = self._create_agent(max_iterations)

    def _create_checkpoint(self):
        """通过单例 WeatherDBConnectionManager 获取 checkpoint 连接"""
        try:
            conn_mgr = get_weather_db_manager()
            self.checkpoint = conn_mgr.get_checkpoint()
            # 注意：checkpoint/store 的连接生命周期由单例统一管理（含 atexit 关闭），
            # WeatherAgent 不再单独维护 ctx，避免提前关闭影响 WeatherMiddleware 等共享方。
            self._checkpoint_ctx = None
        except Exception:
            self._checkpoint_ctx = None
            self.checkpoint = None

    def _create_agent(self, max_iterations: int = 20) -> CompiledStateGraph:
        weather_middleware = WeatherMiddleware()
        agent_executor = create_agent(
            model=self.llm,
            tools=self.tools,
            checkpointer=self.checkpoint,
            system_prompt=WEATHER_PROMPT.replace("\\n", "\n"),
            response_format=None,
            middleware=[
                trim_messages,
                weather_middleware,
                SummarizationMiddleware(
                    model=self.llm,
                    trigger=[
                        ("tokens", 200),
                        ("messages", 5)
                    ],
                    keep=("messages", 3)
                ),
                ModelCallLimitMiddleware(run_limit=max_iterations),
                ToolRetryMiddleware(max_retries=3)
            ]
        )
        """
        创建 ReAct 模式的 Agent。
        - max_iterations: 最大迭代次数（防止死循环）
        """
        return agent_executor

    @staticmethod
    def _normalize_location_suffix(city: str, district: str) -> tuple[str, str]:
        """
        代码兜底：规范化城市、区县级行政单位后缀。
        - 城市：移除城市名称中的"市"后缀（如"北京市" -> "北京"）
        - 区县：移除区县名称中的"区"后缀（如"海淀区" -> "海淀"，如"当涂县" -> "当涂"）
        提示词层之后的强保障，保证输出规则 100% 一致。
        """
        CITY_LEGAL_SUFFIXES = ("特别行政区", "自治区", "自治州", "市", "省", "盟")
        DISTRICT_LEGAL_SUFFIXES = ("自治县", "林区", "特区", "市", "区", "县", "旗")

        def _strip_suffix(s: str, suffixes: tuple[str, ...]) -> str:
            text = s.strip() if s else ""
            if not text:
                return ""
            for suf in suffixes:
                if text.endswith(suf):
                    return text[: -len(suf)]
            return text

        norm_city = _strip_suffix(city, CITY_LEGAL_SUFFIXES)
        norm_district = _strip_suffix(district, DISTRICT_LEGAL_SUFFIXES)

        return norm_city, norm_district

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
            "city": "城市名称（必须后缀要为“市”，如：北京市，上海市，帮我标识清楚）",
            "district": "区县名称（未提及则为空字符串，如果提及，必须后缀要为“区县”写清楚，如：朝阳区，当涂县，帮我标识清楚）",
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
        logger.info(f"analyze_weather_question 分析天气问题结果: {content}")
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
        raw_city = str(result.get("city", "") or "")
        raw_district = str(result.get("district", "") or "")
        norm_city, norm_district = WeatherAgent._normalize_location_suffix(raw_city, raw_district)
        if norm_city != raw_city or norm_district != raw_district:
            logger.info(f"行政后缀规范化：city {raw_city!r}→{norm_city!r}, district {raw_district!r}→{norm_district!r}")
        result["city"] = norm_city
        result["district"] = norm_district
        return result

    def query(self, question: dict) -> dict:
        """
        执行天气查询。
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
            configurable: RunnableConfig = {
                "configurable": {
                    "thread_id": "1"
                }
            }
            result = self.agent_executor.invoke(
                {'messages': messages},
                config=configurable
            )

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


if __name__ == "__main__":
    weather_agent = WeatherAgent()
    result = weather_agent.query({
        "is_weather": True,
        "city": "苏州市",
        "district": "昆山市",
        "weather_indices": ["天气状况", "温度"],
        "reason": "用户询问苏州昆山今天的天气情况"
    })
    print(result)
