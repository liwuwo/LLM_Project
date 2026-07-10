from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware, ToolRetryMiddleware
from langgraph.graph.state import CompiledStateGraph

from llm.llms import deepseek_llm, local_llm
from utils.constants import WEATHER_PROMPT
from utils.logUtils import logger
class WeatherAgent:
    """
    天气查询智能体
    能够通过理解用户对天气方面的问题，调用天气查询工具返回结果。

    具体的调用流程：
    1. 用户输入天气查询问题（如“南京天气”、“南京天气怎么样”等）
    2. 智能体理解用户问题，调用天气查询工具获取天气信息
    3. 智能体将天气信息返回给用户

    工具的调用流程：
    1.
    2. 天气查询工具返回天气信息
    3. 智能体将天气信息返回给用户
    """

    def __init__(
            self,
            use_local_llm: bool = False,
            max_iterations: int = 20,
            verbose: bool = True):
        """
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

    def _create_agent(self,max_iterations: int = 20, verbose: bool = True) -> CompiledStateGraph:
        """
        创建 ReAct 模式的 Agent。
        - max_iterations: 最大迭代次数（防止无限循环）
        - verbose: 是否打印每一步的详细信息
        """
        agent_executor = create_agent(
            model=self.llm,
            tools=self.tools,
            system_prompt=WEATHER_PROMPT.replace("\\n", "\n"),
            response_format=None,
            middleware=[
                ModelCallLimitMiddleware(run_limit=max_iterations),
                ToolRetryMiddleware(max_retries=3)
            ]

            # agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            # verbose=verbose,
            # max_iterations=max_iterations,
            # max_execution_time=120,
            # early_stopping_method="generate",
            # handle_parsing_errors=_handle_parsing_error,
            # return_intermediate_steps=True,
        )
        return agent_executor
