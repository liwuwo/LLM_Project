from langchain.agents.middleware import wrap_tool_call

from db.connection_manager import get_weather_db_manager
from utils.logUtils import logger
from langchain_core.messages import ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from collections.abc import Callable
from langgraph.types import Command
from typing import Any
from rich import print as rprint



class WeatherMiddleware:
    def __init__(self):
        conn_mgr = get_weather_db_manager()
        self._store = conn_mgr.get_store()

    def __del__(self):
        pass

    @wrap_tool_call
    def weather_locationId_memory_middleware(request: ToolCallRequest, handler: Callable[
        [ToolCallRequest], ToolMessage | Command[Any]]) -> ToolMessage | Command[Any]:
        tool_call = request.tool_call
        args = tool_call.get("args", {})
        # item = self._store.get(("location_id",),"101190101")
        # rprint(item)
        args["location_id"] = "101190101"
        logger.info(f"tool_call: {tool_call}")

        new_tool_call: Any = {
            **tool_call,
            "args": args
        }
        logger.info(f"### wrap_tool_call ###new_tool_call: {new_tool_call}")
        new_request = request.override(tool_call=new_tool_call)

        request.tool_call["args"]["location_id"] = "101190101"

        return handler(new_request)
