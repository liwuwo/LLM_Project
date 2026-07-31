from langchain.agents.middleware import AgentMiddleware

from db.connection_manager import get_weather_db_manager
from tools.weather_tools import WeatherApi
from utils.logUtils import logger
from langchain_core.messages import ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from collections.abc import Callable
from langgraph.types import Command
from typing import Any
from rich import print as rprint


class WeatherMiddleware(AgentMiddleware):
    def __init__(self):
        super().__init__()
        conn_mgr = get_weather_db_manager()
        self._store = conn_mgr.get_store()

    def wrap_tool_call(self, request: ToolCallRequest, handler: Callable[
        [ToolCallRequest], ToolMessage | Command[Any]]) -> ToolMessage | Command[Any]:
        tool_call = request.tool_call
        args = tool_call.get("args", {})
        logger.info(f"args: {args}")
        city = str(args.get("city"))
        # 如果args中的district为空，则默认值设为all
        district = args.get("district", "all")
        item = self._store.get((city,), district)
        rprint(item)
        if item is None:
            location_id = WeatherApi.get_location_id(city, str(args.get("district")))
            self._store.put((city,), district, {"location_id": location_id})
        else:
            args["location_id"] = str(item.value["location_id"])
        logger.info(f"tool_call: {tool_call}")

        new_tool_call: Any = {
            **tool_call,
            "args": args
        }
        logger.info(f"### wrap_tool_call ###new_tool_call: {new_tool_call}")
        new_request = request.override(tool_call=new_tool_call)

        request.tool_call["args"]["location_id"] = args["location_id"]

        return handler(new_request)
