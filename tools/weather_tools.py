from typing import Any

from langchain.tools import BaseTool
import json
from pydantic import BaseModel, Field
import requests
from utils.constants import QWEATHER_BASE_URL, QWEATHER_API_KEY
from utils.logUtils import logger
from rich import print as rprint


class WeatherApi:
    @staticmethod
    def _transform_field(field: str) -> str:
        """
        将英文字段名转换为中文。
        :param field: 英文字段名
        :return: 中文字段名
        """
        # 定义字段名映射（英文 -> 中文）
        field_mapping = {
            "code": "状态码",
            "updateTime": "API的最近更新时间",
            "fxLink": "当前数据的响应式页面，便于嵌入网站或应用",
            "daily": "每日预报数据列表",
            "fxDate": "预报日期",
            "sunrise": "日出时间",
            "sunset": "日落时间",
            "moonrise": "月升时间",
            "moonset": "月落时间",
            "moonPhase": "月相名称",
            "moonPhaseIcon": "月相图标代码",
            "tempMax": "最高温度",
            "tempMin": "最低温度",
            "iconDay": "白天天气状况的图标代码",
            "textDay": "白天天气状况文字描述",
            "iconNight": "夜间天气状况的图标代码",
            "textNight": "晚间天气状况文字描述",
            "wind360Day": "白天风向360角度",
            "windDirDay": "白天风向",
            "windScaleDay": "白天风力等级",
            "windSpeedDay": "白天风速，公里/小时",
            "wind360Night": "夜间风向360角度",
            "windDirNight": "夜间风向",
            "windScaleNight": "夜间风力等级",
            "windSpeedNight": "夜间风速，公里/小时",
            "now": "天气实况数据",
            "obsTime": "观测时间",
            "temp": "温度(℃)",
            "feelsLike": "体感温度(℃)",
            "icon": "天气图标代码",
            "text": "天气状况",
            "wind360": "风向角度",
            "windDir": "风向",
            "windScale": "风力等级",
            "windSpeed": "风速(km/h)",
            "humidity": "相对湿度，百分比数值",
            "precip": "预报当天总降水量，默认单位：毫米",
            "pressure": "站点气压，默认单位：百帕",
            "uvIndex": "紫外线强度指数",
            "vis": "能见度，默认单位：公里",
            "cloud": "云量，百分比数值",
            "dew": "露点温度(℃)",
            "refer": "数据来源和许可信息",
            "sources": "原始数据来源，或数据源说明，可能为空",
            "license": "数据许可或版权声明，可能为空"
        }
        return field_mapping.get(field, field)

    @staticmethod
    def translate_keys(obj):
        if isinstance(obj, dict):
            return {WeatherApi._transform_field(k): WeatherApi.translate_keys(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [WeatherApi.translate_keys(item) for item in obj]
        else:
            return obj

    @staticmethod
    def _get_location_id(city: str, district: str | None = None) -> str | None:
        """
        通过和风天气 GeoAPI 将城市/区县名称转换为 location ID。
        :param city: 城市名称
        :param district: 区县名称（可选）
        :return: location ID，未找到时返回 None
        """
        location_query = district if district else city
        params = {
            "location": location_query
        }
        headers = {
            'X-QW-Api-Key': QWEATHER_API_KEY
        }
        try:
            resp = requests.get(
                f"{QWEATHER_BASE_URL}/geo/v2/city/lookup",
                params=params,
                headers=headers,
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            if data.get("code") == "200" and data.get("location"):
                return data["location"][0]["id"]
            logger.warning(f"GeoAPI 未找到 {location_query} 的 location ID: {data}")
            return None
        except Exception as e:
            logger.exception(f"GeoAPI 查询失败: {e}")
            return None

    @staticmethod
    def get_Qweather_info(url: str, city: str, district: str | None = None) -> Any:
        """
        获取天气信息
        :param city: 城市名称
        :param district: 区县名称
        :return: 天气信息
        """
        # 1. 通过 GeoAPI 将城市/区县名称转换为 location ID
        location_id = WeatherApi._get_location_id(city, district)
        if not location_id:
            return f"[查询失败] 无法找到 {city} {district or ''} 对应的地区编码。"

        # 2. 调用和风天气实时天气接口
        headers = {
            "X-QW-Api-Key": QWEATHER_API_KEY,
            "Accept-Encoding": "gzip",
        }
        params = {
            "location": location_id,
        }
        try:
            resp = requests.get(
                f"{QWEATHER_BASE_URL}{url}",
                headers=headers,
                params=params,
                timeout=10,
            )
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            logger.exception(f"和风天气 API 请求失败: {e}")
            return f"[查询失败] 天气 API 请求出错。错误信息: {str(e)}"


class QueryLocationIdModel(BaseModel):
    city: str = Field(..., description="要查询的城市名称。")
    district: str | None = Field(
        default=None,
        description="要查询的区县名称。"
    )


class QueryLocationId(BaseTool):
    """
    查询城市/区县名称转换为 location ID。
    """
    name: str = "QueryLocationId"
    description: str = (
        "根据城市和区县查询 location ID。"
        "城市名称为必填项，区县名称为可选项。"
    )
    args_schema: type[BaseModel] = QueryLocationIdModel

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _run(self, city: str, district: str | None = None):
        """
        通过和风天气 GeoAPI 将城市/区县名称转换为 location ID。
        :param city: 城市名称
        :param district: 区县名称（可选）
        :return: location ID，未找到时返回 None
        """
        location_query = district if district else city
        params = {
            "location": location_query
        }
        headers = {
            'X-QW-Api-Key': QWEATHER_API_KEY
        }
        try:
            resp = requests.get(
                f"{QWEATHER_BASE_URL}/geo/v2/city/lookup",
                params=params,
                headers=headers,
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            if data.get("code") == "200" and data.get("location"):
                return data["location"][0]["id"]
            logger.warning(f"GeoAPI 未找到 {location_query} 的 location ID: {data}")
            return None
        except Exception as e:
            logger.exception(f"GeoAPI 查询失败: {e}")
            return None

    async def _arun(self, city: str, district: str | None = None):
        return self._run(city, district)


class QueryRealTimeWeatherModel(BaseModel):
    city: str = Field(..., description="要查询的城市名称。")
    district: str | None = Field(
        default=None,
        description=(
            "要查询的区县名称。"
            "区县名称如果为空，则查询该城市的所有区县天气信息。"
        )
    )


class QueryRealTimeWeather(BaseTool):
    """
    根据城市和区县查询实时天气信息。
    """
    name: str = "QueryRealTimeWeather"
    description: str = (
        "根据城市和区县查询实时天气信息。"
        "城市名称为必填项，区县名称为可选项。"
    )
    url: str = "/v7/weather/now"
    args_schema: type[BaseModel] = QueryRealTimeWeatherModel

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _run(self, city: str, district: str | None = None) -> str:
        logger.info(f"####_run 查询 {city} {district or ''} 的天气信息####")
        try:
            res = WeatherApi.get_Qweather_info(self.url, city, district)
            data = res.get("now")
            result = json.dumps(data, ensure_ascii=False, indent=2)
            # 解析 JSON 数据
            data = json.loads(result)

            # 执行转换
            translated_data = WeatherApi.translate_keys(data)

            # 返回格式化的中文字符串
            return json.dumps(translated_data, ensure_ascii=False, indent=2)
            # return f"[查询结果] {city} {district or ''} 的天气信息：\n{result}"
        except Exception as e:
            logger.exception(f"查询实时天气信息失败: {e}")
            return f"[查询失败] 实时天气信息查询出错。错误信息: {str(e)}"

    async def _arun(self, city: str, district: str | None = None) -> str:
        return self._run(city=city, district=district)


class QueryFutureWeatherModel(BaseModel):
    city: str = Field(..., description="要查询的城市名称。")
    district: str | None = Field(
        default=None,
        description=(
            "要查询的区县名称。"
            "区县名称如果为空，则查询该城市的所有区县天气信息。"
        )
    )


class QueryFuture7dWeather(BaseTool):
    """
    根据城市和区县查询未来7天的天气预报信息。
    """
    name: str = "QueryFuture7dWeather"
    description: str = (
        "根据城市和区县查询未来7天的天气预报信息。"
        "城市名称为必填项，区县名称为可选项。"
    )
    args_schema: type[BaseModel] = QueryFutureWeatherModel
    url: str = "/v7/weather/7d"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _run(self, city: str, district: str | None = None) -> str:
        logger.info(f"####_run 查询 {city} {district or ''} 的未来7天天气预报信息####")
        try:
            res = WeatherApi.get_Qweather_info(self.url, city, district)
            data = res.get("daily")
            result = json.dumps(data, ensure_ascii=False, indent=2)
            # 解析 JSON 数据
            data = json.loads(result)

            # 执行转换
            translated_data = WeatherApi.translate_keys(data)

            # 返回格式化的中文字符串
            return json.dumps(translated_data, ensure_ascii=False, indent=2)
            # return f"[查询结果] {city} {district or ''} 的天气信息：\n{result}"
        except Exception as e:
            logger.exception(f"查询未来7天天气预报信息失败: {e}")
            return f"[查询失败] 未来7天天气预报信息查询出错。错误信息: {str(e)}"

    async def _arun(self, city: str, district: str | None = None) -> str:
        return self._run(city=city, district=district)


if __name__ == '__main__':
    # from db.config import DATABASE_URL
    #
    # db_manager = MysqlDataBaseManager(DATABASE_URL)
    # tool = QueryTablesStructure(db=db_manager)
    # print(tool.invoke({'table_name': ['order_items', 'orders']}))
    tool = QueryRealTimeWeather(city="南京")
    rprint(tool._run(city="南京", district="江宁区"))
