from langchain.tools import BaseTool
import json
from pydantic import BaseModel, Field
import requests
from utils.constants import QWEATHER_BASE_URL, QWEATHER_API_KEY
from utils.logUtils import logger
from rich import print as rprint

class queryLocationIdModel(BaseModel):
    city: str = Field(..., description="要查询的城市名称。")
    district: str | None = Field(
        default=None,
        description="要查询的区县名称。"
    )
class queryLocationId(BaseTool):
    """
    查询城市/区县名称转换为 location ID。
    """
    name: str = "queryLocationId"
    description: str = (
        "根据城市和区县查询 location ID。"
        "城市名称为必填项，区县名称为可选项。"
    )
    args_schema: type[BaseModel] = queryLocationIdModel

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


class queryWeatherModel(BaseModel):
    city: str = Field(..., description="要查询的城市名称。")
    district: str | None = Field(
        default=None,
        description=(
            "要查询的区县名称。"
            "区县名称如果为空，则查询该城市的所有区县天气信息。"
        )
    )


class queryWeather(BaseTool):
    """
    根据城市和区县查询天气信息。
    """
    name: str = "queryWeather"
    description: str = (
        "根据城市和区县查询天气信息。"
        "城市名称为必填项，区县名称为可选项。"
    )
    args_schema: type[BaseModel] = queryWeatherModel

    def _get_location_id(self, city: str, district: str | None = None) -> str | None:
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

    ## curl --compressed -H "X-QW-Api-Key: f2e02edfbb92429689251003040fe401" 'https://n74wcw2f6y.re.qweatherapi.com/v7/weather/now?location=101010100'

    def get_Qweather_info(self, city: str, district: str | None = None) -> str:
        """
        获取天气信息
        :param city: 城市名称
        :param district: 区县名称
        :return: 天气信息
        """
        # 1. 通过 GeoAPI 将城市/区县名称转换为 location ID
        location_id = self._get_location_id(city, district)
        if not location_id:
            return f"[查询失败] 无法找到 {city} {district or ''} 对应的地区编码。"

        # 2. 调用和风天气实时天气接口
        #    对应 curl: curl --compressed -H "X-QW-Api-Key: xxx" '<base>/v7/weather/now?location=<id>'
        headers = {
            "X-QW-Api-Key": QWEATHER_API_KEY,
            "Accept-Encoding": "gzip",
        }
        params = {
            "location": location_id,
        }
        try:
            resp = requests.get(
                f"{QWEATHER_BASE_URL}/v7/weather/now",
                headers=headers,
                params=params,
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json().get("now")
            return json.dumps(data, ensure_ascii=False, indent=2)
        except requests.RequestException as e:
            logger.exception(f"和风天气 API 请求失败: {e}")
            return f"[查询失败] 天气 API 请求出错。错误信息: {str(e)}"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


    def __transform_field(self, field: str) -> str:
        """
        将英文字段名转换为中文。
        :param field: 英文字段名
        :return: 中文字段名
        """
        # 定义字段名映射（英文 -> 中文）
        field_mapping = {
            "code": "状态码",
            "updateTime": "API的最近更新时间",
            "fxLink": "当前数据的响应式页面",
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
            "humidity": "相对湿度(%)",
            "precip": "降水量(mm)",
            "pressure": "站点气压(百帕)",
            "vis": "能见度(km)",
            "cloud": "云量(%)",
            "dew": "露点温度(℃)",
            "refer": "数据来源信息",
            "sources": "数据来源",
            "license": "数据许可"
        }
        return field_mapping.get(field, field)
    def _run(self, city: str, district: str | None = None) -> str:
        logger.info(f"####_run 查询 {city} {district or ''} 的天气信息####")

        try:
            result = self.get_Qweather_info(city, district)
            # 解析 JSON 数据
            data = json.loads(result)

            # 递归转换键名
            def translate_keys(obj):
                if isinstance(obj, dict):
                    return {self.__transform_field(k): translate_keys(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [translate_keys(item) for item in obj]
                else:
                    return obj

            # 执行转换
            translated_data = translate_keys(data)

            # 返回格式化的中文字符串
            return json.dumps(translated_data, ensure_ascii=False, indent=2)
            # return f"[查询结果] {city} {district or ''} 的天气信息：\n{result}"
        except Exception as e:
            logger.exception(f"查询天气信息失败: {e}")
            return f"[查询失败] 天气信息查询出错。错误信息: {str(e)}"

    async def _arun(self, city: str, district: str | None = None) -> str:
        return self._run(city=city, district=district)


if __name__ == '__main__':
    # from db.config import DATABASE_URL
    #
    # db_manager = MysqlDataBaseManager(DATABASE_URL)
    # tool = queryTablesStructure(db=db_manager)
    # print(tool.invoke({'table_name': ['order_items', 'orders']}))
    tool = queryWeather(city="南京")
    rprint(tool._run(city="南京", district="江宁区"))
