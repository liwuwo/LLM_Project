from langchain.tools import BaseTool

from db.mysql_utils import MysqlDataBaseManager
from utils.logUtils import logger


class queryDBTables(BaseTool):
    """
    文本转SQL工具类
    """
    name: str = "queryDBTables"

    description: str = "查询数据库中所有的表名和各个表对应的备注信息"

    db: MysqlDataBaseManager

    def _run(self):
        try:
            table_list: list[dict] = self.db.get_alltables_comments()
            result: str = f"数据库中共有{len(table_list)}个表，表名和备注信息如下：\n"
            for i, table in enumerate(table_list):
                table_name = table['table_name']
                table_comment = table['table_comment']
                if not table_comment or table_comment.isspace():
                    table_comment = "（暂无备注）"
                result += f"第{i + 1}个表。表名：{table_name}，备注信息：{table_comment}\n"
            return result
        except Exception as e:
            logger.exception(f"查询数据库表信息失败: {e}")
            return f"查询数据库表信息时发生错误: {str(e)}"

    async def _arun(self):
        return self._run()


if __name__ == '__main__':

    from db.config import DATABASE_URL

    db_manager = MysqlDataBaseManager(DATABASE_URL)
    tool = queryDBTables(db=db_manager)
    print(tool.invoke({}))
