from langchain.tools import BaseTool
import json

from pydantic import BaseModel, Field

from db.mysql_utils import MysqlDataBaseManager
from utils.logUtils import logger


class queryDBTablesModel(BaseModel):
    pass


class queryDBTables(BaseTool):
    """
    文本转SQL工具类
    """
    name: str = "queryDBTables"

    description: str = "查询数据库中所有的表名和各个表对应的备注信息"

    db: MysqlDataBaseManager

    args_schema: type[BaseModel] = queryDBTablesModel

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

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


class queryDBColumnsModel(BaseModel):
    table_name: list[str] | None = Field(default=None, description="表名的列表,用于查询数据库中这些指定的表名的表结构，不传则查询所有表")


class queryTablesStructure(BaseTool):
    """
    查询数据库表结构
    """
    name: str = "queryTablesStructure"
    description: str = "根据输入的表名列表，查询数据库中指定的这些表名的表结构"
    db: MysqlDataBaseManager
    args_schema: type[BaseModel] = queryDBColumnsModel

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _run(self, table_name: list[str]) -> str:
        try:
            table_structure: str = self.db.get_table_constructions(table_name)
            if not table_structure:
                return f"数据库中不存在表{table_name}，请检查输入的表名是否正确"
            # table_structure为json数组格式
            tables_data: dict = json.loads(table_structure)
            result_parts: list[str] = []
            for tbl_key, tbl_info in tables_data.items():
                table_name_str = tbl_info.get('table_name', tbl_key)
                comment = tbl_info.get('comment', '') or '（暂无备注）'
                columns: list[dict] = tbl_info.get('columns', [])
                primary_keys: list[str] = tbl_info.get('primary_keys', [])
                foreign_keys: list[dict] = tbl_info.get('foreign_keys', [])
                indexes: list[dict] = tbl_info.get('indexes', [])

                # 构建表头信息
                part = f"表名：{table_name_str}，备注：{comment}\n"

                # 构建字段信息
                part += "字段列表：\n"
                for col in columns:
                    col_name = col.get('name', '')
                    col_type = col.get('type', '')
                    col_nullable = '可为空' if col.get('nullable') else '不可为空'
                    col_default = col.get('default') if col.get('default') else '无'
                    col_comment = col.get('comment', '') or '无'
                    part += f"  字段名：{col_name}，类型：{col_type}，{col_nullable}，默认值：{col_default}，说明：{col_comment}\n"

                # 构建主键信息
                if primary_keys:
                    part += f"主键：{', '.join(primary_keys)}\n"
                else:
                    part += "主键：无\n"

                # 构建外键信息
                if foreign_keys:
                    part += "外键：\n"
                    for fk in foreign_keys:
                        fk_name = fk.get('name', '')
                        constrained = ', '.join(fk.get('constrained_columns', []))
                        referred_table = fk.get('referred_table', '')
                        referred_cols = ', '.join(fk.get('referred_columns', []))
                        part += f"  外键名：{fk_name}，本表字段：{constrained}，关联表：{referred_table}，关联字段：{referred_cols}\n"
                else:
                    part += "外键：无\n"

                # 构建索引信息
                if indexes:
                    part += "索引：\n"
                    for idx in indexes:
                        idx_name = idx.get('name', '')
                        idx_cols = ', '.join(idx.get('column_names', []))
                        idx_unique = '是' if idx.get('unique') else '否'
                        part += f"  索引名：{idx_name}，包含字段：{idx_cols}，是否唯一：{idx_unique}\n"
                else:
                    part += "索引：无\n"

                result_parts.append(part)

            return f"共查询到{len(result_parts)}张表的结构信息：\n" + '\n'.join(result_parts)

        except Exception as e:
            logger.exception(f"查询数据库表结构失败: {e}")
            return f"查询数据库表结构时发生错误: {str(e)}"

    async def _arun(self, table_name: list[str]) -> str:
        return self._run(table_name)


class validateSQlModel(BaseModel):
    sql: str = Field(..., description="要进行验证的SQL语句，校验SQL语句是否符合MySQL方言规则，并确保只允许查询操作")


class validateSQl(BaseTool):
    """
    验证SQL语句
    """
    name: str = "validateSQl"
    description: str = "根据输入的SQL语句，验证SQL语句的合法性，校验SQL语句是否符合MySQL方言规则，并确保只允许查询操作"
    db: MysqlDataBaseManager
    args_schema: type[BaseModel] = validateSQlModel

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _run(self, sql: str) -> str:
        try:
            result: dict = self.db.validate_sql(sql)
            is_valid = result.get('valid', False)
            message = result.get('message', '')
            sql_type = result.get('sql_type', '')
            sanitized_sql = result.get('sanitized_sql', '')

            if not is_valid:
                return f"SQL校验不通过。原因：{message}\n检测到SQL类型：{sql_type if sql_type else '未知'}"

            return (f"SQL校验通过。\n"
                    f"SQL类型：{sql_type}\n"
                    f"校验后SQL：{sanitized_sql}\n"
                    f"说明：{message}")

        except Exception as e:
            logger.exception(f"验证SQL语句失败: {e}")
            return f"验证SQL语句时发生错误: {str(e)}"

    async def _arun(self, sql: str) -> str:
        return self._run(sql)


class executeSQLModel(BaseModel):
    sql: str = Field(..., description="要执行的SQL语句")


class executeSQL(BaseTool):
    """
    执行SQL语句
    """
    name: str = "executeSQL"
    description: str = "根据输入的SQL语句，执行数据库操作"
    db: MysqlDataBaseManager
    args_schema: type[BaseModel] = executeSQLModel

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _run(self, sql: str) -> str:
        try:
            result: list[dict] = self.db.execute_safe_query(sql)
            return json.dumps(result, ensure_ascii=False)
        except Exception as e:
            logger.exception(f"执行SQL语句失败: {e}")
            return f"执行SQL语句时发生错误: {str(e)}"

    async def _arun(self, sql: str) -> str:
        return self._run(sql)


if __name__ == '__main__':
    from db.config import DATABASE_URL

    db_manager = MysqlDataBaseManager(DATABASE_URL)
    tool = executeSQL(db=db_manager)
    print(tool.invoke({"sql": 'select cashier_id,username from cashiers'}))
