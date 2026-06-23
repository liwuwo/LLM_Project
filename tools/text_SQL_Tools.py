from langchain.tools import BaseTool
import json

from pydantic import BaseModel, Field

from db.mysql_utils import MysqlDataBaseManager
from utils.logUtils import logger


class queryDBTablesModel(BaseModel):
    """查询数据库中所有的表名和各个表的备注信息，不需要任何参数。直接调用即可返回所有表的列表。"""
    # 保留一个可选字段以防止 BaseTool 在没有字段时出错
    dummy: str | None = Field(default=None, description="占位字段，不需要填写，可以省略")


class queryDBTables(BaseTool):
    """
    文本转SQL工具类
    """
    name: str = "queryDBTables"

    description: str = (
        "当你不知道数据库中有哪些表时调用此工具。"
        "此工具不接受任何输入参数，返回数据库中所有表的列表以及每个表的中文注释说明。"
    )

    db: MysqlDataBaseManager

    args_schema: type[BaseModel] = queryDBTablesModel

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _run(self, dummy: str | None = None):
        try:
            table_list: list[dict] = self.db.get_alltables_comments()
            if not table_list:
                return "数据库中没有任何表。"
            result: str = f"数据库中共有 {len(table_list)} 个表，表名和备注信息如下：\n"
            for i, table in enumerate(table_list):
                table_name = table['table_name']
                table_comment = table['table_comment']
                if not table_comment or table_comment.isspace():
                    table_comment = "（暂无备注）"
                result += f"{i + 1}. {table_name}  —  {table_comment}\n"
            return result
        except Exception as e:
            logger.exception(f"查询数据库表信息失败: {e}")
            return f"查询数据库表信息时发生错误: {str(e)}"

    async def _arun(self, dummy: str | None = None):
        return self._run(dummy=dummy)


class queryDBColumnsModel(BaseModel):
    table_name: str | list[str] | None = Field(
        default=None,
        description=(
            "想要查看表结构的表名，可以是一个表名字符串（如 'cashiers'）或表名列表（如 ['cashiers', 'departments']）。"
            "如果不提供此参数，将返回数据库中所有表的结构。"
        ),
    )


class queryTablesStructure(BaseTool):
    """
    查询数据库表结构
    """
    name: str = "queryTablesStructure"
    description: str = (
        "当你想了解某个或某些表的字段、主键、外键、索引等信息时调用此工具。"
        "输入参数 table_name 可以是表名字符串、表名列表或省略（省略时返回所有表的结构）。"
        "返回每个表的完整结构信息（字段列表、字段类型、是否允许为空、主键、外键、索引等）。"
    )
    db: MysqlDataBaseManager
    args_schema: type[BaseModel] = queryDBColumnsModel

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _run(self, table_name: str | list[str] | None = None) -> str:
        try:
            # 处理各种输入类型：None / 空列表 / 空字符串 -> 查询所有表
            if table_name is None:
                pass  # 直接传给 mysql_utils
            elif isinstance(table_name, str):
                stripped = table_name.strip()
                if not stripped:
                    table_name = None
                else:
                    # 单个字符串 -> 包成列表
                    table_name = [stripped]
            elif isinstance(table_name, list):
                # 过滤空字符串
                cleaned = [t.strip() for t in table_name if isinstance(t, str) and t.strip()]
                if not cleaned:
                    table_name = None
                else:
                    table_name = cleaned
            else:
                # 未知类型 -> 查询所有表
                table_name = None

            table_structure: str = self.db.get_table_constructions(table_name)
            tables_data: dict = json.loads(table_structure) if table_structure else {}

            # 查询指定表但结果为空的情况
            if table_name is not None and not tables_data:
                return f"数据库中不存在以下表：{table_name}，请检查表名拼写是否正确。"
            if not tables_data:
                return "数据库中没有任何表。"

            result_parts: list[str] = []
            for tbl_key, tbl_info in tables_data.items():
                table_name_str = tbl_info.get('table_name', tbl_key)
                comment = tbl_info.get('comment', '') or '（暂无备注）'
                columns: list[dict] = tbl_info.get('columns', [])
                primary_keys: list[str] = tbl_info.get('primary_keys', [])
                foreign_keys: list[dict] = tbl_info.get('foreign_keys', [])
                indexes: list[dict] = tbl_info.get('indexes', [])

                part = f"【表】{table_name_str}  —  备注：{comment}\n"

                part += "  字段列表：\n"
                for col in columns:
                    col_name = col.get('name', '')
                    col_type = col.get('type', '')
                    col_nullable = '可为空' if col.get('nullable') else '不可为空'
                    col_default = col.get('default') if col.get('default') else '无'
                    col_comment = col.get('comment', '') or '无'
                    part += (
                        f"    - {col_name} ({col_type}) {col_nullable}, "
                        f"默认值: {col_default}, 说明: {col_comment}\n"
                    )

                part += "  主键: "
                part += ", ".join(primary_keys) if primary_keys else "无"
                part += "\n"

                if foreign_keys:
                    part += "  外键:\n"
                    for fk in foreign_keys:
                        fk_name = fk.get('name', '')
                        constrained = ', '.join(fk.get('constrained_columns', []))
                        referred_table = fk.get('referred_table', '')
                        referred_cols = ', '.join(fk.get('referred_columns', []))
                        part += f"    - {fk_name}: {constrained} -> {referred_table}({referred_cols})\n"
                else:
                    part += "  外键: 无\n"

                if indexes:
                    part += "  索引:\n"
                    for idx in indexes:
                        idx_name = idx.get('name', '')
                        idx_cols = ', '.join(idx.get('column_names', []))
                        idx_unique = '唯一索引' if idx.get('unique') else '普通索引'
                        part += f"    - {idx_name} ({idx_cols}) — {idx_unique}\n"
                else:
                    part += "  索引: 无\n"

                result_parts.append(part)

            return (
                f"共查询到 {len(result_parts)} 张表的结构信息：\n\n"
                + '\n'.join(result_parts)
            )

        except Exception as e:
            logger.exception(f"查询数据库表结构失败: {e}")
            return f"查询数据库表结构时发生错误: {str(e)}"

    async def _arun(self, table_name: str | list[str] | None = None) -> str:
        return self._run(table_name=table_name)


class validateSQlModel(BaseModel):
    sql: str = Field(
        ...,
        description=(
            "要进行验证的 SQL 语句（字符串）。"
            "校验该 SQL 是否仅包含只读查询操作（SELECT），并做基本语法检查。"
        ),
    )


class validateSQl(BaseTool):
    """
    验证 SQL 语句（只读检查 + 基本语法检查）。
    只做安全性校验，不执行实际查询。
    """
    name: str = "validateSQl"
    description: str = (
        "当你生成了一个 SQL 语句并想确认它安全合法时调用此工具。"
        "输入要检查的 SQL 字符串。此工具只会校验 SQL 是否仅包含只读 SELECT 查询。"
        "仅在你对生成的 SQL 没有信心时调用。"
        "如果 SQL 已经确定没问题，直接用 executeSQL 执行即可，不需要调用本工具。"
    )
    db: MysqlDataBaseManager
    args_schema: type[BaseModel] = validateSQlModel

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _run(self, sql: str) -> str:
        try:
            if not sql or not sql.strip():
                return "SQL 校验失败：SQL 语句为空。"
            result: dict = self.db.validate_sql(sql)
            is_valid = result.get('valid', False)
            message = result.get('message', '')
            sql_type = result.get('sql_type', '')
            sanitized_sql = result.get('sanitized_sql', '')

            if not is_valid:
                return (
                    f"【SQL 校验不通过】\n"
                    f"原因：{message}\n"
                    f"检测到的 SQL 类型：{sql_type if sql_type else '未知'}\n"
                    f"原始 SQL：{sql}\n"
                    f"—— 请重新编写 SQL，不要用 INSERT/UPDATE/DELETE/DROP 等写操作。"
                )

            return (
                f"【SQL 校验通过】\n"
                f"SQL 类型：{sql_type}\n"
                f"校验后 SQL：{sanitized_sql}\n"
                f"说明：{message}"
            )

        except Exception as e:
            logger.exception(f"验证 SQL 语句失败: {e}")
            return f"验证 SQL 语句时发生错误: {str(e)}"

    async def _arun(self, sql: str) -> str:
        return self._run(sql=sql)


class executeSQLModel(BaseModel):
    sql: str = Field(
        ...,
        description="要执行的 SQL SELECT 查询语句。执行后即可基于返回数据给出最终答案。",
    )


class executeSQL(BaseTool):
    """
    执行 SQL 查询语句（仅允许 SELECT 只读查询）。
    """
    name: str = "executeSQL"
    description: str = (
        "使用此工具对数据库执行 SELECT 查询，直接返回真实的数据库数据。"
        "输入参数为 SQL 字符串。仅允许 SELECT 查询。"
        "调用后请基于返回的数据立即总结出答案给用户（Final Answer），不要再调用其他工具。"
    )
    db: MysqlDataBaseManager
    args_schema: type[BaseModel] = executeSQLModel

    MAX_ROWS: int = 50

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _run(self, sql: str) -> str:
        try:
            if not sql or not sql.strip():
                return "[执行失败] SQL 语句为空。"

            result: list[dict] = self.db.execute_safe_query(sql)
            total = len(result)

            if total == 0:
                return f"[执行结果] 无匹配数据（0 行）。\nSQL: {sql}"

            truncated = False
            display = result
            if total > self.MAX_ROWS:
                display = result[: self.MAX_ROWS]
                truncated = True

            data_lines = []
            for row in display:
                row_str = ", ".join(f"{k}={v}" for k, v in row.items())
                data_lines.append(f"  {row_str}")

            header = f"[执行结果] 共 {total} 行。"
            if truncated:
                header += f"（仅显示前 {self.MAX_ROWS} 行）"
            header += f"  SQL: {sql}\n数据行：\n"

            return header + "\n".join(data_lines)

        except Exception as e:
            logger.exception(f"执行 SQL 失败: {e}")
            return f"[执行失败]\nSQL: {sql}\n错误: {str(e)}"

    async def _arun(self, sql: str) -> str:
        return self._run(sql=sql)


if __name__ == '__main__':
    from db.config import DATABASE_URL

    db_manager = MysqlDataBaseManager(DATABASE_URL)
    tool = queryTablesStructure(db=db_manager)
    print(tool.invoke({'table_name':['order_items', 'orders']}))