from langchain.tools import BaseTool
import json

from pydantic import BaseModel, Field

from db.mysql_utils import MysqlDataBaseManager
from utils.logUtils import logger


def normalize_sql_input(sql):
    """
    统一处理 LLM 可能输出的"嵌套 JSON"形式参数。
    典型场景：LLM 已经按 JSON 写好 {"sql": "SELECT ... FROM ..."}，
    但由于解析器/转义原因，到达工具时 sql 参数变成整段 JSON 字符串。

    返回：清洗后的 SQL 字符串（不会为 None，空值返回 ""）。
    """
    if sql is None:
        return ""
    # 快速路径：看起来已经是干净 SQL，直接返回
    if isinstance(sql, str):
        stripped = sql.strip()
        if stripped and stripped.upper().startswith(
            ('SELECT', 'WITH', 'SHOW', 'DESCRIBE', 'EXPLAIN', 'DESC')
        ):
            return stripped
    try:
        return MysqlDataBaseManager._extract_sql_from_nested_input(sql)
    except Exception:
        return str(sql).strip()


def normalize_table_name_input(table_name):
    """
    从 LLM 可能输出的"嵌套 JSON"形式参数中提取真正的 table_name。
    典型场景：
        - 直接被当成字符串的完整 JSON： '{"table_name": "order_items"}'
        - JSON 中 table_name 字段是列表： '{"table_name": ["orders", "products"]}'
        - 或 LLM 把整段 JSON 直接塞进 list 里： ['{"table_name": "order_items"}']

    返回取值优先级（第一个非空即返回）：
        1) dict["table_name"]  /  json_str -> dict -> dict["table_name"]
        2) 仅一项的 dict 的 value（比如 LLM 用了别的字段名）
        3) 普通字符串 / 普通列表原样返回

    返回类型：None（查所有表） / str（单表） / list[str]（多表）
    """
    if table_name is None:
        return None

    # 规范化：把 list 中仅一项且是 JSON 的情况展开
    raw = table_name
    if isinstance(raw, list) and len(raw) == 1:
        raw = raw[0]

    # 若是 dict，优先取里面的 table_name
    if isinstance(raw, dict):
        val = raw.get("table_name")
        if val is None:
            # 兼容其他可能被 LLM 使用的字段
            for key in ("table_names", "tables", "name", "names"):
                if key in raw:
                    val = raw[key]
                    break
        if isinstance(val, (str, list)) and val:
            return val
        if val is None and raw:
            # 还是拿不到则回退：如果是单 value 的 dict，取那个 value
            only_val = next(iter(raw.values()))
            if isinstance(only_val, (str, list)) and only_val:
                return only_val
        return None

    if isinstance(raw, str):
        stripped = raw.strip()
        if not stripped:
            return None
        # 看起来是 JSON：尝试解析
        if stripped.startswith("{") and stripped.endswith("}"):
            try:
                parsed = json.loads(stripped)
                if isinstance(parsed, dict):
                    return normalize_table_name_input(parsed)
            except (json.JSONDecodeError, ValueError):
                pass
        # 看起来是 "[a, b, c]" 形式的列表字符串（langchain 偶尔会把列表字符串化）
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                parsed = json.loads(stripped)
                if isinstance(parsed, list):
                    return parsed
            except (json.JSONDecodeError, ValueError):
                pass
        # 其他：普通表名字符串，原样返回
        return stripped

    # 已经是正常的 list[str]，原样返回
    if isinstance(raw, list):
        return raw

    # 其他未知类型：转字符串兜底
    return str(raw).strip()


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
            # 先统一处理 LLM 嵌套 JSON 的情况：例如 table_name 是字符串 '{"table_name": "order_items"}'
            table_name = normalize_table_name_input(table_name)

            # 之后按正常输入类型做解析：None/空字符串/空列表 -> 查询所有表
            if table_name is None:
                target_tables = None
            elif isinstance(table_name, str):
                stripped = table_name.strip()
                if not stripped:
                    target_tables = None
                else:
                    # 再次做一次"兜底解析"，防止上面没有命中的 JSON 格式
                    if stripped.startswith("{") and stripped.endswith("}"):
                        try:
                            parsed = json.loads(stripped)
                            if isinstance(parsed, dict):
                                v = parsed.get("table_name")
                                if isinstance(v, str):
                                    target_tables = [v]
                                elif isinstance(v, list):
                                    target_tables = [str(x).strip() for x in v if str(x).strip()]
                                else:
                                    target_tables = None
                            else:
                                target_tables = [stripped]
                        except (json.JSONDecodeError, ValueError):
                            target_tables = [stripped]
                    else:
                        target_tables = [stripped]
            elif isinstance(table_name, list):
                cleaned = []
                for item in table_name:
                    if isinstance(item, str):
                        s = item.strip()
                        if not s:
                            continue
                        # 列表元素也可能被嵌套成 "{'table_name': 'order_items'}"，再解析一层
                        if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
                            try:
                                parsed = json.loads(s)
                                if isinstance(parsed, dict):
                                    v = parsed.get("table_name")
                                    if isinstance(v, str) and v.strip():
                                        cleaned.append(v.strip())
                                    elif isinstance(v, list):
                                        cleaned.extend([str(x).strip() for x in v if str(x).strip()])
                                elif isinstance(parsed, list):
                                    cleaned.extend([str(x).strip() for x in parsed if str(x).strip()])
                                else:
                                    cleaned.append(s)
                            except (json.JSONDecodeError, ValueError):
                                cleaned.append(s)
                        else:
                            cleaned.append(s)
                    else:
                        s = str(item).strip()
                        if s:
                            cleaned.append(s)
                target_tables = cleaned if cleaned else None
            else:
                target_tables = None

            table_structure: str = self.db.get_table_constructions(target_tables)
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
            # 同样处理 LLM 可能产生的"嵌套 JSON"参数
            sql = normalize_sql_input(sql)
            if not sql:
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
            # 解析 LLM 可能产生的"嵌套 JSON"参数，提取真正的 SQL
            sql = normalize_sql_input(sql)
            if not sql:
                return "[执行失败] SQL 语句为空。请提供有效的 SELECT 语句。"

            result: list[dict] = self.db.execute_safe_query(sql)
            total = len(result)

            if total == 0:
                return (
                    f"[执行结果] SQL 查询已成功执行，但数据库中没有匹配的数据（0 行）。\n"
                    f"执行的 SQL: {sql}\n"
                    f"———提示：请尝试修改查询条件（例如放宽 WHERE 条件，或检查表名/字段名是否正确）。"
                )

            truncated = False
            display = result
            if total > self.MAX_ROWS:
                display = result[: self.MAX_ROWS]
                truncated = True

            # 格式化输出为表格样式，便于 LLM 解析数据
            data_lines = []
            for row_idx, row in enumerate(display, 1):
                # 使用 key=value 格式，但每行标上行号，便于读取
                row_items = []
                for k, v in row.items():
                    if v is None:
                        row_items.append(f"{k}=NULL")
                    else:
                        row_items.append(f"{k}={v}")
                data_lines.append(f"  行{row_idx}: " + "; ".join(row_items))

            header = f"[执行结果成功] 共查询到 {total} 行数据。"
            if truncated:
                header += f"（由于数据量较大，仅显示前 {self.MAX_ROWS} 行，请基于这些数据分析）"
            header += f"\n执行的 SQL: {sql}\n数据详情：\n"

            # 在末尾附加明确提示：这是最终数据，可以给 Final Answer 了
            footer = (
                "\n—以上为本次查询返回的数据。请你现在基于上述数据，用中文直接给出 Final Answer。"
                "\n不要再调用任何工具。"
            )

            return header + "\n".join(data_lines) + footer

        except Exception as e:
            logger.exception(f"执行 SQL 失败: {e}")
            return (
                f"[执行失败] SQL 语句执行出错。\n"
                f"SQL: {sql}\n"
                f"错误信息: {str(e)}\n"
                f"———提示：请重新检查 SQL 语法、表名/字段名是否正确，不要再次使用完全相同的 SQL。"
            )

    async def _arun(self, sql: str) -> str:
        return self._run(sql=sql)


if __name__ == '__main__':
    from db.config import DATABASE_URL

    db_manager = MysqlDataBaseManager(DATABASE_URL)
    tool = queryTablesStructure(db=db_manager)
    print(tool.invoke({'table_name':['order_items', 'orders']}))