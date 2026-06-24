from typing import Optional
import json
import re

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import SQLAlchemyError
from db.config import DATABASE_URL

from utils.logUtils import logger


class MysqlDataBaseManager:
    """
    Mysql数据库管理类
    """

    def __init__(self, dbconnection_string: str):
        """
        初始化数据库连接
        :param dbconnection_string: 数据库连接字符串
        """
        self.engine = create_engine(dbconnection_string, echo=False, pool_size=5, pool_recycle=3600)
        # 记录本次请求中执行过的 SQL，供代理在最后统一打印总结
        self._sql_execution_history: list[dict] = []

    def append_sql_history(self, record: dict):
        self._sql_execution_history.append(record)

    def get_sql_history(self) -> list[dict]:
        return list(self._sql_execution_history)

    def get_and_clear_sql_history(self) -> list[dict]:
        history = list(self._sql_execution_history)
        self._sql_execution_history = []
        return history

    def get_alltables_names(self) -> list[str]:
        try:
            inspector = inspect(self.engine)
            return inspector.get_table_names()
        except SQLAlchemyError as e:
            logger.exception(f"Error: {e}")
            raise ValueError(f"Failed to connect to the database:{str(e)}")

    def get_alltables_comments(self) -> list[dict]:
        """
        获取所有表的注释
        :return: 列表，每个元素为字典，包含表名和表注释
        """
        try:
            inspector = inspect(self.engine)
            tables_info = []
            for table_name in inspector.get_table_names():
                # 获取表的元数据信息
                table_comment_obj = inspector.get_table_comment(table_name)
                table_comment = table_comment_obj.get('text', '') if table_comment_obj else ''
                tables_info.append({
                    'table_name': table_name,
                    'table_comment': table_comment
                })
            return tables_info
        except SQLAlchemyError as e:
            logger.exception(f"Error: {e}")
            raise ValueError(f"Failed to get table comments: {str(e)}")

    def get_table_constructions(self, table_name: Optional[object] = None) -> str:
        """
        获取 table_name 中指定的表的结构信息。
        :param table_name: 支持以下输入：
            - None 或省略  -> 返回数据库中所有表的结构
            - 单个字符串   -> 视为单个表名
            - 字符串列表/元组/集合 -> 视为多个表名
        :return: JSON 格式的字符串，包含表结构信息
        """
        try:
            inspector = inspect(self.engine)
            all_db_tables = set(inspector.get_table_names())

            # 【关键修复】处理 LLM 嵌套 JSON 输入：如果 table_name 看起来是
            #   '{"table_name": "order_items"}' 这种整段 JSON 字符串，
            #   或 list 里只有一项且是 JSON，都需要先解析出真正的 table_name 字段。
            def _unwrap(raw):
                if raw is None:
                    return None
                if isinstance(raw, list) and len(raw) == 1:
                    return _unwrap(raw[0])
                if isinstance(raw, dict):
                    for key in ("table_name", "table_names", "tables", "name", "names"):
                        if key in raw and isinstance(raw[key], (str, list)) and raw[key]:
                            return raw[key]
                    # 回退：单 value 的 dict
                    if len(raw) == 1:
                        v = next(iter(raw.values()))
                        if isinstance(v, (str, list)) and v:
                            return v
                    return None
                if isinstance(raw, str):
                    s = raw.strip()
                    if not s:
                        return None
                    if s.startswith("{") and s.endswith("}"):
                        try:
                            parsed = json.loads(s)
                            if isinstance(parsed, dict):
                                return _unwrap(parsed)
                        except (json.JSONDecodeError, ValueError):
                            pass
                    if s.startswith("[") and s.endswith("]"):
                        try:
                            parsed = json.loads(s)
                            if isinstance(parsed, list):
                                return parsed
                        except (json.JSONDecodeError, ValueError):
                            pass
                    return s
                return raw

            table_name = _unwrap(table_name)

            # 规范化输入：统一成 [str] 列表，None 表示所有表
            if table_name is None:
                target_tables: Optional[list[str]] = None
            elif isinstance(table_name, str):
                stripped = table_name.strip()
                # 再次兜底：万一仍是一段 JSON（比如解析被 try/except 跳过了）
                if stripped.startswith("{") and stripped.endswith("}"):
                    try:
                        parsed = json.loads(stripped)
                        if isinstance(parsed, dict):
                            v = parsed.get("table_name")
                            if isinstance(v, str) and v.strip():
                                target_tables = [v.strip()]
                            elif isinstance(v, list):
                                target_tables = [str(x).strip() for x in v if str(x).strip()]
                            else:
                                target_tables = None
                        else:
                            target_tables = [stripped]
                    except (json.JSONDecodeError, ValueError):
                        target_tables = [stripped]
                else:
                    target_tables = [stripped] if stripped else None
            elif isinstance(table_name, (list, tuple, set)):
                cleaned = []
                for t in table_name:
                    if isinstance(t, str):
                        s = t.strip()
                        if not s:
                            continue
                        # 列表元素也可能是嵌套 JSON，再解一层
                        if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
                            try:
                                parsed = json.loads(s)
                                if isinstance(parsed, dict):
                                    v = parsed.get("table_name")
                                    if isinstance(v, str) and v.strip():
                                        cleaned.append(v.strip())
                                    elif isinstance(v, list):
                                        cleaned.extend([str(x).strip() for x in v if str(x).strip()])
                                    else:
                                        cleaned.append(s)
                                elif isinstance(parsed, list):
                                    cleaned.extend([str(x).strip() for x in parsed if str(x).strip()])
                                else:
                                    cleaned.append(s)
                            except (json.JSONDecodeError, ValueError):
                                cleaned.append(s)
                        else:
                            cleaned.append(s)
                    else:
                        s = str(t).strip()
                        if s:
                            cleaned.append(s)
                target_tables = cleaned if cleaned else None
            else:
                logger.warning(f"get_table_constructions: 未知的 table_name 类型 {type(table_name)!r}，将返回所有表")
                target_tables = None

            # 确定要查询的表
            existing_tables: list[str] = []
            missing_tables: list[str] = []
            if target_tables is None:
                existing_tables = list(all_db_tables)
            else:
                for t in target_tables:
                    if t in all_db_tables:
                        existing_tables.append(t)
                    else:
                        missing_tables.append(t)
                if missing_tables:
                    logger.warning(f"以下表将被忽略：{missing_tables}")

            tables_info: dict[str, dict] = {}
            for tbl_name in existing_tables:
                columns_info = []
                for column in inspector.get_columns(tbl_name):
                    columns_info.append({
                        'name': column['name'],
                        'type': str(column['type']),
                        'nullable': column.get('nullable', True),
                        'default': str(column.get('default')) if column.get('default') is not None else None,
                        'comment': column.get('comment', '') or '',
                    })

                pk_constraint = inspector.get_pk_constraint(tbl_name)
                primary_keys = (
                    pk_constraint.get('constrained_columns', [])
                    if pk_constraint and isinstance(pk_constraint, dict)
                    else []
                )

                foreign_keys = []
                for fk in inspector.get_foreign_keys(tbl_name):
                    if not isinstance(fk, dict):
                        continue
                    foreign_keys.append({
                        'name': fk.get('name'),
                        'constrained_columns': fk.get('constrained_columns', []),
                        'referred_table': fk.get('referred_table'),
                        'referred_columns': fk.get('referred_columns', []),
                    })

                indexes = []
                for idx in inspector.get_indexes(tbl_name):
                    if not isinstance(idx, dict):
                        continue
                    indexes.append({
                        'name': idx.get('name'),
                        'column_names': idx.get('column_names', []),
                        'unique': idx.get('unique', False),
                    })

                table_comment_obj = inspector.get_table_comment(tbl_name)
                table_comment = ''
                if isinstance(table_comment_obj, dict):
                    table_comment = table_comment_obj.get('text', '') or ''
                else:
                    table_comment = getattr(table_comment_obj, 'text', '') or ''

                tables_info[tbl_name] = {
                    'table_name': tbl_name,
                    'comment': table_comment,
                    'columns': columns_info,
                    'primary_keys': primary_keys,
                    'foreign_keys': foreign_keys,
                    'indexes': indexes,
                }

            return json.dumps(tables_info, ensure_ascii=False, indent=2)

        except SQLAlchemyError as e:
            logger.exception(f"Error getting table constructions: {e}")
            return json.dumps({}, ensure_ascii=False)

    def validate_sql(self, sql: str) -> dict:
        """
        校验SQL语句是否符合MySQL方言规则，并确保只允许查询操作
        :param sql: SQL语句字符串
        :return: 字典，包含校验结果和详细信息
        """
        result = {
            'valid': False,
            'message': '',
            'sql_type': None,
            'sanitized_sql': None
        }

        try:
            # 1. 基本检查
            if not sql or not sql.strip():
                result['message'] = 'SQL语句不能为空'
                return result

            sql_stripped = sql.strip().rstrip(';')

            # 2. 检测SQL类型(转换为小写进行比较)
            sql_upper = sql_stripped.upper()

            # 定义危险的SQL关键字
            dangerous_keywords = [
                'INSERT', 'UPDATE', 'DELETE', 'DROP', 'ALTER', 'CREATE',
                'TRUNCATE', 'REPLACE', 'MERGE', 'GRANT', 'REVOKE',
                'EXEC', 'EXECUTE', 'CALL', 'SET', 'USE'
            ]

            # 检查是否包含危险关键字
            for keyword in dangerous_keywords:
                # 使用正则表达式匹配完整的单词，避免误判
                pattern = r'\b' + keyword + r'\b'
                if re.search(pattern, sql_upper):
                    result['message'] = f'不允许执行{keyword}操作，只允许SELECT查询'
                    result['sql_type'] = keyword
                    return result

            # 3. 检查是否以SELECT开头
            if not sql_upper.startswith('SELECT'):
                result['message'] = '只允许执行SELECT查询语句'
                return result

            result['sql_type'] = 'SELECT'

            # 4. 检查是否有多个语句(防止SQL注入)
            if ';' in sql_stripped and len(sql_stripped.split(';')) > 1:
                # 检查分号后是否还有其他SQL语句
                remaining = sql_stripped.split(';', 1)[1].strip()
                if remaining and not remaining.startswith('--'):
                    result['message'] = '不允许执行多条SQL语句'
                    return result

            # 5. 检查注释符号的安全性
            # 允许多行注释 /* */ 和单行注释 -- 或 #
            # 但要防止注释绕过检查
            if '--' in sql_stripped:
                # 检查注释后是否有危险内容
                parts = sql_stripped.split('--', 1)
                if len(parts) > 1:
                    comment_part = parts[1].upper()
                    for keyword in dangerous_keywords:
                        if keyword in comment_part:
                            result['message'] = f'注释中包含不允许的关键字: {keyword}'
                            return result

            # 6. 使用SQLAlchemy的text()进行基本的语法验证
            try:
                text_obj = text(sql_stripped)
                result['sanitized_sql'] = str(text_obj)
            except Exception as e:
                result['message'] = f'SQL语法错误: {str(e)}'
                return result

            # 7. 所有检查通过
            result['valid'] = True
            result['message'] = 'SQL语句校验通过'
            result['sanitized_sql'] = sql_stripped

            # 同时写入 SQL 执行历史，便于最终汇总
            self.append_sql_history({
                "raw_input": sql,
                "sql": sql_stripped,
                "success": True,
                "rows": None,
                "error": None,
                "kind": "validate",
            })
            return result

        except Exception as e:
            logger.exception(f"SQL校验异常: {e}")
            result['message'] = f'SQL校验过程出错: {str(e)}'
            self.append_sql_history({
                "raw_input": sql,
                "sql": "",
                "success": False,
                "rows": None,
                "error": str(e),
                "kind": "validate",
            })
            return result

    @staticmethod
    def _extract_sql_from_nested_input(raw_input) -> str:
        """
        从 LLM 可能返回的"嵌套 JSON 输入"中提取真正的 SQL 语句。

        典型场景：LLM 按 ReAct 格式输出 Action Input 时，可能写成：
            Action Input: {"sql": "SELECT MAX(price) FROM products"}
        但由于解析器/转义原因，真正到达工具的 `sql` 参数可能变成：
            '{"sql": "SELECT MAX(price) FROM products"}'  （整段 JSON 字符串）
        甚至是 dict 对象。

        此方法会：
          1) 优先把 dict / 可解析 JSON 字符串中的 `sql` 字段提取出来；
          2) 若仍不是以 SELECT 开头，再尝试去掉外层引号、分号等；
          3) 都失败时原样返回（让后续的语法检查/执行报错给出提示）。
        """
        if raw_input is None:
            return ""

        # 情况 A：已经是 dict（LangChain 某些版本会直接解析成 dict）
        if isinstance(raw_input, dict):
            if 'sql' in raw_input and isinstance(raw_input['sql'], str):
                return raw_input['sql'].strip()
            # 若 dict 只有一个 value 且是字符串，退而求其次用它
            values = [v for v in raw_input.values() if isinstance(v, str)]
            if len(values) == 1:
                return values[0].strip()
            # 否则尝试 JSON 化后再让下面的字符串流程处理
            try:
                raw_input = json.dumps(raw_input, ensure_ascii=False)
            except Exception:
                return str(raw_input)

        if not isinstance(raw_input, str):
            raw_input = str(raw_input)

        sql = raw_input.strip()
        if not sql:
            return ""

        # 情况 B：字符串本身是一个 JSON：形如 "{"sql": "SELECT ..."}"
        # 先用最宽松的方式尝试 json.loads
        try:
            parsed = json.loads(sql)
            if isinstance(parsed, dict):
                candidate = parsed.get('sql')
                if isinstance(candidate, str) and candidate.strip():
                    sql = candidate.strip()
                elif len(parsed) == 1:
                    # 某些工具用的字段名可能是别的（例如 query），兜底取唯一的 string value
                    only_val = next(iter(parsed.values()))
                    if isinstance(only_val, str) and only_val.strip():
                        sql = only_val.strip()
            elif isinstance(parsed, str) and parsed.strip():
                sql = parsed.strip()
        except (json.JSONDecodeError, ValueError):
            pass

        # 情况 C：前后还有多余的引号/反引号，如 '"SELECT ..."'、"`SELECT ...`"
        # 反复剥除，直到稳定或不再是以引号包裹
        for _ in range(3):
            changed = False
            if len(sql) >= 2 and sql[0] == sql[-1] and sql[0] in ('"', "'", '`'):
                sql = sql[1:-1].strip()
                changed = True
            if not changed:
                break

        # 情况 D：字符串里仍包含 "SELECT"，但开头是 "{"sql": ..." 之类的残留
        # 做一次轻量正则兜底：抓取第一个以 SELECT 开头、以 ;/结尾或到末尾的片段
        if not sql.upper().startswith(('SELECT', 'WITH', 'SHOW', 'DESCRIBE', 'EXPLAIN')):
            m = re.search(
                r'(?is)\b(SELECT|WITH|SHOW|DESCRIBE|DESC|EXPLAIN)\b.+(?=;|"|\'|`|\}\s*$|$)',
                sql,
            )
            if m:
                candidate = m.group(0).strip().rstrip(';').strip()
                if candidate:
                    sql = candidate

        return sql.strip().rstrip(';').strip()

    def execute_safe_query(self, sql: str) -> list[dict]:
        """
        安全执行SQL查询语句(只读)
        :param sql: SQL查询语句
        :return: 查询结果列表，每个元素为字典
        """

        try:
            # 1. 处理 LLM 可能的嵌套 JSON 输入，提取真正的 SQL
            raw_sql_input = sql
            sql = self._extract_sql_from_nested_input(sql)
            if not sql:
                record = {
                    "raw_input": raw_sql_input,
                    "sql": "",
                    "success": False,
                    "rows": 0,
                    "error": "SQL 语句为空（解析后未得到有效 SQL）",
                }
                self.append_sql_history(record)
                raise ValueError("SQL 语句为空（解析后未得到有效 SQL）")

            # 2. 创建连接并执行查询
            with self.engine.connect() as connection:
                # 使用参数化查询防止SQL注入
                result = connection.execute(text(sql))

                # 3. 获取列名
                columns = result.keys()

                # 4. 将结果转换为字典列表
                rows = []
                for row in result:
                    row_dict = dict(zip(columns, row))
                    # 处理特殊类型(如datetime)
                    for key, value in row_dict.items():
                        if hasattr(value, 'isoformat'):  # datetime对象
                            row_dict[key] = value.isoformat()
                        elif isinstance(value, bytes):  # bytes对象
                            row_dict[key] = value.decode('utf-8', errors='ignore')
                    rows.append(row_dict)

                logger.info(f"成功执行查询，返回 {len(rows)} 条记录")
                self.append_sql_history({
                    "raw_input": raw_sql_input,
                    "sql": sql,
                    "success": True,
                    "rows": len(rows),
                    "error": None,
                })
                return rows

        except SQLAlchemyError as e:
            logger.exception(f"执行SQL查询失败: {e}")
            self.append_sql_history({
                "raw_input": raw_sql_input if 'raw_sql_input' in locals() else sql,
                "sql": sql if 'sql' in locals() else "",
                "success": False,
                "rows": 0,
                "error": str(e),
            })
            raise ValueError(f"查询执行失败: {str(e)}")


if __name__ == '__main__':
    db = MysqlDataBaseManager(DATABASE_URL)

    print(db.get_table_constructions())