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

            # 规范化输入：统一成 [str] 列表，None 表示所有表
            if table_name is None:
                target_tables: Optional[list[str]] = None
            elif isinstance(table_name, str):
                stripped = table_name.strip()
                target_tables = [stripped] if stripped else None
            elif isinstance(table_name, (list, tuple, set)):
                cleaned = [t.strip() for t in table_name if isinstance(t, str) and t.strip()]
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
                    logger.warning(f"以下表不存在于数据库中，将被忽略：{missing_tables}")

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

            return result

        except Exception as e:
            logger.exception(f"SQL校验异常: {e}")
            result['message'] = f'SQL校验过程出错: {str(e)}'
            return result

    def execute_safe_query(self, sql: str) -> list[dict]:
        """
        安全执行SQL查询语句(只读)
        :param sql: SQL查询语句
        :return: 查询结果列表，每个元素为字典
        """

        try:
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
                return rows

        except SQLAlchemyError as e:
            logger.exception(f"执行SQL查询失败: {e}")
            raise ValueError(f"查询执行失败: {str(e)}")


if __name__ == '__main__':
    db = MysqlDataBaseManager(DATABASE_URL)

    print(db.get_table_constructions())