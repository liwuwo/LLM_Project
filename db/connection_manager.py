import threading
from typing import Optional, Tuple

from langgraph.checkpoint.mysql.pymysql import PyMySQLSaver
from langgraph.store.mysql import PyMySQLStore

from db.config import DATABASE_WEATHER_URL
from utils.logUtils import logger


class WeatherDBConnectionManager:
    """
    天气数据库连接管理器（单例模式）
    统一管理 checkpoint (PyMySQLSaver) 和 store (PyMySQLStore) 连接，
    避免在 WeatherAgent、WeatherMiddleware 中反复创建连接。
    """

    _instance: Optional["WeatherDBConnectionManager"] = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True

        self._checkpoint_ctx: Optional[PyMySQLSaver] = None
        self.checkpoint: Optional[PyMySQLSaver] = None

        self._store_ctx: Optional[PyMySQLStore] = None
        self.store: Optional[PyMySQLStore] = None

        self._setup_lock = threading.Lock()

    def _create_checkpoint(self) -> Optional[PyMySQLSaver]:
        """创建并保持 checkpoint 连接（懒加载+线程安全）"""
        if self.checkpoint is not None:
            return self.checkpoint
        with self._setup_lock:
            if self.checkpoint is not None:
                return self.checkpoint
            try:
                self._checkpoint_ctx = PyMySQLSaver.from_conn_string(DATABASE_WEATHER_URL)
                self.checkpoint = self._checkpoint_ctx.__enter__()
                self.checkpoint.setup()
                logger.info("[单例] 成功连接到天气数据库 checkpoint")
            except Exception as e:
                logger.warning(f"[单例] 无法连接到天气数据库 checkpoint: {e}，将不使用持久化")
                self._checkpoint_ctx = None
                self.checkpoint = None
        return self.checkpoint

    def _create_store(self) -> Optional[PyMySQLStore]:
        """创建并保持 store 连接（懒加载+线程安全）"""
        if self.store is not None:
            return self.store
        with self._setup_lock:
            if self.store is not None:
                return self.store
            try:
                self._store_ctx = PyMySQLStore.from_conn_string(DATABASE_WEATHER_URL)
                self.store = self._store_ctx.__enter__()
                self.store.setup()
                logger.info("[单例] 创建天气 store 成功")
            except Exception as e:
                logger.error(f"[单例] 创建天气 store 失败: {e}")
                self._store_ctx = None
                self.store = None
        return self.store

    def get_checkpoint(self) -> Optional[PyMySQLSaver]:
        """获取 checkpoint 连接，未创建则自动创建"""
        return self._create_checkpoint()

    def get_store(self) -> Optional[PyMySQLStore]:
        """获取 store 连接，未创建则自动创建"""
        return self._create_store()

    def get_connections(self) -> Tuple[Optional[PyMySQLSaver], Optional[PyMySQLStore]]:
        """同时获取 checkpoint 和 store 连接，未创建则自动创建"""
        return self.get_checkpoint(), self.get_store()

    def close_all(self):
        """关闭所有连接（应用退出时调用）"""
        with self._setup_lock:
            if self._store_ctx is not None:
                try:
                    self._store_ctx.__exit__(None, None, None)
                    logger.info("[单例] 已关闭天气 store 连接")
                except Exception as e:
                    logger.warning(f"[单例] 关闭 store 连接时出错: {e}")
                finally:
                    self._store_ctx = None
                    self.store = None

            if self._checkpoint_ctx is not None:
                try:
                    self._checkpoint_ctx.__exit__(None, None, None)
                    logger.info("[单例] 已关闭天气数据库 checkpoint 连接")
                except Exception as e:
                    logger.warning(f"[单例] 关闭 checkpoint 连接时出错: {e}")
                finally:
                    self._checkpoint_ctx = None
                    self.checkpoint = None

    @classmethod
    def reset_instance(cls):
        """重置单例实例（主要用于测试，生产环境一般不需要）"""
        with cls._lock:
            if cls._instance is not None:
                cls._instance.close_all()
                cls._instance = None


def get_weather_db_manager() -> WeatherDBConnectionManager:
    """获取全局单例的天气数据库连接管理器"""
    return WeatherDBConnectionManager()


import atexit

atexit.register(lambda: get_weather_db_manager().close_all())