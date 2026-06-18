from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from sqlalchemy.exc import SQLAlchemyError
from db.config import DATABASE_URL
import datetime

# 创建引擎和会话工厂
engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# 定义 ORM 模型
class ChatHistory(Base):
    """对话历史表"""
    __tablename__ = 'chat_history'

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(50), nullable=False, default='default', index=True)
    role = Column(String(20), nullable=False)  # 'user' or 'assistant'
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.now)

    # 关联文件记录
    media_files = relationship("MediaFile", back_populates="chat", cascade="all, delete-orphan")

    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'role': self.role,
            'content': self.content,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


class MediaFile(Base):
    """媒体文件记录表"""
    __tablename__ = 'media_files'

    id = Column(Integer, primary_key=True, autoincrement=True)
    chat_id = Column(Integer, ForeignKey('chat_history.id', ondelete='CASCADE'), nullable=False)
    file_type = Column(String(20), nullable=False)  # 'image', 'audio', 'video'
    file_path = Column(String(500), nullable=False)
    file_name = Column(String(200))
    created_at = Column(DateTime, default=datetime.datetime.now)

    # 反向关联
    chat = relationship("ChatHistory", back_populates="media_files")

    def to_dict(self):
        return {
            'id': self.id,
            'chat_id': self.chat_id,
            'file_type': self.file_type,
            'file_path': self.file_path,
            'file_name': self.file_name,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


def init_database():
    """初始化数据库表结构"""
    try:
        Base.metadata.create_all(bind=engine)
        print("数据库表初始化成功！")
    except SQLAlchemyError as e:
        print(f"数据库初始化失败: {e}")
        raise


def get_session():
    """获取数据库会话"""
    return SessionLocal()


def save_chat_message(user_id, role, content):
    """保存单条聊天消息"""
    session = get_session()
    try:
        chat_msg = ChatHistory(
            user_id=user_id,
            role=role,
            content=content
        )
        session.add(chat_msg)
        session.commit()
        session.refresh(chat_msg)
        return chat_msg.id
    except SQLAlchemyError as e:
        session.rollback()
        print(f"保存消息失败: {e}")
        raise
    finally:
        session.close()


def get_chat_history(user_id='default', limit=50):
    """获取用户的聊天历史"""
    session = get_session()
    try:
        messages = session.query(ChatHistory).filter(
            ChatHistory.user_id == user_id
        ).order_by(
            ChatHistory.created_at.asc()
        ).limit(limit).all()
        return [msg.to_dict() for msg in messages]
    except SQLAlchemyError as e:
        print(f"获取历史失败: {e}")
        raise
    finally:
        session.close()


def clear_chat_history(user_id='default'):
    """清空用户的聊天历史"""
    session = get_session()
    try:
        session.query(ChatHistory).filter(
            ChatHistory.user_id == user_id
        ).delete(synchronize_session=False)
        session.commit()
    except SQLAlchemyError as e:
        session.rollback()
        print(f"清空历史失败: {e}")
        raise
    finally:
        session.close()


def save_media_file(chat_id, file_type, file_path, file_name=None):
    """保存媒体文件记录"""
    session = get_session()
    try:
        media = MediaFile(
            chat_id=chat_id,
            file_type=file_type,
            file_path=file_path,
            file_name=file_name
        )
        session.add(media)
        session.commit()
        session.refresh(media)
        return media.id
    except SQLAlchemyError as e:
        session.rollback()
        print(f"保存文件记录失败: {e}")
        raise
    finally:
        session.close()
