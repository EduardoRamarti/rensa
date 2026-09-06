import os

from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from sqlalchemy import create_engine, text

from src.langchain_section.config.settings import settings
from src.langchain_section.memory.base import BaseMemoryBackend


class SQLiteMemoryBackend(BaseMemoryBackend):
    """Backend de memoria persistente usando SQLite"""

    def __init__(self, db_path: str = None):
        self.db_path = db_path or settings.SQLITE_DB_PATH
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._connection_string = f"sqlite:///{self.db_path}"

    def get_history(self, session_id: str) -> BaseChatMessageHistory:
        """Retorna el historial de mensajes para una sesion"""
        return SQLChatMessageHistory(
            session_id=session_id,
            connection=self._connection_string
        )

    def clear_history(self, session_id: str) -> None:
        """Elimina el historial de una sesion"""
        history = self.get_history(session_id)
        history.clear()

    def list_sessions(self) -> list(str):
        """Lista todos los session ID disponibles"""
        engine = create_engine(self._connection_string)
        try: 
            with engine.connect() as conn:
                result = conn.execute(
                    text("SELECT DISTINCT session_id FROM message_store")
                )
                return [row[0] for row in result]
        except ImportError:
            return []
