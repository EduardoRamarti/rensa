"""Backend de memoria usando PostgreSQL"""
import os

from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from sqlalchemy import create_engine, text

from src.langchain_section.memory.base import BaseMemoryBackend


class PostgreSQLMemoryBackend(BaseMemoryBackend):
    """Backend de memoria usando PostgreSQL"""
    def __init__(self, database_url: str = None): #Constructor → recibe la url de la bd
        self._database_url = database_url or os.getenv("DATABASE_URL") # si no recibe url pasa el dato default

        if not self._database_url: # Si no se definio como variable de entorno o como argumento
            raise ValueError(
                "DATABASE_URL no esta configurada"
                "Agrega una url a tu .env"
                "postgresql://user:password@host:port/db"
            )
        if not self._database_url.startswith("postgresql"): #en caso que la base sea diferente a postgres
            raise ValueError(
                "DATABASE_URL debe empezar con 'postgresql://'"
            )

    # Obtener el historial de la DB
    def get_history(self, session_id: str) -> BaseChatMessageHistory:
        """Retorna el historial de mensajes para una sesion"""
        # SQLChatMessageHistory realiza: 
        # 1. Conexion inicial 
        # 2. Generalmente, hace una comprobación rápida para verificar si la tabla que guarda los mensajes ya existe. Si no existe, la crea en ese momento.
        # 3. Ejecuta un SELECT para traer los registros
        return SQLChatMessageHistory(
            session_id=session_id,
            connection=self._database_url
        )

    # Limpia de la bd todos los registros (los borra)
    def clear_history(self, session_id: str) -> None:
        """Elimina el historial de una sesion"""
        history = self.get_history(session_id)
        history.clear()


    def list_sessions(self) -> list(str):
        """Lista todos los session ID disponibles"""
        # creando el "motor" de conexión. create_engine toma la URL de tu base de datos (Postgres) 
        # y prepara toda la maquinaria para poder hablar con ella.
        engine = create_engine(self._database_url)
        try: 
            with engine.connect() as conn: #abrir conexion y cerrar la conexion con la bd al final (cerrarla aunque algo salga mal al final)
                result = conn.execute( # Aquí envías tu consulta SQL pura (envuelta en la función text() de SQLAlchemy por seguridad
                    text("SELECT DISTINCT session_id FROM message_store")
                )
                # crea una lista extrayendo la primera columna de cada fila, 
                # es decir me regresa solo las sesiones existentes en la bd y crea la lista con ellas
                return [row[0] for row in result] 
        except ImportError:
            return []