from abc import ABC, abstractmethod

from langchain_core.chat_history import BaseChatMessageHistory


class BaseMemoryBackend(ABC):
    """Contrato que debe cumplir cualquier Backend de memoria"""

    # @abstractmethod → método que se declara en una clase base pero no se implementa, obligando a cualquier clase hija (subclase) a escribir el código para ese método.
    @abstractmethod
    def get_history(self, session_id: str) -> BaseChatMessageHistory: # Recibe un session_id y retorna un BaseChatMessageHistory
        """Retorna el historial de mensajes para una sesion"""

    @abstractmethod
    def clear_history(self, session_id: str) -> None: # Recibe un session_id, sin retornos
        """Elimina el historial de una sesion"""

    @abstractmethod
    def list_sessions(self) -> list(str): 
        """Lista todos los session ID disponibles"""

