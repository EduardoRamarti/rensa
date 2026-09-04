"""Configuracion General"""
"""Used to Set LangChain configuration """
import os

from dotenv import load_dotenv

from src.langchain_section.config import settings

load_dotenv()


class LangChainSettings:
    """Configuracion"""

    # modelos
    CHAT_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    EMBEDDING_MODEL: str = "text-embedding-3-small"

    # Parametros LLM
    DEFAULT_TEMPERATURE: float = 0.7
    LOW_TEMPERATURE: float = 0.1
    MAX_RETRIES: int = 3

    # Rutas de persistencia
    SQLITE_DB_PATH: str = "data/chat_history.db"
    CHROMA_PATH: str = "./data/langchain_chroma"

    # RAG
    CHUNK_SIZE: str = 500
    CHUNK_OVERLAP: int = 50
    TOP_K_RESULTS: int = 3

    # Costos aproximados
    COST_INPUT_PER_MILLION: float = 0.15
    COST_OUTPUT_PER_MILLION: float = 0.60

    @classmethod # Let us exec this method without an instance
    def validate(cls) -> None:
        """Valida variables criticas"""
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError(
                "OPENAI_API_KEY no esta configurada"
                "Crea un archivo .env y prega tu APIKey"
            )


settings = LangChainSettings()
