"""Cliente LLM centralizado"""
"""Create the LLM client"""

from functools import lru_cache

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from src.langchain_section.config.settings import settings


@lru_cache(maxsize=1) # Guardar el resultado de una funcion en cachen y cuando se llame regresaria el resultado de la misma funcion llamada (siempre y cuando sean los mismos argumentos)
def get_llm(temperature: float = None) -> ChatOpenAI: # recibe una temperatura y retorna una instancia de ChatOpenAI
    """retorna cliente LLM"""
    return ChatOpenAI(
        model= settings.CHAT_MODEL,
        temperature= temperature or settings.DEFAULT_TEMPERATURE,
        max_retries = settings.MAX_RETRIES
    )

@lru_cache(maxsize=1)
def get_embeddings() -> OpenAIEmbeddings: # Inicializa una instancia de OpenAIEMbeddings
    """"Retorna el cliente de embeddings"""
    return OpenAIEmbeddings(model=settings.EMBEDDING_MODEL) # establece el modelo 