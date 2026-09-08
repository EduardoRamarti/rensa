import os

from langchain_chroma import Chroma

from src.langchain_section.config.settings import settings
from src.langchain_section.core.llm import get_embeddings


def get_or_create_vectorstore(
        collection_name: str, # Nombre de la coleccion 
        persist_path: str = None, # Path donde vive la db vectorial
) -> Chroma:
    """Obtener un vector store existente"""
    path = persist_path or settings.CHROMA_PATH # si no se comparte un path, utiliza el Path por default
    os.makedirs(path, exist_ok=True) # crea el dir de la db

    return Chroma( #retornar el objeto de chroma db
        collection_name=collection_name,
        embedding_function=get_embeddings(), # con la funcion de embeddings obtenemos el objeto de embeddings del llm
        persist_directory=path
    )
