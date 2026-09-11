
from typing import Annotated, TypedDict

from langgraph.graph import add_messages


# TypedDict para representar el estado del agente RAG, es una estructura de datos que define los campos 
# y sus tipos para el estado del agente RAG. Se asegura que las llaves sean extrictamente las definidas y que los valores sean del tipo especificado.
class RAGAgentState(TypedDict): # es un diccionario
    """Estados""" # es la estructura de datos que se difinio para que las funciones se pasen variables entre si sin perder informacion en el camino

    # Annotated sirve para acumular el historial de la conversación sin borrar lo que ya estaba
    # Este campo almacena el historial de mensajes de la conversación
    messages: Annotated[list, add_messages]
    question: str
    retrieve_docs: list[str]
    response: str
    needs_retrieval: bool 
    sources: list[dict]