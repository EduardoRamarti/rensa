from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable

from src.langchain_section.core.llm import get_llm


# Funciona como un runnable 
def build_assistant_chain(system_prompt: str = None) -> Runnable:
    """Construye una cadena base del asistente con soporte en el historial"""
    default_system = """Eres un asistente tecnico experto en Python e IA.
    Tienes acceso al historial completo de esta conversacion.
    Usalo para dar respuestas contextuales y coherentes.
    Si el usuario hace referencia a algo anterior, recuerdalo."""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt or default_system),
        MessagesPlaceholder(variable_name="history"), # Insertar el historial de cada sesion antes de llamar al LLM 
        ("human", "{input}")
    ])

    return prompt | get_llm() | StrOutputParser()
