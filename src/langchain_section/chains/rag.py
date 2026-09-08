"""Pipeline de RAG"""

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core. output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough

from src.langchain_section.config.settings import settings
from src.langchain_section.core.llm import get_llm


def format_docs(docs: list[Document]) -> str: #Recibe una lista de documentos. Retorna el texto del documento
    # Recibe esa lista de documentos. 
    # Como los LLMs no entienden objetos de Python, 
    # esta función toma cada documento, saca de la metadata la fuente y la página
    """Convierte lista de documents en texto para el prompt """
    return "\n\n---\n\n".join([
        f"[Fuente: {doc.metadata.get('source', 'desconocida')},"
        f"Página: {doc.metadata.get('page', "N/A")}]\n {doc.page_content}"
        for doc in docs
    ])

def build_rag_chain(vectorstore: Chroma) -> tuple[Runnable, object]:# se convierete en runnable
    """Pipeline RAG con LCEL"""
    retriever = vectorstore.as_retriever( # componente de busqueda semantica → Lo que hace es convertir la base de datos Chroma en un "buscador inteligente"
        search_type="similarity",
        search_kwargs={"k": settings.TOP_K_RESULTS} #tope de chunks → define cuántos chunks te devuelva cada vez que alguien pregunte algo (en este caso 3)
    ) 

    prompt = ChatPromptTemplate.from_messages([
        ("system", """Eres un asistente que responde preguntas
basándote ÚNICAMENTE en el contexto proporcionado.

Contexto recuperado de los documentos:
{context}

Instrucciones:
- Si la respuesta está en el contexto, respóndela con precisión.
- Si no está, di: "No encontré esa información en los documentos."
- Cita la fuente cuando sea posible.
- No inventes ni supongas información."""),
        ("human", "{question}")
    ])


    rag_chain = ( # se crea la chain (runnable) tambien conocido como pipe
        {
            "context":  retriever | format_docs, # arma el lboque de texto con los pedazos encontrados → primero el retriever, lo que salga de el, va a format docs
            "question": RunnablePassthrough() #deja que se inyecte la pregunta tal cual se realizo 
        }
        | prompt
        | get_llm(temperature=settings.LOW_TEMPERATURE)
        | StrOutputParser()
    )

    return rag_chain, retriever
