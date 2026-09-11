""" Los nodos son funciones """
import json
from typing import Literal

from langchain.messages import AIMessage, HumanMessage
from langchain_chroma import Chroma

from src.langchain_section.config.settings import settings
from src.langchain_section.core.llm import get_llm
from src.langchain_section.graphs.states import RAGAgentState


def node_analyze(state: RAGAgentState) -> dict: #  Recibe el estado del agente (contrato) → retorna un diccionario con la decision de si necesita buscar documentos o no
    """Nodo que decide si la pregunta necesita buscar documentos"""
    llm = get_llm(temperature=0.1) # Inicializa el cliente LLM con una temperatura baja 

    recent_context = ""
    if state.get("messages") and len(state["messages"]) >= 2: # Evalua si hay al menos 2 mensajes en el historial de la conversación
        last_two = state["messages"][-2:] # solo devuelve los ultimos 2 mensajes 
        recent_context = "\n".join([ # Hace un join a la lista proporcionada de los ultimos 2 mensajes 
            f"{'Usuario' if message.type == 'human' else 'IA'}: {message.content[:150]}" # le agrega al mensaje un prefijo dependiendo de quien es el mensaje 
            for message in last_two # recorre los dos ultimos mensajes 
            if hasattr(message, 'content') and message.content # primero dice, "tiene sl attribute tal?" si lo tienes "tambien ese attribute tiene algo dentro?"
        ])

    # Crea el prompt para el LLM. Agrega el contexto que se obtuvo de los ultimos 2 mensajes y la pregunta actual del usuario.
    prompt = f"""Analiza si esta pregunta necesita buscar en la base de conocimiento empresarial.
Contexto inmediato (últimos 2 mensajes):
{recent_context if recent_context else 'Sin contexto previo'} 
Pregunta actual: {state['question']}
Responde ÚNICAMENTE con este JSON (sin markdown):
{{"needs_retrieval": true, "reason": "razón breve"}}
REGLAS ESTRICTAS:
needs_retrieval = true SIEMPRE que:
  - La pregunta pida información específica (políticas, procedimientos, datos)
  - Mencione documentos, manuales, contratos o información interna
  - Sea una pregunta factual sobre la empresa o sus procesos
  - Haya cualquier duda
needs_retrieval = false SOLO cuando sea OBVIO:
  - Saludos puros: "hola", "gracias", "adiós"
  - Aclaración de lo que la IA dijo en el mensaje inmediato anterior
En caso de duda: needs_retrieval = true"""

    result = llm.invoke([HumanMessage(content=prompt)]) # crea el mensaje del usuario que se le pasa al LLM 

    try:
        content = result.content.strip() # quita los espacios en blanco 
        if "```" in content: # revisa si la respuesta tiene las comillas de markdown (y si las tiene hace lo siguiente)
            content = content.split("```")[1] # Hace el split y regresa el segundo elemento de la lista generada
            if content.startswith("json"): # revisa de nuevo si tiene el titulo 'json' al inicio (parte del markdown)
                #content = content[4:]
                content = content.removeprefix("json") # elimina el prefijo 'json'

        decision = json.loads(content.strip()) # toma ese texto plano, lee los corchetes y las comillas, verifica que la sintaxis sea correcta y construye un diccionario de Python
        needs_retrieval = decision.get("needs_retrieval", True) # obtiene el valor de needs_retrieval, en caso que no se encuentre, lo coloca como true. 
        reason = decision.get("reason", "") # Obtiene el valor de reason, en caso que no se encuentre, lo coloca como cadena vacia.

    except json.JSONDecodeError: # si existe un erro al intentar hacer el parseo del json (pasar de content a decision)
        needs_retrieval = True
        reason = "Error de parseo, buscando por defecto"

    print(f"[analyze] needs_retrieval={needs_retrieval} | {reason}")

    return {"needs_retrieval": needs_retrieval}



def node_retrieve(state: RAGAgentState, vectorstore: Chroma) -> dict: # Recibe el estado del agente (contrato) y el vectorstore (coleccion) → retorna un diccionario con los documentos recuperados y sus fuentes
    """Busca los chunks mas relevantes en ChromaDB"""
    print(f"[ retrieve] Buscando: '{state['question'][:60]}...'") # Obtiene del state la pregunta del usuario 

    docs = vectorstore.as_retriever( # Con el "buscador inteligente" (as_retriever) busca dentro de la coleccion 
        search_type="similarity",
        search_kwargs={"k": settings.TOP_K_RESULTS} #limita la cantidad de resultados a devolver 
    ).invoke(state["question"]) # invoke → ejecuta la busqueda en la coleccion, usando la pregunta dentro del estado (la pregunta del usuario)

    retrieved_texts = [doc.page_content for doc in docs] # page_content → obtiene el texto de cada documento (chunk) encontrado en la busqueda

    sources = [ # nueva lista de diccionarios donde se setean los metadatos de cada documento (fuente y pagina)
        {
            "file": doc.metadata.get("file_name", "desconocida"),
            "page": doc.metadata.get("page", "N/A"),
        }
        for doc in docs
    ]

    print(f" [retrieve] {len(docs)} chunks encontrados")

    for src in sources:
        print(f" -> {src['file']} (pag. {src['page']})") # muestra en pantalla los documentos encontrados y las paginas donde encontro similitudes 

    return { # diccionario con la lista de textos de los documentos encontrados y la lista de fuentes (metadatos)
        "retrieved_docs": retrieved_texts, 
        "sources": sources
    }



def node_generate(state: RAGAgentState) -> dict: #  Recibe el estado del agente (contrato) → retorna un diccionario con la respuesta generada por el LLM y el mensaje de la IA
    """Genera la respuesta"""
    llm = get_llm(temperature=settings.LOW_TEMPERATURE) # Inicializa el cliente LLM 

    if state.get("retrieved_docs"): # Verifica si el estado tiene a retrieved_docs como True (si se encontraron documentos en la busqueda)
        docs_text = "\n\n --- \n\n".join(state["retrieved_docs"]) # hace un join sobre los textos de los documentos encontrados, separandolos con un --- para que el LLM pueda diferenciarlos mejor.
        # le agrega al contexto los textos que encontro en la busqueda de los documentos 
        context_section = f"""INFORMACION DE LOS DOCUMENTOS EMPRESARIALES: {docs_text} 
        INSTRUCCIÓN: Basa tu respuesta principalmente en estos documentos.
        Si la información no está aquí, dilo claramente.
        """

    else: # en caso que no haya hecho un retrieve a los documentos: 
        context_section = "No se encontraron documentos relevantes. Responde con conocimiento general"

    history_text = ""
    if state.get("messages"): # verifica si hay mensajes anteriores (historial)
        previous_msgs = state["messages"][:-1] # toma todos los mensajes menos el ultimo (que es la pregunta actual del usuario)
        # Guarda los ultimos 6 mensajes del historial, si hay mas de 6. Si hay menos, agrega los que haya 
        recent_msgs = previous_msgs[-6:] if len(previous_msgs) > 6 else previous_msgs
        if recent_msgs: # verifica que si haya historial 
            history_text = "\n".join([ # hace el join sobre el list comprehension de los mensajes obtenidos 
                f"{'Usuario' if message.type == 'human' else 'Asistente'}: {message.content[:200]}" # le agrega un prefijo dependindo de quien es el mensaje y limita el mensaje a 200 chars
                for message in recent_msgs # recorre los mensajes recientes 
                if hasattr(message, 'content') and message.content #primero dice, "tiene sl attribute tal?" si lo tienes "tambien ese attribute tiene algo dentro?"
            ])

    # Crea el prompt para el LLM: 
    # Agrega el contexto que se obtuvo de los documentos encontrados 
    # y el historial de la conversación, junto con la pregunta actual del usuario.
    prompt = f"""Eres un asistente de conocimiento empresarial experto.
{context_section}
HISTORIAL RECIENTE:
{history_text if history_text else 'Inicio de conversación'}
PREGUNTA: {state['question']}
INSTRUCCIONES:
- Si tienes documentos, úsalos como fuente principal
- Cita los documentos cuando sea relevante
- Si algo no está en los documentos, dilo honestamente
- Usa el historial solo para referencias contextuales
- Responde en español de forma clara y profesional"""

    result = llm.invoke([HumanMessage(content=prompt)]) # llama al LLM con el prompt generado y obtiene la respuesta generada por el LLM

    # Con documentos si retrieved_docs tiene algo, Sin documentos si no tiene nada
    used_docs = "Con documentos" if state.get( 
        "retrieved_docs") else "Sin documentos"

    print(f" [generate] {used_docs} ({len(result.content)} chars)") # muestra en pantalla si se usaron documentos o no y la cantidad de caracteres de la respuesta generada por el LLM

    return { # regresa el dictionary con la respuesta generada por el LLM y el mensaje de la IA
        "response": result.content,
        "messages": [AIMessage(content=result.content)]
    }



def decide_retrieval_path(state: RAGAgentState) -> Literal["retrieve", "generate"]:
    """Función de decisión"""
    if state.get("needs_retrieval", True):
        return "retrieve"
    return "generate"

