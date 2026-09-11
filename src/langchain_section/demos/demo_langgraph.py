import uuid
from pathlib import Path

from langchain.messages import HumanMessage

from src.langchain_section.core.document_loader import load_directory, split_documents
from src.langchain_section.core.embeddings import get_or_create_vectorstore
from src.langchain_section.graphs.rag_agent import build_rag_agent
from src.langchain_section.graphs.states import RAGAgentState
from src.langchain_section.memory.base import BaseMemoryBackend
from src.langchain_section.memory.postgresql_memory import PostgreSQLMemoryBackend
from src.langchain_section.memory.sqlite_memory import SQLiteMemoryBackend

DOCUMENTS_DIR = Path("data/documents")
COLLECTION_NAME = "knowledge_base"
CHROMA_PATH = "./data/chromadb_knowledge"


def setup_vectorstore():
    """Cargar o inicializar vectorstore"""
    vectorstore = get_or_create_vectorstore( # Crea u obtiene el vectorstore de Chroma (un objeto de chroma db)
        collection_name=COLLECTION_NAME, persist_path=CHROMA_PATH
    )

    count = vectorstore._collection.count() # El numero de chunks indexados en la collection de ChromaDB

    if count > 0: # si la coleccion tiene chunks indexados: 
        print(f"Base de conocmiento: {count} chunks indexados")
        return vectorstore #retorna el objeto de ChromaDB

    print("Base de conocimiento vacia. Indexando documentos...")
    docs = load_directory(DOCUMENTS_DIR) # Carga todos los archivos soportados en la carpeta DOCUMENTS_DIR y retorna una lista de objetos Document de Langchain

    if not docs:
        print(f"\n Agrega archivos .txt o .pdf en {DOCUMENTS_DIR}")
        return None

    chunks = split_documents(docs) # Divide los docs en chunks
    vectorstore.add_documents(chunks) # Indexa los chunks en la collection de ChromaDB

    print(f"{len(chunks)} chunks indexados")
    return vectorstore

def setup_memory_backend() -> BaseMemoryBackend:
    """Intentar conectar a PostgreSQL o en su defecto usar SQLite"""
    try:
        backend = PostgreSQLMemoryBackend() # Crear backend de memoria usando PostgreSQL 
        backend.list_sessions() # Lista las sesiones para verificar que la conexión a la base de datos es exitosa
        print("Memoria: PostgreSQL")
        return backend # Retorna el backend de memoria usando PostgreSQL
    except Exception as e:
        print(f"PostgeSQL no disponible: {e}")
        print("Usando SQLite como fallback de desarrollo")
        return SQLiteMemoryBackend() # retorna el backend de memoria usando SQLite como fallback

def select_session(backend: BaseMemoryBackend) -> str:
    """Seleccionar sesión o crear una nueva"""
    exists_sessions = backend.list_sessions() # se listan las sesiones existentes en la base de datos (PostgreSQL o SQLite)

    print("\n")
    print("=" * 55)
    print("Gestión de sesiones")
    print("=" * 55)

    if exists_sessions: # en caso de existir sesiones previas: 
        print(f"\nConversaciones guardadas ({len(exists_sessions)}): ") #  muestra la cantidad de sesiones guardadas
        for index, session_id in enumerate(exists_sessions, 1):
            messages = backend.get_history(session_id).messages # Trae los mensajes del session_id que se esta iterando y los guarda en messages
            last_message = ""
            if messages: # si existen mensajes
                last_message = f" - último: '{messages[-1].content[:40]}...'" # se obtiene el ultimo mensaje de la sesion y se limita a 40 caracteres para mostrarlo en pantalla
            print(f" {index}. {session_id}{last_message}") # le coloca el indice de la sesion, su session_id y el ultimo mensaje 

        print("\nOpciones:")
        print("  n -> Nueva conversación")
        print("  1,2,3... -> Retomar conversación existente")
        print("  ID -> Escribir un session_id específico")

        choise = input("\nElige: ").strip().lower()

        if choise == "n": # empieza una nueva sesion 
            session_id = str(uuid.uuid4()) # genera nuevo session_id 
            print(f"\nNueva sesión: {session_id}")
            return session_id # retorna el nuevo session_id

        if choise.isdigit(): # si fue un numero, es decir, un indice de las sesiones existentes 
            idx = int(choise) - 1 # convierte el indice a 
            if 0 <= idx < len(exists_sessions): # si el indice es mayor o igual a 0 y el indice es menor al total de sesiones 
                session_id = exists_sessions[idx] # obtiene el session_id de la lista de sesiones existentes usando el indice
                messages = backend.get_history(session_id).messages # Obtiene los mensajes de la conversacion con el session_id seleccionado y los guarda en messages
                print(f"\n Retomando sesión: {session_id}")
                print(f" {len(messages)} mensajes previos")
                return session_id
            else:
                print("Número inválido. Creando nueva sesión")

        if choise and choise != "n": # si choise no esta vacio y es diferente a n (nueva sesion)
            session_id = choise # se asume que el usuario ingreso un session_id especifico
            messages = backend.get_history(session_id).messages # obtiene los mensajes del historico (bd)
            if messages:
                print(f"\n Retomando sesión: {session_id}")
                print(f" {len(messages)} mensajes previos")
            else:
                print(f"\n Nueva sesión con ID: {session_id}")
            return session_id

        session_id = str(uuid.uuid4()) # genera un nuevo session_id en caso de que el usuario no haya ingresado nada o haya ingresado algo invalido
        print(f"\nNueva sesión: {session_id}")
        return session_id

    else: # en caso de que no existan sesiones previas:
        session_id = str(uuid.uuid4())
        print(f"\nPrimera sesión: {session_id}")
        return session_id

def load_history(backend: BaseMemoryBackend, session_id: str) -> list:
    """Carga mensajes previos"""
    history = backend.get_history(session_id)
    return list(history.messages)

def save_messages(backend: BaseMemoryBackend, session_id: str, human_message: str, ai_message: str) -> None:
    """Persiste el turno de conversacion en la base de datos"""
    history = backend.get_history(session_id) # obtiene el historial de la sesion actual y lo guarda en history
    history.add_user_message(human_message) # agrega el mensaje del usuario al historial de la sesion actual
    history.add_ai_message(ai_message) # agrega el mensaje de la IA al historial de la sesion actual

def run_chat( # recibe un agent (grafo compilado), un backend de memoria y un session_id
    agent,
    backend: BaseMemoryBackend,
    session_id: str
) -> None:
    history = load_history(backend, session_id) # carga el historial de mensajes previos de la sesion seleccionada (si es que existe) y lo guarda en history
    print(f"\n{'=' * 55}")
    print("Asistente de Conocimiento Empresarial")
    print(f"Sesión: {session_id}")
    print(f"{'=' * 55}")
    print("Comandos: 'sesion' | 'historial' | 'limpiar' | 'salir'")
    print("-" * 55)
    print()

    while True: # comienza bluce de interaccion con el usuario
        try:
            user_input = input("Tu: ").strip()

            if not user_input:
                continue

            if user_input.lower() == "salir":
                messages = backend.get_history(session_id).messages # obtiene los mensajes de la sesion actual y los guarda en messages
                print(f"\nSesion guardada: {session_id}")
                print(f"{len(messages)} mensajes en {'PostgreSLQ' if isinstance(backend, PostgreSQLMemoryBackend) else 'SQLite'}")
                break
            if user_input.lower() == "sesion":
                print(f"\nID de la sesion actual {session_id}")
                messages = backend.get_history(session_id).messages
                print(f"Mensajes en esta sesion: {len(messages)}")
                continue
            if user_input.lower() == "historial": # muestra el historial de la sesion actual
                messages = backend.get_history(session_id).messages
                if not messages:
                    print("Historial vacio\n")
                    continue
                print("\nUltimos mensajes de la sesion: ")
                for message in messages[-6:]: # hace el recorrido de los ultimos 6 mensajes de la sesion 
                    rol = "Tu" if message.type == "human" else "IA" # asigna el rol dependiendo de si el mensaje es del usuario o de la IA
                    print(f"{rol}: {messages.content[:90]}...")
                print()
                continue
            if user_input.lower() == "limpiar": # limpia el historial de la sesion actual
                backend.clear_history(session_id) # borra todos los mensajes de la sesion actual en la base de datos
                history = []
                print("Historial de esta sesion borrada.\n")
                continue

            history = load_history(backend, session_id) # carga el historial de mensajes previos de la sesion seleccionada (si es que existe) y lo guarda en history
            initial_state: RAGAgentState = { # crea el estado inicial del agente RAG (contrato) con los mensajes previos, la pregunta actual del usuario y otros campos necesarios para el flujo de la conversación
                # por cada iteracion se destruye el estado actual y se crea uno nuevo con los mensajes previos y la pregunta actual del usuario
                "messages": history + [HumanMessage(content=user_input)],
                "question": user_input,
                "retrieve_docs": [],
                "response": "",
                "needs_retrieval": True,
                "sources": [],
            }

            print()

            # sa única línea de código es la que pone a trabajar toda la arquitectura
            # El grafo de LangGraph se encarga de pasar el contrato (state) entre los nodos (funciones) y ejecutar el flujo de la conversación del agente RAG.
            final_state = agent.invoke(initial_state) 

            # el final_state es simplemente un diccionario normal de Python, el "contrato" final. 
            # Durante el viaje por la arquitectura, tu última función (node_generate) hizo esto por dentro: 
            # estado["response"] = "texto generado por el LLM"
            response = final_state["response"]

            user_retrieval = bool(final_state.get("retrieve_docs")) # si retrieve_docs tiene documentos, significa que el agente RAG tuvo que buscar en la base de conocimiento para responder la pregunta del usuario. Si no, significa que el agente RAG respondió directamente sin buscar en la base de conocimiento.

            if user_retrieval: # en caso que se encontraran documentos en la base de conocimiento para responder la pregunta del usuario, se imprime un mensaje indicando que la respuesta fue obtenida a partir de la búsqueda en documentos. Si no, se imprime un mensaje indicando que la respuesta fue generada directamente por el agente RAG sin buscar en documentos.
                print(f"AI [busco en documentos]: {response}")
            else:
                print(f"AI [respondio directo]: {response}")

            sources = final_state.get("sources", []) # obtiene las fuentes de los documentos consultados para responder al usuario. Si no los uso, regresa la lista vacia 

            if sources: # si hubo fuentes consultadas: 
                print("\nFuentes consultadas")
                show = set()
                for source in sources: # recorrer la lista de sources 
                    key = f"{source['file']}_p{source['page']}" # formatea la fuente con el nombrel del archivo y el numero de pagina 
                    if key not in show: # si la fuente no ha sido mostrada antes, se imprime en pantalla y se agrega a la lista de fuentes mostradas
                        print(f"{source['file']} (pag.{source['page']})")
                        show.add(key)
            print()

            save_messages(backend, session_id, user_input, response) # guarda el turno de conversacion, se le da como argumento el backedn, un session_id, la pregunta del usuario y la respuesta de la IA para que se guarde en la base de datos (PostgreSQL o SQLite)

        except KeyboardInterrupt: # en caso que el usuario interrumpa la ejecucion con Ctrl+C
            print("\nHasta luego\n")
            print(f"ID de sesion: {session_id}")
            break
        except Exception as e: # si sucede algun error durante la ejecucion del buble. 
            print(f"Error: {e}")

# Funcion de entrada principal
def main()->None:

    print('='*50)
    print("Asistente de conocimeintos empresarial")
    print('='*50)

    # Configurar vectorstore → Crear almacenamiento de embeddings si no existe
    vectorstore = setup_vectorstore()

    if vectorstore is None:
        return

    backend = setup_memory_backend() # Crea el backend de memoria (PostgreSQL o SQLite) dependiendo de la disponibilidad

    agent = build_rag_agent(vectorstore) # Regresa el objeto (grafo) compilado (ejecutable) del agente RAG, listo para ser invocado con un contrato (state) inicial.
    print("Grafo LangGraph compilado")

    session_id = select_session(backend) # Selecciona una sesion existente o crea una nueva, retorna el session_id

    run_chat(agent, backend, session_id) # Inicia el bucle de interaccion con el usuario, pasando el agente RAG, el backend de memoria y el session_id seleccionado o creado.


if __name__ == '__main__':
    main()
