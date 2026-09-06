from langchain_core.runnables import RunnableWithMessageHistory

from src.langchain_section.chains.base import build_assistant_chain
from src.langchain_section.memory.base import BaseMemoryBackend
from src.langchain_section.memory.postgresql_memory import PostgreSQLMemoryBackend
from src.langchain_section.memory.sqlite_memory import SQLiteMemoryBackend


def build_chatbot(backend: BaseMemoryBackend) -> RunnableWithMessageHistory:
    """Construyendo chatbot
    Args:
        backend: Cualquier implementacion BaseMemory Backend
    Returns:
        Chatbot: listo para invocar con session_id
    """
    # 1. Crear el chain base 
    chain = build_assistant_chain()

    # RunnableWithMessageHistory → En un solo parrafo: 
    # Es una cadena (runnable) que inyecta en mi cadena original el historial que obtuvo 
    # y lo junta con mi input al realizar el invoke. 
    # Luego  guarda dos registros en mi bd uno con el input de usuario como human y el otro como ai 
    return RunnableWithMessageHistory( # Genera runnable con memoria
        chain, # Runnable que sera la cadena (pipeline) a utilizar 
        backend.get_history, # El objeto llama a su metodo get_history para obtener el historial de la bd
        # los dos siguientes se realizan cuando se hizo el chain 
        input_messages_key="input", # Le dice a la cadena: "Cuando el usuario te ejecute usando .invoke(), busca su nuevo mensaje dentro de la llave llamada 'input'
        history_messages_key="history" # Le dice a la cadena: "Una vez que vayas a la base de datos y recuperes los mensajes pasados, inyéctalos en el Prompt Template usando el nombre 'history'
    )

def run_chat_session(
        chatbot: RunnableWithMessageHistory, #recibe el nuevo runnable
        backend: BaseMemoryBackend, # recibe el backend del tipo de base de datos que se esta usando
        session_id: str # recibe un session id 
) -> None:
    print(f"\nSesion activa:{session_id}")

    messages = backend.get_history(session_id=session_id).messages # regresa la lista de mensajes que se guardaron el la bd y tambien crea por primera vez la tabla en la bd

    if messages:
        print(f"Retomando la conversacion ({len(messages)}) mensajes previos")
    else: 
        print("Nueva conversacion")

    print("Comandos: 'historial' | 'limpiar' | 'sesiones' | 'salir'\n")

    while True:
        try:
            user_input = input("Tu: ").strip()

            if not user_input:
                continue

            if user_input.lower() == "salir":
                print(f"Hasta luego! (Historial Guardado)")
                break

            if user_input.lower() == "historial":
                messages = backend.get_history(session_id).messages
                if not messages:
                    print(f"[Historial vacio]\n")
                    continue

                print(f"Ultimos mensajes de: {session_id}")
                for message in messages[-6:]: # de todos los menssages que obtuvimos, trae a partir de los ultimos 6 
                    rol = "Tu" if message.type == "human" else "AI"
                    print(f" {rol}: {message.content[:70]}...")
                print()
                continue

            if user_input.lower() == "limpiar":
                backend.clear_history(session_id)
                print("HISTORIAL BORRADO\n")
                continue

            if user_input.lower() == "sesiones":
                sessions = backend.list_sessions()
                print(f"\n Sesiones disponibles: {sessions}\n")
                continue

            response = chatbot.invoke(
                {"input": user_input},
                #El config es el parámetro donde le indicas al sistema el identificador único (session_id) 
                # en el momento exacto de la ejecución, para que sepa de qué usuario específico debe ir a buscar 
                # y guardar el historial de la conversación
                config={"configurable": {"session_id": session_id}}
            )
            print(f"IA: {response}\n")

        except KeyboardInterrupt:
            print("Hasta luego!")
            break
        except Exception as e:
            print(f"Error: {e}\n")


if __name__ == '__main__':
    print("="*60)
    print("Memoria persistente con SQLITE y Postgres")
    print("="*60)

    print("\nQue Backend memory usar?")
    print("1. SQLite")
    print("2. PostgreSQL (requiere Docker ejecutandose)")

    choice = input("Elige (1/2): ").strip()

    if choice == "2":
        try:
            backend = PostgreSQLMemoryBackend()
            print("Conectado a PostgreSQL")
        except ValueError as e:
            print(f"{e}")
            print("Usando SQLite como respaldo...")
            backend = SQLiteMemoryBackend()
    else:
        backend = SQLiteMemoryBackend()
        print("Conectado a SQLite")

    chatbot = build_chatbot(backend)
    run_chat_session(chatbot, backend, "user_demo_001")