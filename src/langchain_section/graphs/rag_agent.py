
from functools import partial

from langchain_chroma import Chroma
from langgraph.graph import END, START, StateGraph

from src.langchain_section.graphs.nodes import (
    decide_retrieval_path,
    node_analyze,
    node_generate,
    node_retrieve,
)
from src.langchain_section.graphs.states import RAGAgentState


def build_rag_agent(vectorstore: Chroma):
    """Construir y compilar RAG agentico"""
    # Se instancia stategraph con el contrato (estado) de RAGAgentState
    # StateGraph lo que hace es crear un grafo de estados que representa el flujo de la conversación del agente RAG, donde cada nodo es una función que opera sobre el estado del agente.
    # en otras palabras, stategraph se encarga de pasar el contrato entre las funciones (nodos)
    builder = StateGraph(RAGAgentState) 

    # partial se usa para crear una nueva función a partir de otra (clonarla):
    # fijando algunos argumentos. En este caso, se fija el argumento vectorstore de node_retrieve, 
    # creando una nueva función node_retrieve_with_vs que solo necesita el estado del agente como argumento.
    node_retrieve_with_vs = partial(node_retrieve, vectorstore=vectorstore)

    # Basicamente decirle a los nodos: 
    # vas primero, luego tu y al final tu. (cuando se ejecute el invoke)
    # La primera funcion (nodo) va a recibir el estado y hacer un attach de lo que regrese ese primer nodo. 
    # Luego vas con el segundo ya con el estado actualizado, etc con el tercero
    builder.add_node("analyze", node_analyze)
    builder.add_node("retrieve", node_retrieve_with_vs)
    builder.add_node("generate", node_generate)

    # Punto de entrada  al grafo. 
    # Lo prmero que haces al arrancar es ejecutar el nodo analyze 
    builder.add_edge(START, "analyze")

    # Ejecuta mi función para decidir, y usa un IF para ver qué ruta tomar
    builder.add_conditional_edges( # El motor arranca e inyecta el contrato (el state) al nodo analyze.
        "analyze", # Hace su petición HTTP a OpenAI, obtiene el JSON y actualiza el contrato: state["needs_retrieval"] = True
        
        # El motor ve que hay una bifurcación. Toma el mismo contrato actualizado y se la inyecta a la función de decisión
        # Esa función lee la cubeta (if state["needs_retrieval"]), actúa como un if/else tradicional, 
        # y devuelve un simple texto, por ejemplo: "retrieve"
        decide_retrieval_path, 
        {# este diccionario es un mapa de rutas: El diccionario de rutas traduce la llave que nos regreso decide_retrieval_path a un nombre de nodo
            "retrieve": "retrieve",
            "generate": "generate"
        }# El framework usa ese nombre para buscar en su registro interno qué código de Python (qué función) está asociada a ese nombre.
        # LangGraph ejecuta la función correspondiente en seguida, pasando el contrato actualizado (state) como argumento.
    )

    # Aqui se dice estas dos son rutas de seguimiento que van por parte del grafo.
    # Es decir, si decide_retrieval_path devuelve "retrieve", entonces se ejecuta el nodo retrieve y luego se ejecuta el nodo generate.
    builder.add_edge("retrieve", "generate")

    # En caso de que decida_retrieval_path devuelva "generate", entonces se ejecuta el nodo generate y luego se termina el flujo.
    builder.add_edge("generate", END)

    # Convierte el "plano" (blueprint) de configuración en el ejecutable final.
    # El compilador de LangGraph toma el grafo de estados y genera un objeto que puede ser invocado con un contrato (state) inicial.
    # 1. Valida las rutas: Revisa que el grafo tenga sentido y no haya errores (por ejemplo, que un add_edge no esté apuntando a un nodo que se te olvidó crear, o que no haya un callejón sin salida).
    # 2. Congela el mapa: Bloquea todas las reglas de los nodos y los edges para que ya no se puedan modificar.
    # 3. Crea el motor: Toma todo ese mapa congelado y te devuelve un objeto nuevo (el grafo compilado) que por fin expone el método .invoke() o .stream().
    return builder.compile()

