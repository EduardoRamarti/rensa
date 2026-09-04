"""Demo LCEL"""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from src.langchain_section.core.llm import get_llm

llm = get_llm() # Hace instancia del LLM


def demo_simple_chain() -> None:
    """Cadena simple"""
    """Forma básica y lineal de conectar componentes usando LCEL"""
    # ChatPromptTemplate (role, contenido)
    prompt = ChatPromptTemplate.from_messages(
    #Generador de plantillas universal para APIs de chat.
        [
            (
                "system",
                "Eres un experto en {tema}. Responde de forma concisa, maximo 3 oraciones",
            ),
            ("human", "{pregunta}"), #mismo rol que user
        ]
    )

    # StrOutParser
    # 1 Actúa como un "filtro" al final de la tubería.
    # 2 Recibe ese objeto gigante AIMessage.
    # 3 Busca automáticamente la propiedad content.
    # 4 Lo extrae y desecha todo el resto (los metadatos, los tokens, etc.).
    # 5 Te devuelve única y exclusivamente un string puro de Python.
    parser = StrOutputParser()

    # Cadena (Chain) Blueprint
    # Solo estás conectando las tuberías. Le estás diciendo a Python: "Cuando te dé la orden, quiero que los datos viajen por aquí".
    chain = prompt | llm | parser

    # Ejecuta un solo input de principio a fin y te da la respuesta final de golpe. Como el create de OpenAI
    response = chain.invoke({"tema": "Python", "pregunta": "¿Que es un decorador?"})

    print("Simple Chain")
    print(response)
    print()


def demo_steps_inspection() -> None:
    """Invoca cada componente"""
    """ autopsia de la cadena (blueprint) """
    prompt = ChatPromptTemplate.from_messages(
        [("system", "Eres un asistente tecnico."), ("human", "{pregunta}")]
    )

    parser = StrOutputParser()

    input_message = {"pregunta": "Que es una API REST?"}

    # Paso 1 Prompt
    messages = prompt.invoke(input_message)
    print("Paso 1 - Prompt Output")
    print(f"Tipo: {type(messages).__name__}")
    print(f" Mensajes: {messages.messages}")

    # Paso 2 LLM -> AIMessage
    ai_message = llm.invoke(messages)
    print("Paso 2 - LLM Output")
    print(f" Tipo: {type(ai_message).__name__}")
    print(f" AIMessage: {ai_message}")
    print(f" Mensajes: {ai_message.content[:100]}...")

    # Paso3: Parser extraer texto
    text = parser.invoke(ai_message)
    print("Paso 3 - Parser Output")
    print(f" Tipo: {type(text).__name__}")
    print(f" Mensajes: {text[:100]}...")
    print()


def demo_batch() -> None:
    """Batch Method"""
    """Procesamiento concurrente de múltiples entradas"""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Clasifica el sentimiento. SOLO: positivo, negativo, o neutro."),
            ("system", "{texto}"),
        ]
    )

    chain = prompt | llm | StrOutputParser()

    inputs = [
        {"texto": "Me encanta este framework, es increible"},
        {"texto": "El servidor estuvo caido 3 horas. Es inaceptable"},
        {"texto": "La version 2.0 esta disponible"},
        {"texto": "Perdi todos mis datos por un bug critico"},
        {"texto": "La documentacion es muy clara."},
    ]

    results = chain.batch(inputs)
    # Batch multiples inputs en paralelo → concuerrencia

    print("BATCH PROCESSING")
    for input_message, result in zip(inputs, results):
        print(f" [{result}] {input_message['texto'][:50]}...")
    print()


def demo_streaming() -> None:
    """Metodo streaming"""
    """Generación de respuestas en tiempo real (Streaming)."""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Explica conceptos tecnicos de forma clara"),
            ("human", "Explica que es {concepto} en dos parrafos"),
        ]
    )

    chain = prompt | llm | StrOutputParser()

    print("STREAMING EN TIEMPO REAL.")
    print("IA: ", end="", flush=True)

    # con forme va obteniendo cada chunk en este codigo, o regresa para imprimirlo sobre la misma linea en la que esta escribiendo
    for chunk in chain.stream({"concepto": "La ventana de contexto en LLMs"}):
        print(chunk, end="", flush=True)

    print()


def demo_passthrough() -> None:
    """Passthrough"""
    """Inyección de contexto dinámico (Simulación de RAG)."""
    """Preparación dinámica de variables: Demuestra cómo tomar un input simple del usuario (un string), buscar información adicional en una base de datos (retriever) y usar RunnablePassthrough para conservar la pregunta original, armando automáticamente el diccionario que necesita el Prompt."""

    # Simula retriever
    #recibir una pregunta (str) y regresar un contexto (str). tiene su "almacenado o bd simulada" y recorre esa base para poder devolver el contexto que contenga la palabra dentro de las keys del almacenamiento
    def search_context(question: str) -> str:
        contexts = {
            "python": "Python fue creado por Guido Van Rossum en 1991.",
            "langchain": "LangChain es un framework para aplicaciones con LLMs",
            "developer": "Persona que se dedica a desarrollo de software.",
        }

        for keyword, ctx in contexts.items():
            if keyword.lower() in question.lower():
                return ctx
        return "No se encontro contexto relevante"

    # RunnableLambda → Su trabajo es tomar una función normal de Python y "disfrazarla" para que LangChain la trate como si fuera un componente nativo de su sistema.
    retriever = RunnableLambda(search_context)

    # Generacion del prompt 
    prompt = ChatPromptTemplate([
        ("system", "Responde usando este contexto: \n {context}"),
        ("human", "{question}")
    ])

# Cuando estás programando la cadena y escribes RunnablePassthrough(), literalmente estás dejando una nota que dice: "Aquí va a ir el input original, guárdame el lugar".
# Todo el bloque del diccionario está en "pausa", esperando. En el instante en que haces chain.invoke("Que es python"), ese evento actúa como el gatillo.
# Ocurre esto:
# Llega el input.
# El diccionario actúa como un divisor de corriente eléctrica: duplica ese input.
# Ejecución paralela:
# Manda una copia por el cable del retriever (que hace el trabajo pesado de buscar).
# Manda otra copia por el cable del RunnablePassthrough() (que simplemente la deja caer al otro lado).
    chain = (
        {
            #Cuando LangChain ve un diccionario literal {} conectado a un pipe |, lo convierte automáticamente por detrás en un componente llamado RunnableParallel (ejecución en paralelo).
            "context": retriever,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    response = chain.invoke("¿Que es python?")
    print("PASSTRHROUGH")
    print(response)
    print()


if __name__ == "__main__":
    print("=" * 60)
    print("LangChain LCEL - fundamentos")
    print("=" * 60)
    # demo_simple_chain()
    # demo_steps_inspection()
    # demo_batch()
    # demo_streaming()
    demo_passthrough()
