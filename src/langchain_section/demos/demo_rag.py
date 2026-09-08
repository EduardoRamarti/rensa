from pathlib import Path

from src.langchain_section.chains.rag import build_rag_chain
from src.langchain_section.core.document_loader import load_directory, split_documents
from src.langchain_section.core.embeddings import get_or_create_vectorstore

DOCUMENT_DIR = Path("data/documents")
COLLECTION_NAME = "demo_rag_real"
CHROMA_PATH = "./data/chromadb_rag_demo"


def index_documents() -> tuple:
    """Carga archivos reales desde dicso y divide en chunks"""
    docs = load_directory(DOCUMENT_DIR)

    if not docs: # si docs esta vacio (sin lista de docs cargados)
        return None, 0  # (vectorstore, num_chunks)

    chunks = split_documents(docs) # se cortan los docs en chunks 

    if not chunks:
        print("❌ No se generaron chunks. Los archivos podrían estar vacíos.")
        return None, 0

    # Persistencia de los datos

    vectorstore = get_or_create_vectorstore( # crea la bd vectorial y lo devuelve como un objeto de ChromaDB
        collection_name=COLLECTION_NAME,
        persist_path=CHROMA_PATH
    )

    current_count = vectorstore._collection.count()

    if current_count > 0:
        print(f"\n. Ya hay {current_count} chunks indexados")
        answer = input("¿Reindexar desde cero? (s/N): ").strip().lower() # pregunta si quiere reindexar documentos nuevos o existentes

        if answer == "s":
            vectorstore._client.delete_collection(COLLECTION_NAME) # Elimina completamente la colleccion 
            vectorstore = get_or_create_vectorstore( # Vuelve a indexar los documentos 
                collection_name=COLLECTION_NAME,
                persist_path=CHROMA_PATH
            )

            vectorstore.add_documents(chunks) # añade los chunks de los documentos a la collection 
            print(f"Reindexado: {len(chunks)} chunks en ChromaDB")

        else:
            print(f"Usando índice existente ({current_count} chunks)") # solo muestra la cantidad de chunks actualmente en la coleccion

    else: # en caso que la collection este vacia 
        vectorstore.add_documents(chunks)
        print(f"{len(chunks)} chunks indexados en ChromaDB")

    return vectorstore, vectorstore._collection.count() # regresa el objeto de chroma y la cantidad de chunks 


def show_used_source(docs: list) -> None:
    """Muestra archivos y fragmentos que uso el sistema"""
    if not docs:
        return

    print("\nFuentes consultadas: ")

    viewed_sources = {}

    for doc in docs:
        name = doc.metadata.get("file_name", "desconocida") # obtiene el nombre del archivo
        page = doc.metadata.get("page", None) # obtiene la pagina 
        start = doc.metadata.get("start_index", None) # obtiene desde donde inicia

        if name not in viewed_sources: # si el nombre del doc no esta aun en el dict
            viewed_sources[name] = [] # agrega el nombre y sera una lista vacia

        info = ""

        if page is not None:
            info = f"pag. {page + 1}"
        elif start is not None:
            info = f"pos. {start}"

        if info: # si info tiene informacion agregalo a su respectivo archivo 
            viewed_sources[name].append(info)

    for file, locations in viewed_sources.items(): # recorre el diccionario y obtiene el archivo y su locacion
        if locations: # si la locacion no esta vacia 
            print(f"{file} ({', '.join(locations)})")
        else:
            print(f"{file}")


def show_welcome(num_chunks: int) -> None: # Usado para dar el mensaje de bienvenida 
    """Muestra el mensaje de bienvenida con estado del sistema."""
    print("\n" + "=" * 60)
    print("🔍 RAG Interactivo — Chat con tus documentos")
    print("=" * 60)
    print(f"  Carpeta de documentos: {DOCUMENT_DIR}")
    print(f"  Chunks en el índice:   {num_chunks}")
    print()
    print("  Comandos:")
    print("    'archivos'  → ver archivos indexados")
    print("    'reindexar' → recargar archivos del disco")
    print("    'chunks'    → ver cantidad de chunks")
    print("    'salir'     → terminar")
    print("-" * 60)
    print()


def main() -> None:
    """Main"""
    print("=" * 60)
    print("Iniciando RAG con archivos reales...")
    print("=" * 60)

    vectorstore, num_chunks = index_documents()

    if vectorstore is None: # si la collection esta vacia 
        print("\nPasos para empezar")
        print(f" 1. Crea la carpeta: {DOCUMENT_DIR}")
        print(" 2. Agregar archivos .txt o .pdf")
        print(" 3. Vuelve a ejecutar este script")
        return

    rag_chain, retriever = build_rag_chain(vectorstore) #entrada al pipeline, regresa el runnable y el retriever 
    show_welcome(num_chunks)

    while True:
        try:
            user_input = input("Tu: ").strip()

            if not user_input:
                continue

            if user_input.lower() == "salir":
                print("Hasta luego")
                break

            if user_input.lower() == "archivos":
                files = list(DOCUMENT_DIR.glob("*.txt")) + list(DOCUMENT_DIR.glob("*.pdf")) + list(DOCUMENT_DIR.glob("*.TXT")) + list(DOCUMENT_DIR.glob("*.PDF"))
                files = list(set(files))

                if not files:
                    print(f"No hay archivos en {DOCUMENT_DIR}\n")
                else: 
                    print(f"\n Archivos en {DOCUMENT_DIR}")
                    for file in sorted(files):
                        size_kb = file.stat().st_size / 1024
                        print(f"{file.name} ({size_kb:.1f}) KB")
                    print()
                continue

            if user_input.lower() == "chunks":
                count = vectorstore._collection.count()
                print(f"\n Chunks en ChromaDB: {count}\n")
                continue

            if user_input.lower() == "reindexar":
                print("\n Reindexando...")
                vectorstore, num_chunks = index_documents()
                if vectorstore:
                    rag_chain, retriever = build_rag_chain(vectorstore)
                    print(f"Listo. {num_chunks} chunks disponibles \n")
                continue

            retrieved_docs = retriever.invoke(user_input) # se esta ejecutando exclusivamente la fase de búsqueda en tu base de datos

            if not retrieved_docs:
                print("\n IA: No encontre información relevante en los documentos.\n")
                continue

            print("\n IA: ", end="", flush=True)
            answer = rag_chain.invoke(user_input) # hace la llamada para que corra la chain (pipeline) y me regrese y regresa la respuesta del llm
            print(answer)

            show_used_source(retrieved_docs) #muestra de que archivo se obtuvieron 
            print()
        except KeyboardInterrupt:
            print("\nHasta luego\n")
            break
        except Exception as e:
            print(f"\n Error: {e}\n")


if __name__ == "__main__":
    main()
