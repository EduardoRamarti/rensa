from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.langchain_section.config.settings import settings

SUPPORTED_EXTENSION = {".txt", ".pdf"}


def load_file(file_path: Path) -> list[Document]:
    """Carga un archivo y retorna una lista de Documents de Langchain"""
    if not file_path.exists(): # en caso que el archivo no exista
        raise FileNotFoundError(f"Archivo no encontrado: {file_path}")

    extension = file_path.suffix.lower() # Obtiene la extension del archivo 

    if extension not in SUPPORTED_EXTENSION: # si la extension no es parte de las extensiones soportadas
        raise ValueError(
            f"Extension '{extension}' no soportada Usa: {', '.join(SUPPORTED_EXTENSION)}"
        )

    print(f"Cargando: {file_path.name}", end="")

    if extension == ".pdf": # si la extension es un pdf 
        # PyPDFLoader es una herramienta de LangChain que sirve para: 
        # leer, cargar y procesar archivos en formato PDF 
        loader = PyPDFLoader(str(file_path)) # recibe el path del archio como str y crea el objeto
        docs = loader.load() # .load() → función principal que ejecuta la lectura del archivo PDF y devuelve el contenido estructurado en una lista de objetos Document
        # Ejmeplo de docs: 
        # [ 
        #   Document( 
        #       page_content="Este es todo el texto extraído de la primera página de tu PDF. Puede incluir saltos de línea \n y otros caracteres.", 
        #       metadata={'source': 'mi_documento.pdf', 'page': 0} siempre empieza en la pagina 0
        #        )
        for doc in docs: # Para cada doc agrega a la metadata:  
            doc.metadata["file_name"] = file_path.name
            doc.metadata["file_type"] = "pdf"

    elif extension == ".txt": # si la ext es un txt
        loader = TextLoader(str(file_path), encoding="utf-8") # recibe el path del archivo en str y el encoding para evitar errores por simbolos 
        docs = loader.load()  # .load() → función principal que ejecuta la lectura del archivo TXT y devuelve el contenido estructurado en una lista de objetos Document
        for doc in docs: # agrega esta metadata en cada objeto
            doc.metadata["source"] = str(file_path)
            doc.metadata["file_name"] = file_path.name
            doc.metadata["file_type"] = "txt"

    print(f"{len(docs)} seccion/es")
    return docs


def load_directory(directory_path: Path) -> list[Document]:
    """Carga todos los archivos soportados en una carpeta"""
    if not directory_path.exists(): # En caso que el path no exista lo crea
        directory_path.mkdir(parents=True, exist_ok=True)
        print(f"Carpeta creada: {directory_path}")
        print("Agrega archivos .txt o .pdf y vuelve a ejecutar")
        return []

    all_files = []

    for ext in SUPPORTED_EXTENSION:
        # .glob → busca archivos y carpetas que coincidan con un patrón específico (como comodines) dentro de una ruta
        all_files.extend(directory_path.glob(f"*{ext}"))
        all_files.extend(directory_path.glob(f"*{ext.upper()}")) # en caso que la ext sea en mayusculas

    all_files = list(set(all_files)) # limpia la lista eliminando duplicados con set y regresandola a list

    if not all_files: # caso de que este vacio el path 
        print(f"No se encontraron archivos en: {directory_path}")
        print("Agrega archivos .txt o .pdf y vuelve a ejecutar")
        return []

    print(f"Archivos encontrados en {directory_path}: ")

    all_docs = []
    errors = []

    for file_path in sorted(all_files): # Una vez que se ordeno la lista:
        try:
            docs = load_file(file_path) # Carga el archivo con la funcion que se creo  
            all_docs.extend(docs) # Une las listas que regresa load_file
        except Exception as e: # agrega a una lista los errores por documento
            errors.append((file_path.name, str(e)))
            print(f" Error cargando {file_path.name}: {e}")

    if errors: # Si hubo errores: 
        print(
            f"\n{len(errors)} archivo(s) con errores, {len(all_docs)} documento(s) cargados"
        )
    else:
        print(
            f"\n{len(all_files)} documento(s) cargados de {len(all_docs)} seccion(es) totales"
        )
    return all_docs # regresa la lista de los documentos cargados 


def split_documents(
    docs: list[Document], # lista de Documents
    chunk_size: int = None, # tamaño del chunk
    chunk_overlap: int = None, # overlap de chunks
) -> list[Document]:
    """Divide los documentos en chunks para indexacion"""
    splitter = RecursiveCharacterTextSplitter(  # RecursiveCharacterTextSplitter es el divisor de texto y su función principal es  devolver pedazos más pequeños (llamados chunks)
        chunk_size=chunk_size or settings.CHUNK_SIZE, # numero de caracteres por chunks 
        chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP, # cantidad de caracteres a los que tiene que hacer overlap
        separators=["\n\n", "\n", ". ", " ", ""], # lista de "lugares ideales" o sugerencias de dónde cortar los chunks
        add_start_index=True # agrega la posicion del chunk dentro del doc original a los metadatos saber de donde vino cada chunk
    )

    chunks = splitter.split_documents(docs) # partiendo documentos
    # Esto es un ejemplo de lo que contiene chunks: 
    #     [
    #     # --- CHUNK 1 (Viene de la Página 1) ---
    #     Document(
    #         page_content="Este es el inicio del texto de la primera página. El TextSplitter cortó el texto justo aquí al llegar al límite de caracteres.", 
    #         metadata={'source': 'mi_documento.pdf', 'page': 0}
    #     ),
        
    #     # --- CHUNK 2 (Viene de la misma Página 1) ---
    #     Document(
    #         page_content="límite de caracteres. Y aquí continúa el resto del texto de la primera página. Nota cómo se repiten un poco las palabras anteriores por el overlap.", 
    #         metadata={'source': 'mi_documento.pdf', 'page': 0}
    #     ),

    #     # --- CHUNK 3 (Viene de la Página 2) ---
    #     Document(
    #         page_content="Aquí empieza el texto de la segunda página del PDF, que también fue cortado de acuerdo a los límites...", 
    #         metadata={'source': 'mi_documento.pdf', 'page': 1}
    #     )
    # ]
    print(f"{len(docs)} seccion/es -> {len(chunks)} chunks"
          f"(tamaño: ~{chunk_size or settings.CHUNK_SIZE} chars, "
          f"Overlap: {chunk_overlap or settings.CHUNK_OVERLAP})")

    return chunks
