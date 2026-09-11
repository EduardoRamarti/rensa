# Rensa

Rensa es un proyecto de aprendizaje y demostración basado en LangChain, LangGraph, ChromaDB y OpenAI para construir aplicaciones con:

- Recuperación Augmentada Generativa (RAG)
- Agentes LangGraph con decisión de recuperación
- Memoria persistente de conversaciones
- Carga y segmentación de documentos en formato TXT y PDF
- Demos educativas de LCEL, streaming, mensajes con historial y vectorstore

El repositorio está organizado como una colección de scripts de ejemplo que muestran distintos niveles de integración con LLMs y bases vectoriales.

## Índice

- [Qué incluye este proyecto](#qué-incluye-este-proyecto)
- [Stack tecnológico](#stack-tecnológico)
- [Requisitos previos](#requisitos-previos)
- [Instalación](#instalación)
- [Configuración de entorno](#configuración-de-entorno)
- [Estructura del proyecto](#estructura-del-proyecto)
- [Cómo ejecutar los demos](#cómo-ejecutar-los-demos)
- [Flujo de trabajo recomendado](#flujo-de-trabajo-recomendado)
- [Detalles de arquitectura](#detalles-de-arquitectura)
- [Solución de problemas](#solución-de-problemas)
- [Notas importantes](#notas-importantes)

## Qué incluye este proyecto

Este repositorio combina varios ejemplos prácticos:

1. RAG básico con documentos locales
   - Carga de archivos `.txt` y `.pdf`
   - División en chunks con `RecursiveCharacterTextSplitter`
   - Indexación en ChromaDB
   - Recuperación semántica con embeddings OpenAI
   - Respuesta generada por un modelo de chat

2. Agente LangGraph con decisión inteligente
   - Nodo `analyze` para decidir si requiere búsqueda
   - Nodo `retrieve` para recuperar documentos relevantes
   - Nodo `generate` para producir la respuesta final
   - Estado compartido con `TypedDict`

3. Memoria persistente de conversaciones
   - SQLite como alternativa ligera
   - PostgreSQL como backend opcional
   - Historial por sesión usando `SQLChatMessageHistory`

4. Demos LCEL
   - cadenas simples
   - batch processing
   - streaming
   - `RunnablePassthrough` y `RunnableLambda`

## Stack tecnológico

- Python
- LangChain
- LangGraph
- ChromaDB
- OpenAI API (`ChatOpenAI`, `OpenAIEmbeddings`)
- SQLAlchemy
- PostgreSQL (opcional para memoria persistente)
- Docker Compose (para levantar PostgreSQL)
- `uv` como herramienta recomendada para gestión de dependencias

## Requisitos previos

- Python 3.14 (según lo indicado en `pyproject.toml`)
- `uv` recomendado
- Accesso a una API key de OpenAI
- Docker y Docker Compose (si quieres usar PostgreSQL)

## Instalación

### Opción recomendada con uv

```bash
uv sync
```

### Opción con pip

```bash
pip install -e .
```

Si usas `pip`, también puedes instalar manualmente las dependencias listadas en `pyproject.toml`.

## Configuración de entorno

Crea un archivo `.env` en la raíz del proyecto con este formato:

```dotenv
OPENAI_API_KEY=tu_api_key_aqui
OPENAI_MODEL=gpt-4o-mini
DATABASE_URL=postgresql://rensa_user:rensa_password@localhost:5432/rensa_db
```

Variables principales:

- `OPENAI_API_KEY`: clave de OpenAI requerida para LLM y embeddings
- `OPENAI_MODEL`: modelo de chat a usar (por defecto `gpt-4o-mini`)
- `DATABASE_URL`: URL de PostgreSQL para memoria de sesiones

El proyecto también usa valores por defecto en `src/langchain_section/config/settings.py`, por ejemplo:

- `EMBEDDING_MODEL = text-embedding-3-small`
- `CHUNK_SIZE = 500`
- `CHUNK_OVERLAP = 50`
- `TOP_K_RESULTS = 3`
- `SQLITE_DB_PATH = data/chat_history.db`
- `CHROMA_PATH = ./data/langchain_chroma`

## Estructura del proyecto

```text
rensa/
├── compose.yaml
├── pyproject.toml
├── README.md
├── data/
│   ├── chromadb_knowledge/
│   ├── chromadb_rag_demo/
│   ├── documents/
│   │   └── informacion_personal.txt
│   └── chat_history.db
├── src/
│   ├── langchain_section/
│   │   ├── chains/
│   │   │   ├── base.py
│   │   │   └── rag.py
│   │   ├── config/
│   │   │   └── settings.py
│   │   ├── core/
│   │   │   ├── document_loader.py
│   │   │   ├── embeddings.py
│   │   │   └── llm.py
│   │   ├── demos/
│   │   │   ├── demo_langgraph.py
│   │   │   ├── demo_lcel.py
│   │   │   ├── demo_memory.py
│   │   │   └── demo_rag.py
│   │   ├── graphs/
│   │   │   ├── __init__.py
│   │   │   ├── nodes.py
│   │   │   ├── rag_agent.py
│   │   │   └── states.py
│   │   └── memory/
│   │       ├── base.py
│   │       ├── postgresql_memory.py
│   │       └── sqlite_memory.py
│   └── rensa/
│       └── __init__.py
└── .env
```

## Cómo ejecutar los demos

Desde la raíz del proyecto, puedes correr los ejemplos con `python -m` o con `uv run`.

### 1) Demo RAG básico

Este demo carga documentos desde `data/documents`, los indexa en ChromaDB y permite conversar con ellos.

```bash
python -m src.langchain_section.demos.demo_rag
```

O con uv:

```bash
uv run python -m src.langchain_section.demos.demo_rag
```

Comandos interactivos disponibles en este script:

- `archivos`: muestra los archivos disponibles
- `chunks`: muestra cuántos chunks existen en la colección
- `reindexar`: vuelve a indexar documentos desde cero
- `salir`: termina la sesión

### 2) Demo LangGraph con agente RAG

Este es el ejemplo más completo del proyecto. Incluye:

- carga o creación del vectorstore
- fallback a SQLite si PostgreSQL no está disponible
- selección de sesión
- historial persistente
- flujo LangGraph con nodo de análisis, recuperación y generación

```bash
python -m src.langchain_section.demos.demo_langgraph
```

Comandos disponibles:

- `sesion`: muestra el ID actual
- `historial`: muestra mensajes recientes
- `limpiar`: borra el historial de la sesión
- `salir`: termina la conversación

### 3) Demo de memoria persistente

Este ejemplo se enfoca en la capa de memoria y cómo se usa `RunnableWithMessageHistory` con diferentes backends.

```bash
python -m src.langchain_section.demos.demo_memory
```

Te da la opción de elegir:

- SQLite
- PostgreSQL

### 4) Demo LCEL

Demuestra conceptos fundamentales de LangChain Expression Language:

- cadenas simples
- batch
- stream
- `RunnablePassthrough`

```bash
python -m src.langchain_section.demos.demo_lcel
```

## Flujo de trabajo recomendado

### Caso 1: Probar el RAG con documentos locales

1. Coloca tus archivos `.txt` o `.pdf` en `data/documents/`
2. Configura tu `.env`
3. Ejecuta `demo_rag.py`
4. Usa los comandos interactivos para indexar, consultar y reindexar

### Caso 2: Probar el agente completo con historial

1. Configura `OPENAI_API_KEY`
2. Levanta PostgreSQL opcionalmente con Docker
3. Ejecuta `demo_langgraph.py`
4. Selecciona o crea una sesión
5. Haz preguntas y revisa respuestas con fuentes consultadas

### Caso 3: Probar memoria independiente

1. Ejecuta `demo_memory.py`
2. Elige SQLite o PostgreSQL
3. Prueba historial, limpieza de sesión y listado de sesiones

## Detalles de arquitectura

### 1. Capa de configuración

`src/langchain_section/config/settings.py` centraliza la configuración:

- modelo de chat
- modelo de embeddings
- rutas de almacenamiento ChromaDB y SQLite
- tamaño de chunks y overlap
- número de resultados devueltos por el retriever

### 2. Carga y segmentación

`src/langchain_section/core/document_loader.py`:

- detecta archivos `.txt` y `.pdf`
- usa `TextLoader` o `PyPDFLoader`
- agrega metadata útil
- corta documentos en chunks con `RecursiveCharacterTextSplitter`

### 3. Embeddings y vectorstore

`src/langchain_section/core/embeddings.py`:

- crea o reutiliza una colección ChromaDB
- usa `OpenAIEmbeddings`
- persiste el vectorstore en disco

### 4. LLM centralizado

`src/langchain_section/core/llm.py`:

- define un cliente `ChatOpenAI`
- expone `get_llm()` y `get_embeddings()`
- usa caché con `lru_cache`

### 5. Memoria

`src/langchain_section/memory/sqlite_memory.py` y `src/langchain_section/memory/postgresql_memory.py`:

- implementan una interfaz común (`BaseMemoryBackend`)
- guardan historial de conversaciones por sesión
- permiten listar sesiones existentes y limpiar historial

### 6. Grafo LangGraph

`src/langchain_section/graphs/rag_agent.py`:

- crea un `StateGraph`
- conecta los nodos:
  - `analyze`
  - `retrieve`
  - `generate`
- define la ruta condicional entre análisis y recuperación

Los nodos principales están en:

- `src/langchain_section/graphs/nodes.py`
- `src/langchain_section/graphs/states.py`

## Solución de problemas

### Error de OpenAI API key

Si aparece un error relacionado con `OPENAI_API_KEY`, revisa que el archivo `.env` exista y que la variable esté correctamente configurada.

### No se encuentran documentos

Asegúrate de tener archivos `.txt` o `.pdf` dentro de `data/documents/`.

### PostgreSQL no responde

Puedes levantar el servicio con:

```bash
docker compose up -d db
```

Si no quieres usar PostgreSQL, el proyecto usa SQLite automáticamente como fallback en `demo_langgraph.py`.

### ChromaDB no se actualiza

En algunos casos puede ser útil borrar manualmente las carpetas de ChromaDB y volver a indexar:

```text
data/chromadb_knowledge/
data/chromadb_rag_demo/
```

### Error al ejecutar módulos

La configuración del proyecto usa imports desde `src`, así que ejecuta todos los scripts desde la raíz del repositorio.

## Notas importantes

- El proyecto incluye varios ejemplos educativos, no un único producto final.
- La demostración más completa es `demo_langgraph.py`.
- La CLI declarada en `pyproject.toml` (`rensa = "rensa:main"`) no está implementada todavía en `src/rensa/__init__.py`, por lo que por ahora se recomienda ejecutar los módulos directamente.
- El archivo `.env` se carga automáticamente por `load_dotenv()` en `src/langchain_section/config/settings.py`.

## Sugerencias para ampliar el proyecto

- Añadir una API REST o FastAPI para exponer el agente
- Añadir autenticación para sesiones por usuario
- Agregar soporte para más tipos de documentos
- Mejorar prompts y decisiones del agente
- Añadir pruebas automatizadas
- Añadir una interfaz web con Streamlit o Gradio

## Licencia

Este README describe el proyecto actual sin declarar una licencia específica. Si deseas publicar el repositorio públicamente, revisa si necesitas añadir una licencia apropiada.
