"""Modulo de grafos"""
from src.langchain_section.graphs.rag_agent import build_rag_agent

__all__ = ["build_rag_agent"] 


"""
__all__ es como un "mostrador" de tu tienda: 
en lugar de que los clientes entren y busquen en todos los estantes (todos los archivos), 
tú les muestras solo lo importante que quieres vender. 
Define qué funciones o clases son públicas y accesibles, y permite que otros importen directamente 
desde el paquete sin necesidad de conocer la estructura interna de carpetas, 
haciendo el código más limpio y fácil de mantener.
"""