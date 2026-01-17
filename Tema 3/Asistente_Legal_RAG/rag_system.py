"""
Asistente Legal RAG - Sistema de Recuperación Augmentada por LLMs (RAG)
Este módulo implementa un sistema RAG utilizando LangChain para asistir en consultas legales.
Utiliza Chroma como base de datos vectorial, OpenAIEmbeddings para generar embeddings y 
ChatOpenAI como modelo de lenguaje.
"""

from langchain_community.vectorstores import Chroma                         # Base vectorial
from langchain_openai import OpenAIEmbeddings, ChatOpenAI                   # Embedding y LLM
from langchain_core.prompts import PromptTemplate                           # Plantillas de prompt
from langchain_core.runnables import RunnablePassthrough                    # Passthrough para prompt de usuario
from langchain_core.output_parsers import StrOutputParser                   # Parser de salida a unidad de texto
from langchain_classic.retrievers.multi_query import MultiQueryRetriever    # Recuperador multi consulta
from langchain_classic.retrievers import EnsembleRetriever                  # Recuperador MMR 
import streamlit as st                                                      # Interfaz de usuario    

from config import *                                                        # Configuraciones externas 
from prompts import *                                                       # Prompts externos 

# Decorador de caché para inicializar el sistema RAG solo una vez
@st.cache_resource 
def initialize_rag_system():
    
    """Inicializa el sistema RAG con vector store, LLM y recuperador multi consulta."""

    # VECTOR STORE CHROMA
    vectorstore = Chroma(
        embedding_function=OpenAIEmbeddings(model=EMBEDDING_MODEL),
        persist_directory=PERSIST_DIRECTORY
    )

    # MODELOS LLM
    llm_queries = ChatOpenAI(model=QUERY_MODEL, temperature=0)
    llm_generation = ChatOpenAI(model=GENERATION_MODEL, temperature=0)

    # RETRIEVER MMR (MAXIMAL MARGINAL RELEVANCE)
    base_retriever = vectorstore.as_retriever(
        search_type=SARCH_TYPE,
        search_kwargs={ 
            "k": SEARCH_K,
            "fetch_k": MMR_FETCH_K,
            "lambda_mult": MMR_DIVERSITY_LAMBDA
        }   
    )

    # RETRIEVER SIMILARITY (BASADO EN SIMILITUD DE COSENO)
    similarity_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={ "k": SEARCH_K } 
    )

    # PROMPT PRESONALIZADO PARA MULTIQUERYRETRIEVER
    multi_query_prompt = PromptTemplate.from_template(MULTI_QUERY_PROMPT)

    # MULTIQUERYRETRIEVER CON PROMPT PERSONALIZADO BASADO EN MMR
    mmr_multi_retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever,
        llm=llm_queries,
        prompt=multi_query_prompt
    )

    # ACTIVACION DEL RETRIEVER HIBRIDO SI ESTÁ INDICADO EN LA CONFIGURACIÓN
    if ENABLE_HYBRID_SEARCH:

        # RETRIEVER HIBRIDO COMBINANDO MMR Y SIMILARITY
        ensemble_retriever = EnsembleRetriever(
            retrievers=[mmr_multi_retriever, similarity_retriever],
            weights=[0.7, 0.3], # Mayor peso a MMR
            similarity_threshold=SIMILATITY_THRESHOLD
        )

        # SE UTILIZA EL RETRIEVER HIBRIDO (MMR MULTIQUERY) + (SIMILARITY)
        final_retriever = ensemble_retriever

    else:

        # SE UTILIZA SOLO EL RETRIEVER MMR MULTIQUERY
        final_retriever = mmr_multi_retriever

    prompt = PromptTemplate.from_template(RAG_TEMPLATE)

    # Función para formatear y preprocesar los documentos recuperados
    def format_docs(docs):
        formatted = []

        for i,doc in enumerate(docs, 1):

            header = f"[FRAGMENTO {i}]"                             # Encabezado del fragmento    

            if doc.metadata:

                if 'source' in doc.metadata:
                    source = doc.metadata['source'].split("\\")[-1] if '\\' in doc.metadata['source'] else doc.metadata['source']
                    header += f"  - Fuente: {source}"               # Añade fuente del fragmento si está disponible
                if 'page' in doc.metadata:
                    header += f"  - Página: {doc.metadata['page']}" # Añadir página del fragmento

            content = doc.page_content.strip()                       # Contenido del fragmento
            formatted.append(f"{header}\n{content}")                 # Añade el fragmento formateado a la lista

        return "\n\n".join(formatted)

    # La idea es crear un pipeline que genere el prompt final
    # la aproximación básica seria => prompt | llm_generation | stroutputparser()
    # 1. prompt necesita el contexto (context) y la pregunta del usuario (question)

    rag_chain = (
        {
            "context": final_retriever | format_docs, 
            "question": RunnablePassthrough()
        }
        | prompt
        | llm_generation
        | StrOutputParser()
    )

    return rag_chain, mmr_multi_retriever

def query_rag(question):

    try:
        rag_chain, retriever = initialize_rag_system()

        # Obtenemos la respuesta del sistema RAG
        response = rag_chain.invoke(question)

        # Obtenemos los documentos relevantes (opcional)
        documents = retriever.invoke(question) # o tambien retriever.invoke(question)

        # Fomatear la salida de los documentos relevantes
        docs_info = []
        for i, doc in enumerate(documents[:SEARCH_K], 1):
            doc_info = {
                "fragmento": i,
                "contenido": doc.page_content[:1000]+"..." if len(doc.page_content) > 1000 else doc.page_content,
                "fuente": doc.metadata.get("source", "No especificada").split,
                "pagina": doc.metadata.get("page", "No especificada")
            }
            docs_info.append(doc_info)

        return response, docs_info

    except Exception as e:

        error_msg = f"Error al procesar la consulta: {str(e)}"
        return error_msg, []

def get_retriever_info():
    """Obtiene información sobre la configuración del retriever."""
    info = {
        "tipo": f"SearchType={SARCH_TYPE.upper()} + mULTIQUERY " + (" + HYBRID" if ENABLE_HYBRID_SEARCH else ""), 
        "documentos": SEARCH_K,
        "diversidad": MMR_DIVERSITY_LAMBDA,
        "cabdidatos": MMR_FETCH_K,
        "umbral": SIMILATITY_THRESHOLD if ENABLE_HYBRID_SEARCH else "N/A"
    }
    return info
   