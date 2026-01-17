"""
Script para crear y persistir un vector store a partir de documentos PDF.
Utiliza Chroma como base de datos vectorial y OpenAIEmbeddings para generar embeddings.
"""

import warnings
warnings.filterwarnings(
    "ignore",
    message="Core Pydantic V1 functionality isn't compatible with Python 3.14 or greater",
    category=UserWarning,
    module="langchain_core._api.deprecation"
)

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


loader = PyPDFDirectoryLoader("C:\\workspace\\Python\\curso_langchain\\Tema 3\\7-contratos")
documents = loader.load()

print(f"Se cargaron {len(documents)} documentos")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=5000, 
    chunk_overlap=1000
)

chunks = text_splitter.split_documents(documents)

print(f"Se crearon {len(chunks)} chunks de texto")

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=OpenAIEmbeddings(model="text-embedding-3-large"),
    persist_directory="C:\\workspace\\Python\\curso_langchain\\Tema 3\\7-vector_store_contratos"
)