import warnings

warnings.filterwarnings(
    "ignore",
    message="Core Pydantic V1 functionality isn't compatible with Python 3.14 or greater",
    category=UserWarning,
    module="langchain_core._api.deprecation"
)

from langchain_community.vectorstores import Chroma #requirió instalar pydantic-settings
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

loader = PyPDFDirectoryLoader("C:\\workspace\\Python\\curso_langchain\\Tema 3\\7-contratos")
documents = loader.load()

print(f"Se cargaron {len(documents)} documentos")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, 
    chunk_overlap=200
)

chunks = text_splitter.split_documents(documents)

print(f"Se crearon {len(chunks)} chunks de texto")

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=OpenAIEmbeddings(model="text-embedding-3-large"),
    persist_directory="C:\\workspace\\Python\\curso_langchain\\Tema 3\\7-vector_store_contratos"
)

consulta = "¿Donde se encuantra el local del contraro en el que participa María Jimenez Campos?"
resultados = vectorstore.similarity_search(consulta, k=2)

print("top 3 resultados mas relevantes:\n")
for i, doc in enumerate(resultados, 1):
    print("--"*50)
    print(f"CONTENIDO {i}\n: {doc.page_content}\n")
    print(f"METADATOS {i}\n: {doc.metadata}\n")