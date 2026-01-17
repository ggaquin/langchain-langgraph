import warnings

warnings.filterwarnings(
    "ignore",
    message="Core Pydantic V1 functionality isn't compatible with Python 3.14 or greater",
    category=UserWarning,
    module="langchain_core._api.deprecation"
)

"""
 Uso de la vieja version de langchain_community
 from langchain_community.vectorstores import Chroma
"""
from langchain_chroma import Chroma 

from langchain_openai import OpenAIEmbeddings

vectorstore = Chroma(
    embedding_function=OpenAIEmbeddings(model="text-embedding-3-large"),
    persist_directory="C:\\workspace\\Python\\curso_langchain\\Tema 3\\7-vector_store_contratos"
)

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 2})


consulta = "¿Donde se encuantra el local del contraro en el que participa María Jimenez Campos?"
resultados = retriever.invoke(consulta)

print("top 2 resultados mas relevantes:\n")
for i, doc in enumerate(resultados, 1):
    print("--"*50)
    print(f"CONTENIDO {i}\n: {doc.page_content}\n")
    print(f"METADATOS {i}\n: {doc.metadata}\n")