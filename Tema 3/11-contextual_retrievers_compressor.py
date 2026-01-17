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
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor

vectorstore = Chroma(
    embedding_function=OpenAIEmbeddings(model="text-embedding-3-large"),
    persist_directory="C:\\workspace\\Python\\curso_langchain\\Tema 3\\7-vector_store_contratos"
)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

base_retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=base_retriever,
    llm=llm
) 

compressor = LLMChainExtractor.from_llm(llm)

compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=multi_query_retriever
)

consulta = "¿En que contratos participa María Jimenez Campos como arrendadora?."
resultados = compression_retriever.invoke(consulta)

print("top resultados mas relevantes:\n")
for i, doc in enumerate(resultados, 1):
    print("--"*50)
    print(f"CONTENIDO {i}\n: {doc.page_content}\n")
    print(f"METADATOS {i}\n: {doc.metadata}\n")