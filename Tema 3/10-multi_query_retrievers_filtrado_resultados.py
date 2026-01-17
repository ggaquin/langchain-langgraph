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
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def format_docs(docs):
    return "\n\n".join(f"Fragmento de {doc.metadata.get('source', 'desconocido')}:\n{doc.page_content}" for doc in docs)

vectorstore = Chroma(
    embedding_function=OpenAIEmbeddings(model="text-embedding-3-large"),
    persist_directory="C:\\workspace\\Python\\curso_langchain\\Tema 3\\7-vector_store_contratos"
)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

base_retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 2})
retriever = MultiQueryRetriever.from_llm(
    retriever=base_retriever,
    llm=llm
) 

template = """Basándote únicamente en el contexto proporcionado, responde a la pregunta. 
Si encuentras información de diferentes personas, asegúrate de mencionar solo la que corresponde a la consulta.

Contexto:
{context}

Pregunta: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser() # El resultado se convierte a cadena de texto
)

consulta = "¿En que contratos participa María Jimenez Campos como arrendadora?."
resultados = rag_chain.invoke(consulta)
print(type(resultados))
print(resultados)
 