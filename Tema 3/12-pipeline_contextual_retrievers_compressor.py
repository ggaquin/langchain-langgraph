from langchain_chroma import Chroma 
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import DocumentCompressorPipeline, LLMChainExtractor
from langchain_community.document_transformers import EmbeddingsRedundantFilter
from langchain_text_splitters import CharacterTextSplitter

persist_directory="C:\\workspace\\Python\\curso_langchain\\Tema 3\\7-vector_store_contratos"

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
embeddings=OpenAIEmbeddings(model="text-embedding-3-large")

vectorstore = Chroma(
    embedding_function=embeddings,
    persist_directory=persist_directory
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=0, separator=".")
redundant_filter = EmbeddingsRedundantFilter(embeddings=embeddings)
relevant_filter = LLMChainExtractor.from_llm(llm)

pipeline_compressor = DocumentCompressorPipeline(
    transformers=[splitter, redundant_filter, relevant_filter]
)

compression_retriever = ContextualCompressionRetriever(
    base_compressor=pipeline_compressor,
    base_retriever=retriever
)

consulta = "¿Cuáles son las cláusulas de rescisión del contrato de María Jimenez Campos como arrendadora?."
resultados = compression_retriever.invoke(consulta)

print(f"Búsqueda en: {persist_directory}")
print(f"Consulta: {consulta}\n")

if not resultados:
    print("No se encontraron fragmentos relevantes después de la compresión.")
else:
    print("top resultados mas relevantes:\n")
    for i, doc in enumerate(resultados, 1):
        print("--"*50)
        print(f"CONTENIDO {i}\n: {doc.page_content}\n")
        print(f"METADATOS {i}\n: {doc.metadata}\n")