from langchain_community.document_loaders import PyMuPDFLoader

loader = PyMuPDFLoader("C:\\workspace\\Python\\curso_langchain\\Tema 3\\Archivo.pdf")

pages = loader.load()

for i, page in enumerate(pages):
    print(f"=== Page {i + 1} ===")
    print(f"Contenido : {page.page_content}")
    print(f"Metadatos: {page.metadata}")
    print(f"="*50)