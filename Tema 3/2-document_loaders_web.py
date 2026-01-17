from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader("https://techmind.ac/")

pages = loader.load()

# for i, page in enumerate(pages):
#     print(f"=== Page {i + 1} ===")
#     print(f"Contenido : {page.page_content}")
#     print(f"Metadatos: {page.metadata}")
#     print(f"="*50)

print(pages)