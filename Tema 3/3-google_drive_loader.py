# pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib
from langchain_community.document_loaders import GoogleDriveLoader

credentials_path = "C:\\workspace\\Python\\curso_langchain\\Tema 3\\credentials.json"  # Replace with the path to your credentials JSON file
token_path = "C:\\workspace\\Python\\curso_langchain\\Tema 3\\token.json"  # Replace with the path to your token JSON file

loeader = GoogleDriveLoader(
    folder_id="1i5P6_nW12TUaurj4XGzo_waSesVZv_rZ",  # Replace with your Google Drive folder ID
    credentials_path=credentials_path,  # Replace with the path to your credentials JSON file
    token_path=token_path,  # Replace with the path to your token JSON file
    recursive=True
 )

documents = loeader.load()
print(documents)