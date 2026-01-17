import warnings

warnings.filterwarnings(
    "ignore",
    message="Core Pydantic V1 functionality isn't compatible with Python 3.14 or greater",
    category=UserWarning,
    module="langchain_core._api.deprecation"
)

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

"""
Conexion a OpenAI Chat-gpt usando plantillas (invoke)
"""

# creacion del modelo
chat = ChatOpenAI(model="gpt-4o-mini",temperature=0.7)
# creación del template
prompt = ChatPromptTemplate.from_template("Saluda al usuario con su nombre.\nNombre del usuario {nombre}.\nAsistente:")
# Creacion de la cadena con LCEL (LangChain Expression Language)
chain = prompt | chat
# Consulta al modelo
resultado = chain.invoke({"nombre": "Gustavo"})
# Respuesta del modelo
print(resultado.content)