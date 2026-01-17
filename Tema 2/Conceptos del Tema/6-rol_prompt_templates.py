import warnings

warnings.filterwarnings(
    "ignore",
    message="Core Pydantic V1 functionality isn't compatible with Python 3.14 or greater",
    category=UserWarning,
    module="langchain_core._api.deprecation"
)

from langchain_core.prompts import ChatPromptTemplate,SystemMessagePromptTemplate,HumanMessagePromptTemplate

plantilla_sistema = SystemMessagePromptTemplate.from_template(
    "Eres un {rol} experto en {especialidad}. Respónde de manera {tono}"
)

plantilla_humano = HumanMessagePromptTemplate.from_template(
    "Mi pregunta sobre {tema} es: {pregunta}"
)

chat_prompt = ChatPromptTemplate.from_messages([
    plantilla_sistema,
    plantilla_humano
])

mensajes = chat_prompt.format_messages(
    rol = "nutricionista",
    especialidad = "dietas veganas",
    tono =  "profesional pero accesible",
    tema = "proteinas",
    pregunta = "¿Cuales son las mejores fuentes de proteina veganas para un atleta profesional"
)

for m in mensajes:
    print(m.content)