import warnings

warnings.filterwarnings(
    "ignore",
    message="Core Pydantic V1 functionality isn't compatible with Python 3.14 or greater",
    category=UserWarning,
    module="langchain_core._api.deprecation"
)

from pydantic import BaseModel,Field
from langchain_openai import ChatOpenAI

class AnalisisTexto(BaseModel):
    resumen: str = Field(description="Resumen breve del texto")
    sentimiento: str = Field(description="Sentimiento del texto (Positivo, Negativo o Neutro)")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.6)

structured_llm = llm.with_structured_output(AnalisisTexto)
texto_prueba = "Me encantó la nueva pelicula de acción, tiene muchos efectos especiales y emocion."

resultado = structured_llm.invoke(f"Analiza el siguiente texto: {texto_prueba}")

# print(type(resultado))
# print(dir(resultado))
print(resultado.model_dump_json(indent=4))
 