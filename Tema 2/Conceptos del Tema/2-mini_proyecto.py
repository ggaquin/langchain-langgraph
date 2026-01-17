from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
import json

# configuracion del modelo
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# procesador de texto
def preprocess_text(text):
    """ Limpia el texto eliminando espacios extras y limitando la longitud """

    text = text.strip()
    text = text[:500]
    return text

# convierte la funcion preprocess_text en Runnable
preprocesador = RunnableLambda(preprocess_text)

# funcion de resumen
def generate_summary(text):
    """ Genera un resumen conciso del texto """

    prompt = f"Resume en una sola oracion {texto}"
    response = llm.invoke(prompt)
    return response.content

# funcion de analisis de sentimientos
def analyze_sentiment(text):
    """ Analiza el sentimiento del texto y devuelve un JSON estructurado """

    prompt = f"""Analiza el sentimiento del siguiente texto.
    Responde UNICAMENTE en formato json válido
    {{"sentimiento":"positivo|negativo|neutro", "razon": "justificacion breve"}} 
    Texto: {text}
    """
    response = llm.invoke(prompt)
    try:
        return json.loads(response.content)
    except json.JSONDecoderError:
        return {"sentimiento": "neutro", "razon": "Error en analisis"}

# funcion de combinacion de resumen y analisis de sentimiento
def merge_results(data):
    """ Combina los resultados de ambas ramas en un formato unificado """

    return {
        "resumen": data["resumen"],
        "sentimiento": data["sentimiento_data"]["sentimiento"],
        "razon": data["sentimiento_data"]["razon"]
    }

# funcion de procesamiento principal
def process_one(t):
    resumen = generate_summary(t)               # 1er llamado al LLM
    sentimiento_data = analyze_sentiment(t)     # 2do llamado al LLM

    return merge_results ({"resumen": resumen, "sentimiento_data": sentimiento_data})

process = RunnableLambda(process_one)

# cadena final
chain = preprocesador | process

# Prueba del sistema ----------------------------------------------------------------------------------------

# Prueba con diferentes textos
textos_prueba = [
    "Estoy encantado con mi compra. El producto llegó en perfectas condiciones y exactamente como se describe en la página. Además, el envío fue rápido y el servicio al cliente fue excepcional.",
    "Personal bastante maleducado, especialmente el anfitrión y la anfitriona. Súper engreídos y parecen encantados de rechazarte incluso cuando intentas reservar con semanas de antelación. Incluso con 2 semanas de antelación, aparentemente, no hay sitio a no ser que quieras comer a las 3 de la tarde. Actúan como si tuvieran una estrella Michelin y que su comida es suficiente para mantenerlos en el negocio. No puedo esperar a ver esta basura yuppy poco original cerrar en un año, como todos los demás en Nueva York antes que él.",
    "El clima está nublado hoy, probablemente llueva más tarde."
]

for texto in textos_prueba:
    resultado = chain.invoke(texto)
    print(f"Texto: {texto}")
    print(f"Resultado: {resultado}")
    print("-" * 50)

