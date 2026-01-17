from langchain_core.runnables import RunnableLambda

paso1 = RunnableLambda(lambda x: f"Numero {x}")

def duplica_texto(texto):
    return [texto] * 2

paso2 = RunnableLambda(duplica_texto)

cadena = paso1 | paso2

resultado = cadena.invoke(32)

print(f"Resultado: {resultado}")
