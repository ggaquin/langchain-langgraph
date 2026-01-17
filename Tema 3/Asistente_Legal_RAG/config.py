# Configuracion de modelos
EMBEDDING_MODEL = "text-embedding-3-large"
QUERY_MODEL = "gpt-4o-mini"
GENERATION_MODEL = "gpt-4o"

# Configuracion de la base de datos vectorial  
PERSIST_DIRECTORY = "C:\\workspace\\Python\\curso_langchain\\Tema 3\\7-vector_store_contratos"

# Configuracion del Retriever
# maximal marginal relevance 
SARCH_TYPE = "mmr"
# numero de documentos a recuperar
SEARCH_K = 2
# balance entre relevancia y diversidad (0 a 1), donde 0 es solo relevancia y 1 es solo diversidad
MMR_DIVERSITY_LAMBDA = 0.7
# numero de documentos a recuperar inicialmente para luego aplicar MMR
MMR_FETCH_K = 20

# Configuracion alternativa para retriever hibrido
ENABLE_HYBRID_SEARCH = True
# umbral de similitud minima para considerar un documento relevante
SIMILATITY_THRESHOLD = 0.75 