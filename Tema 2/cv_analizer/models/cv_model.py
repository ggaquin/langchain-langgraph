from pydantic import BaseModel, Field

class AnalisisCV(BaseModel):

    """ 
    ----------------------------------------------------
    Modelo de datos para el análisis completo de un CV -
    ----------------------------------------------------
    """

    nombre_candidato: str = Field(description="Nombre completo del candidato estraido del CV.")
    experiencia_años: int = Field(description="Años totales de experiencia relevante.")
    habilidades_clave: list[str] = Field(description="Lista de las 5-7 habilidades del candidato mas relevantes para el puesto.")
    educacion: str = Field(description="Nivel de educación mas alto y especialización del candidato.")
    experiencia_relevante: str =Field(description="Resumen consciso de la experiencia más relevante para el puesto específico.")
    fortalezas: list[str] = Field(description="3-5 principales fertalezas del candidato basadas en su perfil.")
    areas_mejora: list[str] = Field(description="2-4 áreas donde el candidato podría desarrollarse o mejorar.")
    porcebtaje_ajuste: int = Field(description="Porcentaje de ajuete al puesto (0-100) basado en experiencia, habilidades y formación.",ge=0, le=100)
