import warnings

warnings.filterwarnings(
    "ignore",
    message="Core Pydantic V1 functionality isn't compatible with Python 3.14 or greater",
    category=UserWarning,
    module="langchain_core._api.deprecation"
)

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
import streamlit as st


""" **************************************************************************
Chatbot personalizado con chat-gpt + plantillas (stream) + streamlit framework
************************************************************************** """

# Configuracion de la página
st.set_page_config(page_title="Chatbot Personalizado", page_icon="🤖")
st.title("🤖 Chatbot Personalizado")
st.markdown("Este es un **Chatbot de ejemplo** construido con LangChain + Streamlit. ¡Escribe un mensaje para comenzar!")

# Boton de nueva conversacion
if st.button("💬 Nueva Conversación"):
    st.session_state.messages = []
    st.rerun()

# Menu lateral
with st.sidebar:
    st.header("Configuracion")
    model_temperature = st.slider("Temperatura", 0.0, 1.0, 0.5, 0.1)
    model_name = st.selectbox("Modelo", ["gpt-3.5-turbo", "gpt-4", "gpt-4o-mini"])
    personality = st.selectbox(
        "Personalidad del Asistente",
        [
            "Útil y amigable",
            "Profesional y formal", 
            "Casual y relajado",
            "Experto técnico",
            "Creativo y divertido"
        ]
    )

system_messages = {
    "Útil y amigable": "Eres un asistente útil y amigable llamado ChatBot Pro. Responde de manera clara y concisa.",
    "Profesional y formal": "Eres un asistente profesional y formal. Proporciona respuestas precisas y bien estructuradas.",
    "Casual y relajado": "Eres un asistente casual y relajado. Habla de forma natural y amigable, como un buen amigo.",
    "Experto técnico": "Eres un asistente experto técnico. Proporciona respuestas detalladas con precisión técnica.",
    "Creativo y divertido": "Eres un asistente creativo y divertido. Usa analogías, ejemplos creativos y mantén un tono alegre."
}

# Inicializar Modelo 
chat_model = ChatOpenAI(model=model_name, temperature=model_temperature)

# Templata de Mensajes del modelo
prompt_template = ChatPromptTemplate.from_messages([
    # Mensaje de sistema
    ("system",system_messages[personality]),
    # Historial y mensaje actual
    ("human","historial de conversacion:\n{chat_history}\nPresunta actual:\n{user_input}")
])

# Inicializar historial de mensajes
if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostar historial de mensajes
for msg in st.session_state.messages:
    if isinstance(msg, SystemMessage):
        # Es prompt del sistema
        continue
    
    # Es respuesta del modelo o input del usuario
    role = "assistant" if isinstance(msg, AIMessage) else "user"

    # imprime el mensaje seguon el rol
    with st.chat_message(role):
        st.markdown(msg.content)

# Prompt de usuario
user_input = st.chat_input("Escribe te mensaje: ")

if user_input:

    # Muestra texto introducido por el usuario
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Almacenamiento en sesion del mensaje
    st.session_state.messages.append(HumanMessage(content=user_input))

    # Generar una cadena LCEL (LangChain Expression Language)
    chain = prompt_template | chat_model

    try:

        # Mostrar conntenido de la respuesta en la interfáz palabra por palabra (como los chatbots comerciales)
        with st.chat_message("assistant"):
            response_placeholder = st.empty() # Un contenedor de la interfaz vacio
            full_response = ""

            text_history = ""
            for msg in st.session_state.messages:
                if isinstance(msg,HumanMessage):
                    text_history += f"Usuario: {msg.content}.\n"
                elif isinstance(msg,AIMessage):
                    text_history += f"Asistente: {msg.content}\n"

            # Usa stream() en lugar de invoke() para no esperar la respuesta completa
            # el metodo strem retorna un objeto AIMessageChunk
            for chunk in chain.stream({"user_input": user_input, "chat_history": text_history}):
                full_response += chunk.content
                response_placeholder.markdown(full_response + "▌")
            
            response_placeholder.markdown(full_response)

            # Almacenaniento en historial de input de usuario y respuesta
            st.session_state.messages.append(HumanMessage(content=user_input))
            st.session_state.messages.append(AIMessage(content=full_response))

    except Exception as e:
        st.error(f"Error al generar respuesta: {str(e)}")
        st.info("Verifica que tu API Key de OpenAI esté configurada correctamente.")