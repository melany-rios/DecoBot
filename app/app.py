import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from langchain.memory import ConversationBufferMemory
from langchain_huggingface import HuggingFacePipeline
from langchain import LLMChain
from langchain.prompts import PromptTemplate
import torch

# -------------------------
# CONFIGURACIÓN DEL MODELO
# -------------------------

MODEL_NAME = "google/flan-t5-large"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

pipe = HuggingFacePipeline.from_model_id(
    model_id=MODEL_NAME,
    task="text2text-generation",
    model_kwargs={"temperature": 0.4, "max_length": 256}
)

# -------------------------
# MEMORIA
# -------------------------

memory = ConversationBufferMemory(
    memory_key="chat_history",
    input_key="human_input"
)

# -------------------------
# PROMPT DE SISTEMA
# -------------------------

template = """
Sos DecoBot, un asistente experto en diseño de muebles.
Tu tarea es detectar INTENCIONES del usuario y ENTIDADES (tipo_mueble, material, color, dimensiones)
y almacenarlas en memoria según aparezcan.

Utilizá la Historia de Conversación para inferir valores faltantes y
al final calcular el precio total del mueble:

PRECIOS BASE:
- mesa: 80000
- silla: 40000
- sillón: 120000
- estantería: 90000

MATERIALES:
- madera: 40000
- metal: 30000
- vidrio: 50000
- tela: 20000

DIMENSIONES:
- pequeña: -15%
- mediana: 0%
- grande: +20%

COLOR:
Sumá un plus simbólico de 2000 pesos.

AL FINAL:
Si el usuario pregunta por el total, devolvé:
- Tipo de mueble
- Material
- Color
- Dimensiones
- Cálculo detallado
- Precio final en pesos

Chat History:
{chat_history}

Usuario: {human_input}
DecoBot:
"""

prompt = PromptTemplate(
    input_variables=["chat_history", "human_input"],
    template=template
)

chain = LLMChain(
    llm=pipe,
    prompt=prompt,
    memory=memory
)

# -------------------------
# STREAMLIT UI
# -------------------------

st.set_page_config(page_title="DecoBot", page_icon="🛋️", layout="centered")

st.title("🛋️ DecoBot – Diseñador y Cotizador de Muebles con IA")

st.write("Decime qué mueble querés diseñar. Podés elegir material, color y tamaño. \
Luego pedime el precio final para generar la cotización completa.")

user_message = st.text_input("Escribí tu mensaje:")

if st.button("Enviar"):
    if user_message.strip() != "":
        response = chain.run(user_message)
        st.write("### 🤖 DecoBot:")
        st.write(response)

        st.write("### 📝 Memoria actual:")
        st.code(memory.buffer)
