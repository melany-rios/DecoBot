import streamlit as st
import os
import re

# LangChain - imports
from langchain_community.llms import HuggingFaceHub
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

# -------------------------------------------------------------
# 🎨 ESTILOS STREAMLIT
# -------------------------------------------------------------
st.set_page_config(page_title="DecoBot", page_icon="🪑", layout="centered")

st.markdown("""
<style>
.chat-bubble-user {
    background-color: #e8e8e8;
    padding: 10px;
    border-radius: 10px;
    margin-bottom: 8px;
    text-align: right;
}
.chat-bubble-bot {
    background-color: #dff2ff;
    padding: 10px;
    border-radius: 10px;
    margin-bottom: 8px;
    text-align: left;
}
</style>
""", unsafe_allow_html=True)


# -------------------------------------------------------------
# 🔧 CONFIG LLM - MANEJO SEGURO DE TOKENS
# -------------------------------------------------------------
# Intenta obtener el token de los secrets de Streamlit Cloud, luego de variables de entorno
HF_TOKEN = st.secrets.get("HUGGINGFACEHUB_API_TOKEN", os.getenv("HUGGINGFACEHUB_API_TOKEN"))

if not HF_TOKEN:
    st.error("""
    ⚠️ Token de HuggingFace no configurado.
    
    **Para desarrollo local:**
    - Crea un archivo `.streamlit/secrets.toml` con:
    ```toml
    HUGGINGFACEHUB_API_TOKEN = "tu_token_aqui"
    ```
    
    **Para producción:**
    - Configura el secret en Streamlit Cloud
    """)
    st.stop()

os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_TOKEN

try:
    llm = HuggingFaceHub(
        repo_id="tiiuae/falcon-7b-instruct",
        model_kwargs={"temperature": 0.6, "max_new_tokens": 200},
    )
    memory = ConversationBufferMemory()
    chat = ConversationChain(llm=llm, memory=memory)
except Exception as e:
    st.error(f"Error inicializando el modelo: {e}")
    st.stop()

# -------------------------------------------------------------
# 📦 CATÁLOGO
# -------------------------------------------------------------
CATALOGO = {
    "mesa": {
        "precio": 85000,
        "materiales": {
            "madera": 10000,
            "roble": 15000,
            "metal": 20000,
            "vidrio templado": 18000,
        },
    },
    "silla": {
        "precio": 40000,
        "materiales": {
            "madera": 10000,
            "metal": 12000,
            "tapizada": 20000,
        },
    },
    "cama": {
        "precio": 120000,
        "materiales": {
            "madera natural": 0,
            "roble": 25000,
            "metal": 30000,
        },
    },
    "estante": {
        "precio": 60000,
        "materiales": {
            "madera": 10000,
            "pino": 10000,
            "metal": 15000,
        },
    },
}

# ESTADOS
if "carrito" not in st.session_state:
    st.session_state.carrito = []

if "proceso" not in st.session_state:
    st.session_state.proceso = {"producto": None, "material": None, "medidas": None}

if "esperando_confirmacion" not in st.session_state:
    st.session_state.esperando_confirmacion = False


# -------------------------------------------------------------
# 🔎 DETECTOR DE INTENCIONES
# -------------------------------------------------------------
def detectar_intencion(texto):
    texto = texto.lower()

    if any(x in texto for x in ["ver", "catalogo", "lista", "productos", "mostrar"]):
        return "ver_catalogo"

    if "carrito" in texto:
        return "ver_carrito"

    if any(x in texto for x in ["finalizar", "pagar", "checkout"]):
        return "finalizar"

    if any(x in texto for x in ["salir", "chau", "adios"]):
        return "salir"

    for prod in CATALOGO.keys():
        if prod in texto:
            return "seleccionar_producto"

    if any(x in texto for x in ["madera", "roble", "metal", "vidrio", "tapizada", "pino"]):
        return "material"

    if any(x in texto for x in ["alto", "ancho", "profund", "cm", "medidas", "x"]):
        return "medidas"

    return "otro"


# -------------------------------------------------------------
# 🛠 FUNCIONES DEL CHATBOT
# -------------------------------------------------------------
def ver_catalogo():
    res = "📋 *Catálogo disponible:*\n\n"
    for p, info in CATALOGO.items():
        res += f"• {p.capitalize()} — ${info['precio']}\n"
    return res + "\n¿Qué mueble querés ver?"


def seleccionar_producto(texto):
    for prod in CATALOGO:
        if prod in texto.lower():
            st.session_state.proceso["producto"] = prod
            st.session_state.proceso["material"] = None
            st.session_state.proceso["medidas"] = None

            materiales = "\n".join(
                [f"• {m} (+${p})" for m, p in CATALOGO[prod]["materiales"].items()]
            )

            return f"🪑 Elegiste *{prod}*.\n\nMateriales disponibles:\n{materiales}\n\n¿Qué material querés?"

    return "No identifiqué ese producto. Probá con mesa, silla, cama o estante."


def seleccionar_material(texto):
    prod = st.session_state.proceso["producto"]
    materiales = CATALOGO[prod]["materiales"]

    for m in materiales:
        if m in texto.lower():
            st.session_state.proceso["material"] = m
            return "Perfecto. Ingresá las medidas (alto x ancho x profundidad)."

    return "Ese material no está disponible. Probá nuevamente."


def seleccionar_medidas(texto):
    numeros = re.findall(r"\d+", texto)

    if len(numeros) < 3:
        return "Las medidas deben tener alto, ancho y profundidad (ej: 75x120x60)."

    alto, ancho, prof = map(int, numeros[:3])
    st.session_state.proceso["medidas"] = (alto, ancho, prof)

    return confirmar_item()


def confirmar_item():
    prod = st.session_state.proceso["producto"]
    mat = st.session_state.proceso["material"]
    med = st.session_state.proceso["medidas"]

    base = CATALOGO[prod]["precio"]
    extra = CATALOGO[prod]["materiales"][mat]
    precio_final = base + extra

    st.session_state.carrito.append(
        {"producto": prod, "material": mat, "medidas": med, "precio": precio_final}
    )

    return f"🛒 Agregué tu {prod} ({mat}). Precio final: ${precio_final}\n\n¿Querés seguir comprando o finalizar?"


def ver_carrito():
    if not st.session_state.carrito:
        return "Tu carrito está vacío."

    txt = "📦 *Carrito:*\n\n"
    total = 0

    for i, item in enumerate(st.session_state.carrito, 1):
        txt += (
            f"{i}. {item['producto']} — {item['material']} — {item['medidas']} cm — ${item['precio']}\n"
        )
        total += item["precio"]

    txt += f"\n💰 Total: ${total}\n\n¿Querés confirmar la compra?"

    return txt


def finalizar_compra(texto):
    if "si" in texto.lower():
        st.session_state.carrito = []
        st.session_state.esperando_confirmacion = False
        return "🎉 ¡Compra completada! ¿Querés seguir comprando?"
    else:
        st.session_state.esperando_confirmacion = False
        return "Perfecto. ¿Querés seguir comprando?"


# -------------------------------------------------------------
# ⚙️ MOTOR PRINCIPAL
# -------------------------------------------------------------
def procesar(texto):
    if st.session_state.esperando_confirmacion:
        return finalizar_compra(texto)

    intent = detectar_intencion(texto)

    if intent == "ver_catalogo":
        return ver_catalogo()

    if intent == "ver_carrito":
        return ver_carrito()

    if intent == "seleccionar_producto":
        return seleccionar_producto(texto)

    if intent == "material" and st.session_state.proceso["producto"]:
        return seleccionar_material(texto)

    if intent == "medidas" and st.session_state.proceso["material"]:
        return seleccionar_medidas(texto)

    if intent == "finalizar":
        st.session_state.esperando_confirmacion = True
        return ver_carrito()

    if intent == "salir":
        return "¡Gracias por usar DecoBot! 👋"

    return chat.run(texto)


# -------------------------------------------------------------
# 🚀 INTERFAZ STREAMLIT
# -------------------------------------------------------------
st.title("🪑 DecoBot — Diseño de Muebles a Medida")
st.write("Tu asistente para comprar muebles personalizados.")

if "mensajes" not in st.session_state:
    st.session_state.mensajes = []

user_input = st.text_input("Escribí tu mensaje:")

if st.button("Enviar") and user_input:
    respuesta = procesar(user_input)

    st.session_state.mensajes.append(("user", user_input))
    st.session_state.mensajes.append(("bot", respuesta))

# Render del chat
for rol, msg in st.session_state.mensajes:
    clase = "chat-bubble-user" if rol == "user" else "chat-bubble-bot"
    st.markdown(f'<div class="{clase}">{msg}</div>', unsafe_allow_html=True)
