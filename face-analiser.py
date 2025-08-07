import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageOps
from deepface import DeepFace
from insightface.app import FaceAnalysis

# Configuração da página
st.set_page_config(page_title="Análise Facial", layout="centered")

# Inicializa o modelo InsightFace
@st.cache_resource
def load_insightface():
    app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=0)
    return app

app_insight = load_insightface()

# Funções auxiliares
def translate_gender(gender_id):
    return "Homem" if gender_id == 1 else "Mulher"

def map_age_to_range(age):
    if age <= 12:
        return "Criança (0-12)"
    elif age <= 17:
        return "Adolescente (13-17)"
    elif age <= 29:
        return "Jovem Adulto (18-29)"
    elif age <= 49:
        return "Adulto (30-49)"
    elif age <= 64:
        return "Meia-idade (50-64)"
    else:
        return "Idoso (65+)"

def translate_emotion(emotion):
    translation = {
        "angry": "Bravo",
        "disgust": "Nojo",
        "fear": "Medo",
        "happy": "Feliz",
        "sad": "Triste",
        "surprise": "Surpreso",
        "neutral": "Neutro",
    }
    return translation.get(emotion, emotion)

def analyze_emotion(frame):
    try:
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        return result[0]['dominant_emotion']
    except Exception as e:
        st.warning(f"Erro na análise emocional: {e}")
        return "Desconhecida"

def process_image(pil_image):
    # Corrige rotação da imagem com base nos metadados EXIF
    pil_image = ImageOps.exif_transpose(pil_image)

    frame_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    faces = app_insight.get(frame_bgr)
    emotion = analyze_emotion(frame_bgr)
    emotion_translated = translate_emotion(emotion)

    for face in faces:
        age = face.age
        gender = translate_gender(face.gender)
        faixa = map_age_to_range(age)
        bbox = face.bbox.astype(int)

        cv2.rectangle(frame_bgr, tuple(bbox[:2]), tuple(bbox[2:]), (0, 255, 0), 2)

        labels = [gender, faixa, emotion_translated]
        x, y = bbox[0], bbox[1] - 10

        h, w, _ = frame_bgr.shape
        font_scale = max(0.6, h / 800)
        line_height = int(30 * font_scale)

        for i, label in enumerate(labels):
            cv2.putText(
                frame_bgr,
                label,
                (x, y + i * line_height),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 255, 0),
                2
            )

    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

# Interface do Streamlit
st.title("📷 Análise Facial com DeepFace + InsightFace")
st.markdown("Envie uma imagem ou use a câmera para detectar idade, gênero e emoção.")

image_input = st.file_uploader("📁 Envie uma imagem", type=["jpg", "jpeg", "png"])
camera_input = st.camera_input("📸 Ou tire uma foto")

if image_input or camera_input:
    uploaded_image = Image.open(image_input or camera_input)
    uploaded_image = ImageOps.exif_transpose(uploaded_image)  # Corrige rotação da imagem
    st.image(uploaded_image, caption="Imagem Original", use_column_width=True)

    with st.spinner("Analisando..."):
        result_img = process_image(uploaded_image)
        st.image(result_img, caption="Resultado da Análise", use_column_width=True)
