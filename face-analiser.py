import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageOps
from deepface import DeepFace
from insightface.app import FaceAnalysis

# Configuração da página
st.set_page_config(
    page_title="Análise Facial", 
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Inicializa o modelo InsightFace
@st.cache_resource
def load_insightface():
    try:
        app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
        app.prepare(ctx_id=0)
        return app
    except Exception as e:
        st.error(f"Erro ao carregar InsightFace: {e}")
        return None

app_insight = load_insightface()

# Funções auxiliares
def translate_gender(gender_id):
    return "Homem" if gender_id == 1 else "Mulher"

def map_age_to_range(age):
    if age <= 12:
        return "Criança"
    elif age <= 17:
        return "Adolescente"
    elif age <= 29:
        return "Jovem"
    elif age <= 49:
        return "Adulto"
    elif age <= 64:
        return "Meia-idade"
    else:
        return "Idoso"

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
    return translation.get(emotion, "Neutro")

def fix_image_orientation(pil_image):
    """Corrige a orientação da imagem baseada nos metadados EXIF"""
    try:
        corrected_image = ImageOps.exif_transpose(pil_image)
        return corrected_image
    except:
        return pil_image

def analyze_emotion(frame):
    """Analisa a emoção usando DeepFace"""
    try:
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        if isinstance(result, list) and len(result) > 0:
            return result[0]['dominant_emotion']
        elif isinstance(result, dict):
            return result['dominant_emotion']
        return "neutral"
    except:
        return "neutral"

def process_image(pil_image):
    """Processa a imagem e detecta faces"""
    try:
        # Corrige orientação
        pil_image = fix_image_orientation(pil_image)
        
        # Converte para OpenCV
        frame_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        if app_insight is None:
            return np.array(pil_image), []
        
        # Detecta faces
        faces = app_insight.get(frame_bgr)
        
        if len(faces) == 0:
            return np.array(pil_image), []
        
        # Analisa emoção
        emotion = analyze_emotion(frame_bgr)
        emotion_translated = translate_emotion(emotion)
        
        # Dimensões da imagem
        h, w, _ = frame_bgr.shape
        font_scale = max(0.5, min(w, h) / 1000)
        
        results = []
        
        # Processa cada face
        for face in faces:
            age = int(face.age)
            gender = translate_gender(face.gender)
            faixa = map_age_to_range(age)
            
            # Salva informações
            results.append({
                'age': age,
                'gender': gender,
                'age_range': faixa,
                'emotion': emotion_translated
            })
            
            # Desenha retângulo
            bbox = face.bbox.astype(int)
            cv2.rectangle(frame_bgr, tuple(bbox[:2]), tuple(bbox[2:]), (0, 255, 0), 3)
            
            # Texto simples
            label = f"{gender}, {age} anos, {emotion_translated}"
            
            # Posição do texto
            x, y = bbox[0], bbox[1] - 10
            if y < 30:
                y = bbox[3] + 25
            
            # Fundo para texto
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
            cv2.rectangle(frame_bgr, (x-2, y-text_height-5), (x+text_width+2, y+5), (0,0,0), -1)
            
            # Desenha texto
            cv2.putText(frame_bgr, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), 2)
        
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB), results
        
    except Exception as e:
        st.error(f"Erro: {e}")
        return np.array(pil_image), []

# Interface principal
st.title("📷 Análise Facial")
st.markdown("### Tire uma foto para detectar idade, gênero e emoção")

# Apenas entrada por câmera
camera_input = st.camera_input("📸 Câmera", key="camera")

if camera_input:
    try:
        # Carrega imagem
        uploaded_image = Image.open(camera_input)
        
        # Mostra imagem original em tamanho menor
        st.subheader("📸 Foto capturada")
        st.image(uploaded_image, width=300)
        
        if app_insight is None:
            st.error("⚠️ Modelo não disponível")
        else:
            # Processa
            with st.spinner("🔍 Analisando..."):
                result_img, results = process_image(uploaded_image)
            
            # Mostra resultado
            if results:
                st.subheader("✨ Resultado")
                st.image(result_img, width=400)
                
                # Mostra informações detalhadas
                st.subheader("📊 Detalhes")
                for i, info in enumerate(results, 1):
                    with st.expander(f"Pessoa {i}"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Gênero", info['gender'])
                            st.metric("Idade", f"{info['age']} anos")
                        with col2:
                            st.metric("Faixa etária", info['age_range'])
                            st.metric("Emoção", info['emotion'])
            else:
                st.warning("🔍 Nenhuma face detectada")
                st.info("💡 Dica: Posicione o rosto de frente para a câmera com boa iluminação")
                
    except Exception as e:
        st.error(f"❌ Erro ao processar: {e}")
        st.button("🔄 Tentar novamente", key="retry")

# Rodapé com informações
st.markdown("---")
with st.expander("ℹ️ Sobre"):
    st.markdown("""
    **Como usar:**
    1. Clique no botão da câmera
    2. Tire uma foto
    3. Aguarde a análise
    
    **O que detectamos:**
    - 👤 Gênero
    - 🎂 Idade 
    - 😊 Emoção
    
    **Dicas para melhores resultados:**
    - Use boa iluminação
    - Mantenha o rosto de frente
    - Evite óculos escuros ou máscaras
    """)
