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

def fix_image_orientation(pil_image):
    """Corrige a orientação da imagem baseada nos metadados EXIF"""
    try:
        # Usa ImageOps.exif_transpose para corrigir automaticamente a orientação
        corrected_image = ImageOps.exif_transpose(pil_image)
        return corrected_image
    except Exception as e:
        st.warning(f"Não foi possível corrigir a orientação: {e}")
        return pil_image

def analyze_emotion(frame):
    """Analisa a emoção usando DeepFace"""
    try:
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        if isinstance(result, list) and len(result) > 0:
            return result[0]['dominant_emotion']
        elif isinstance(result, dict):
            return result['dominant_emotion']
        else:
            return "Neutro"
    except Exception as e:
        st.warning(f"Erro na análise emocional: {e}")
        return "Desconhecida"

def process_image(pil_image):
    """Processa a imagem e detecta faces, idade, gênero e emoção"""
    try:
        # Corrige a orientação da imagem
        pil_image = fix_image_orientation(pil_image)
        
        # Converte PIL para OpenCV
        frame_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        # Verifica se o modelo foi carregado
        if app_insight is None:
            st.error("Modelo InsightFace não foi carregado corretamente.")
            return np.array(pil_image)
        
        # Detecta faces
        faces = app_insight.get(frame_bgr)
        
        if len(faces) == 0:
            st.warning("Nenhuma face detectada na imagem.")
            return np.array(pil_image)
        
        # Analisa emoção uma vez para toda a imagem
        emotion = analyze_emotion(frame_bgr)
        emotion_translated = translate_emotion(emotion)
        
        # Dimensões da imagem para escalar fonte
        h, w, _ = frame_bgr.shape
        font_scale = max(0.6, min(w, h) / 800)
        line_height = int(30 * font_scale)
        
        # Processa cada face detectada
        for i, face in enumerate(faces):
            age = int(face.age)
            gender = translate_gender(face.gender)
            faixa = map_age_to_range(age)
            
            # Desenha retângulo ao redor da face
            bbox = face.bbox.astype(int)
            cv2.rectangle(frame_bgr, tuple(bbox[:2]), tuple(bbox[2:]), (0, 255, 0), 2)
            
            # Prepara labels
            labels = [
                f"Gênero: {gender}",
                f"Idade: {age} anos ({faixa})",
                f"Emoção: {emotion_translated}"
            ]
            
            # Posição inicial do texto
            x, y = bbox[0], bbox[1] - 15
            
            # Garante que o texto não saia da imagem
            if y < line_height * len(labels):
                y = bbox[3] + line_height
            
            # Desenha cada label
            for j, label in enumerate(labels):
                text_y = y + j * line_height
                
                # Fundo para o texto (melhor legibilidade)
                (text_width, text_height), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2
                )
                cv2.rectangle(
                    frame_bgr,
                    (x - 5, text_y - text_height - 5),
                    (x + text_width + 5, text_y + 5),
                    (0, 0, 0),
                    -1
                )
                
                # Texto
                cv2.putText(
                    frame_bgr,
                    label,
                    (x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (0, 255, 0),
                    2
                )
        
        # Converte de volta para RGB
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        
    except Exception as e:
        st.error(f"Erro no processamento da imagem: {e}")
        return np.array(pil_image)

# Interface do Streamlit
st.title("📷 Análise Facial com DeepFace + InsightFace")
st.markdown("Envie uma imagem ou use a câmera para detectar idade, gênero e emoção.")

# Opções de entrada
col1, col2 = st.columns(2)

with col1:
    image_input = st.file_uploader(
        "📁 Envie uma imagem", 
        type=["jpg", "jpeg", "png"],
        help="Formatos aceitos: JPG, JPEG, PNG"
    )

with col2:
    camera_input = st.camera_input(
        "📸 Ou tire uma foto",
        help="Use a câmera do dispositivo"
    )

# Processamento da imagem
if image_input or camera_input:
    try:
        # Carrega a imagem
        uploaded_image = Image.open(image_input or camera_input)
        
        # Mostra a imagem original
        st.subheader("🖼️ Imagem Original")
        st.image(uploaded_image, caption="Imagem carregada", use_column_width=True)
        
        # Verifica se os modelos estão disponíveis
        if app_insight is None:
            st.error("⚠️ Modelo InsightFace não está disponível. Verifique a instalação.")
        else:
            # Processa a imagem
            with st.spinner("🔍 Analisando faces, idade, gênero e emoção..."):
                result_img = process_image(uploaded_image)
            
            # Mostra o resultado
            st.subheader("✨ Resultado da Análise")
            st.image(result_img, caption="Análise completa", use_column_width=True)
            
            # Informações adicionais
            st.info("💡 **Dica:** Para melhores resultados, use imagens com faces bem iluminadas e frontais.")
            
    except Exception as e:
        st.error(f"❌ Erro ao processar a imagem: {e}")
        st.markdown("**Possíveis soluções:**")
        st.markdown("- Verifique se a imagem está em formato válido (JPG, JPEG, PNG)")
        st.markdown("- Tente com uma imagem diferente")
        st.markdown("- Reinicie a aplicação")

# Informações sobre a aplicação
with st.expander("ℹ️ Sobre esta aplicação"):
    st.markdown("""
    **Tecnologias utilizadas:**
    - **InsightFace:** Detecção de faces, idade e gênero
    - **DeepFace:** Análise de emoções
    - **OpenCV:** Processamento de imagens
    - **Streamlit:** Interface web
    
    **Funcionalidades:**
    - ✅ Correção automática de orientação de imagens
    - ✅ Detecção de múltiplas faces
    - ✅ Análise de idade, gênero e emoção
    - ✅ Interface responsiva para mobile
    
    **Nota:** A precisão pode variar dependendo da qualidade e iluminação da imagem.
    """)
