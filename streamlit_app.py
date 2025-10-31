import streamlit as st
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import io
import json
import os
from datetime import datetime
import base64
import requests
import folium  
from streamlit_folium import st_folium  

# Configuration de la page
st.set_page_config(page_title="Messagerie", page_icon="📸", layout="centered")

@st.cache_resource
def load_opencv():
    """Charge OpenCV et NumPy avec cache persistant"""
    try:
        import cv2
        import numpy as np
        return cv2, np, True
    except Exception as e:
        st.error(f"Erreur chargement OpenCV: {e}")
        return None, None, False

@st.cache_resource
def load_mediapipe():
    """Charge MediaPipe avec cache persistant"""
    try:
        import mediapipe as mp
        return mp, True
    except Exception as e:
        st.error(f"Erreur chargement MediaPipe: {e}")
        return None, False

# Charger les bibliothèques
cv2, np, CV2_AVAILABLE = load_opencv()
mp, MEDIAPIPE_AVAILABLE = load_mediapipe()

# CSS pour un design moderne et élégant
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    .main .block-container {
        padding: 2rem 1rem;
        max-width: 900px;
    }
    
    h1 {
        color: white !important;
        text-align: center;
        font-weight: 300 !important;
        letter-spacing: 2px;
        text-shadow: 0 2px 10px rgba(0,0,0,0.2);
        margin-bottom: 2rem !important;
    }
    
    h2 {
        color: white !important;
        font-weight: 400 !important;
        font-size: 1.3rem !important;
        margin-top: 2rem !important;
    }
    
    .message-container-admin {
        display: flex;
        justify-content: flex-end;
        margin-bottom: 20px;
        animation: slideInRight 0.3s ease;
    }
    
    .message-container-user {
        display: flex;
        justify-content: flex-start;
        margin-bottom: 20px;
        animation: slideInLeft 0.3s ease;
    }
    
    .message-content {
        max-width: 70%;
        background: white;
        border-radius: 20px;
        padding: 15px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
    }
    
    @keyframes slideInRight {
        from { opacity: 0; transform: translateX(30px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    @keyframes slideInLeft {
        from { opacity: 0; transform: translateX(-30px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 25px !important;
        padding: 0.6rem 2rem !important;
        font-weight: 500 !important;
        letter-spacing: 0.5px !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4) !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6) !important;
    }
    
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%) !important;
        box-shadow: 0 4px 15px rgba(245, 87, 108, 0.4) !important;
    }
    
    .stTextInput > div > div > input {
        background: rgba(255, 255, 255, 0.95) !important;
        border: 2px solid rgba(255, 255, 255, 0.3) !important;
        border-radius: 15px !important;
        padding: 0.8rem 1rem !important;
        color: #333 !important;
        font-size: 1rem !important;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #f5576c !important;
        box-shadow: 0 0 0 3px rgba(245, 87, 108, 0.2) !important;
    }
    
    .stCameraInput > div {
        background: rgba(255, 255, 255, 0.95) !important;
        border-radius: 20px !important;
        padding: 1rem !important;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1) !important;
    }
    
    .stDownloadButton > button {
        background: rgba(255, 255, 255, 0.9) !important;
        color: #667eea !important;
        border-radius: 10px !important;
        padding: 0.4rem 0.8rem !important;
        font-size: 1.2rem !important;
        border: 2px solid #667eea !important;
    }
    
    .stAlert {
        background: rgba(255, 255, 255, 0.95) !important;
        border-radius: 15px !important;
    }
    
    hr {
        margin: 1rem 0 !important;
        border-color: rgba(255, 255, 255, 0.2) !important;
    }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%) !important;
    }
    
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: white !important;
    }
    
    img {
        border-radius: 15px !important;
    }
            
    /* Style pour les onglets */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 0.5rem;
    }

    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: white !important;
        border-radius: 10px;
        padding: 0.5rem 1.5rem;
        font-weight: 500;
    }

    .stTabs [aria-selected="true"] {
        background: rgba(255, 255, 255, 0.2) !important;
    }

    /* Style pour la carte */
    .folium-map {
        border-radius: 20px !important;
        overflow: hidden;
        box-shadow: 0 4px 20px rgba(0,0,0,0.2) !important;
    }
</style>
""", unsafe_allow_html=True)

# Configuration GitHub
GITHUB_TOKEN = st.secrets.get("GITHUB_TOKEN", "") if hasattr(st, 'secrets') else ""
GITHUB_REPO = st.secrets.get("GITHUB_REPO", "") if hasattr(st, 'secrets') else ""
GITHUB_BRANCH = "main"
DATA_FILE = "messages_data.json"
TELEGRAM_BOT_TOKEN = st.secrets.get("TELEGRAM_BOT_TOKEN", "") if hasattr(st, 'secrets') else ""
TELEGRAM_GROUP_CHAT_ID = st.secrets.get("TELEGRAM_GROUP_CHAT_ID", "") if hasattr(st, 'secrets') else ""

def github_update_file(file_path, content, sha=None, message="Update data"):
    """Met à jour un fichier sur GitHub"""
    if not GITHUB_TOKEN or not GITHUB_REPO:
        return False
    
    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{file_path}"
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json"
    }
    
    data = {
        "message": message,
        "content": base64.b64encode(content.encode('utf-8')).decode('utf-8'),
        "branch": GITHUB_BRANCH
    }
    
    if sha:
        data["sha"] = sha
    
    try:
        response = requests.put(url, headers=headers, json=data, timeout=10)
        return response.status_code in [200, 201]
    except Exception as e:
        st.error(f"Erreur GitHub UPDATE: {str(e)}")
        return False

def github_get_file(file_path):
    """Récupère un fichier depuis GitHub via l'API Blob"""
    if not GITHUB_TOKEN or not GITHUB_REPO:
        return None
    
    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{file_path}"
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json"
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code != 200:
            return None
        
        file_info = response.json()
        sha = file_info.get('sha')
        size = file_info.get('size', 0)
        
        if size < 900000 and 'content' in file_info and file_info['content']:
            encoded_content = file_info['content'].replace('\n', '').replace('\r', '')
            decoded_content = base64.b64decode(encoded_content).decode('utf-8')
            return {
                'content': decoded_content,
                'sha': sha
            }
        
        blob_url = f"https://api.github.com/repos/{GITHUB_REPO}/git/blobs/{sha}"
        blob_response = requests.get(blob_url, headers=headers, timeout=30)
        
        if blob_response.status_code != 200:
            return None
        
        blob_data = blob_response.json()
        
        if 'content' not in blob_data:
            return None
        
        encoded_content = blob_data['content'].replace('\n', '').replace('\r', '')
        decoded_content = base64.b64decode(encoded_content).decode('utf-8')
        
        return {
            'content': decoded_content,
            'sha': sha
        }
        
    except Exception as e:
        return None

def create_map_view():
    """Crée une vue carte avec tous les messages géolocalisés"""
    st.header("🗺️ Carte des photos")
    
    geolocated_messages = [msg for msg in st.session_state.messages if msg.get('location')]
    
    if not geolocated_messages:
        st.info("Aucune photo géolocalisée pour le moment. Les futures photos avec localisation apparaîtront ici !")
        return
    
    latitudes = [msg['location']['latitude'] for msg in geolocated_messages]
    longitudes = [msg['location']['longitude'] for msg in geolocated_messages]
    center_lat = sum(latitudes) / len(latitudes)
    center_lon = sum(longitudes) / len(longitudes)
    
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=3,
        tiles='OpenStreetMap'
    )
    
    for msg in geolocated_messages:
        if not msg.get('location'):
            continue
        
        lat = msg['location']['latitude']
        lon = msg['location']['longitude']
        
        color = '#667eea' if msg['sender'] == 'admin' else '#f5576c'
        icon_color = 'blue' if msg['sender'] == 'admin' else 'pink'
        
        img_bytes = io.BytesIO()
        thumb = msg['image_with_text'].copy()
        thumb.thumbnail((300, 300))
        thumb.save(img_bytes, format='PNG')
        img_b64 = base64.b64encode(img_bytes.getvalue()).decode()
        
        timestamp = datetime.fromisoformat(msg['timestamp']).strftime('%d/%m/%Y %H:%M')
        sender_name = "Le cousin" if msg['sender'] == 'admin' else "La cousine"
        
        popup_html = f"""
        <div style="width: 300px; text-align: center;">
            <h4 style="margin: 5px 0; color: {color};">{sender_name}</h4>
            <p style="margin: 5px 0; font-size: 12px; color: #666;">{timestamp}</p>
            <img src="data:image/png;base64,{img_b64}" style="width: 100%; border-radius: 10px; margin-top: 10px;">
            {f'<p style="margin-top: 10px; font-style: italic;">"{msg["text"]}"</p>' if msg.get('text') else ''}
        </div>
        """
        
        folium.Marker(
            location=[lat, lon],
            popup=folium.Popup(popup_html, max_width=320),
            icon=folium.Icon(color=icon_color, icon='camera', prefix='fa'),
            tooltip=f"{sender_name} - {timestamp}"
        ).add_to(m)
    
    st_folium(m, width=None, height=600)
    
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("📍 Total localisations", len(geolocated_messages))
    
    with col2:
        admin_count = len([m for m in geolocated_messages if m['sender'] == 'admin'])
        st.metric("🔵 Le cousin", admin_count)
    
    with col3:
        user_count = len([m for m in geolocated_messages if m['sender'] == 'user'])
        st.metric("🔴 La cousine", user_count)

def load_counters():
    """Charge les compteurs depuis GitHub"""
    file_data = github_get_file(DATA_FILE)
    
    if file_data:
        try:
            data = json.loads(file_data['content'])
            return data.get('counters', {"admin": 0, "user": 0})
        except:
            pass
    return {"admin": 0, "user": 0}

def load_messages():
    """Charge les messages depuis GitHub"""
    try:
        file_data = github_get_file(DATA_FILE)
        
        if not file_data:
            return []
        
        content = file_data['content']
        
        if not content or content.strip() == "":
            return []
        
        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            return []
        
        messages_data = data.get('messages', [])
        
        messages = []
        for idx, msg in enumerate(messages_data):
            try:
                if 'image_with_text_b64' in msg:
                    img_data = base64.b64decode(msg['image_with_text_b64'])
                    msg['image_with_text'] = Image.open(io.BytesIO(img_data))
                
                if 'original_image_b64' in msg:
                    img_data = base64.b64decode(msg['original_image_b64'])
                    msg['original_image'] = Image.open(io.BytesIO(img_data))
                
                if 'location' not in msg:
                    msg['location'] = None

                messages.append(msg)
                
            except Exception as e:
                continue
        
        return messages
        
    except Exception as e:
        return []

def save_messages():
    """Sauvegarde les messages sur GitHub"""
    try:
        messages_to_save = []
        for msg in st.session_state.messages:
            msg_copy = {
                'timestamp': msg['timestamp'],
                'text': msg['text'],
                'sender': msg['sender'],
                'id': msg['id'],
                'location': msg.get('location')  # ✅ AJOUT
            }
            
            if 'image_with_text' in msg:
                img_bytes = io.BytesIO()
                msg['image_with_text'].save(img_bytes, format='PNG', optimize=True, compress_level=6)
                msg_copy['image_with_text_b64'] = base64.b64encode(img_bytes.getvalue()).decode()
            
            if 'original_image' in msg:
                img_bytes = io.BytesIO()
                msg['original_image'].save(img_bytes, format='PNG', optimize=True, compress_level=6)
                msg_copy['original_image_b64'] = base64.b64encode(img_bytes.getvalue()).decode()
            
            messages_to_save.append(msg_copy)
        
        data = {
            'messages': messages_to_save,
            'passwords': st.session_state.user_passwords,
            'counters': st.session_state.counters
        }
        
        file_data = github_get_file(DATA_FILE)
        sha = file_data['sha'] if file_data else None
        
        return github_update_file(DATA_FILE, json.dumps(data, indent=2), sha, "Update messages")
        
    except Exception as e:
        st.error(f"Erreur sauvegarde: {str(e)}")
        return False

def load_passwords():
    """Charge les mots de passe depuis GitHub"""
    file_data = github_get_file(DATA_FILE)
    
    if file_data:
        try:
            data = json.loads(file_data['content'])
            return data.get('passwords', ["crush"])
        except:
            pass
    return ["crush"]

def send_telegram_notification(sender, has_text):
    """Envoie une notification Telegram au groupe"""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_GROUP_CHAT_ID:
        return False
    
    try:
        import random
        
        messages_admin = [
            "📸 Nouveau message de ta cousine préférée !",
            "✨ une beauté absolue vient de poster une photo !",
            "🎉 Regarde ! une vision de paradie vient d'apparaitre !",
            "💌 Tu as reçu un message de la femme de ta vie !",
            "🔔 Ding dong ! tu as enfin reçu ce que tu attendais tout ce temps !",
            "📬 Viens voir cette pepite qui vient d'arriver !",
            "🌟 une beauté absolue pense à toi !",
            "💕 Message tout frais de ta cousine préférée !",
            "🎨 une beauté absolue partage un moment avec toi !",
            "🚀 Un message arrive en direction de ton coeur !",
            "Arrete d'esperer c'est ta cousine ! il y aura rien de plus !",
            "Attend au moins la fin de ton cours pour voir ce message",
            "Assis toi pour pas tomber par terre face a une tel beautée",
            "C'est bon tu vas passer une bonne journnée grace à ce message",
            "Baisse ta luminositée, tu vas être éblouie",
        ]
        
        messages_user = [
            "📸 Nouveau message de ton homme !",
            "✨ un homme grandiose vient de poster une photo !",
            "🎉 Regarde ! un être malicieux a envoyé quelque chose !",
            "💌 Tu as reçu un message rempli d'affection !",
            "🔔 Ding dong ! C'est encore et toujours moi !",
            "📬 Nouveau dans la boîte : tu l'attendais et il est enfin là !",
            "🌟 un homme grandiose pense (encore et toujours) à toi !",
            "💕 Message tout frais de ton plus grand fan !",
            "🎨 ton cousin PREFERE partage un instant de sa vie avec toi !",
            "🚀 Message en approche de ton future mari !",
            "Ton impatience de voir ce message est palpable",
            "On espère que ta famille ne tombera pas sur ce message",
            "Si tu réagie comme ça a chaque notif tes potes vont se poser des questions",
            "C'est pour toi bébou... il a encore pensé a toi !",
            "Viens voir ce corps d'apollon",
        ]
        
        if sender == "admin":
            base_message = random.choice(messages_user)
        else:
            base_message = random.choice(messages_admin)
        
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        response = requests.post(url, json={
            "chat_id": TELEGRAM_GROUP_CHAT_ID,
            "text": base_message
        }, timeout=5)
        
        return response.status_code == 200
    except:
        return False

def reload_heavy_libraries():
    """Force le rechargement des bibliothèques lourdes"""
    global CV2_AVAILABLE, MEDIAPIPE_AVAILABLE, cv2, mp, np
    
    try:
        import importlib
        import sys
        
        if 'cv2' in sys.modules:
            del sys.modules['cv2']
        import cv2
        import numpy as np
        CV2_AVAILABLE = True
        
        if 'mediapipe' in sys.modules:
            del sys.modules['mediapipe']
        import mediapipe as mp
        MEDIAPIPE_AVAILABLE = True
        
        return True
    except Exception as e:
        return False

# Initialisation des variables de session
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'is_admin' not in st.session_state:
    st.session_state.is_admin = False
if 'libs_checked' not in st.session_state:
    if not CV2_AVAILABLE or not MEDIAPIPE_AVAILABLE:
        reload_heavy_libraries()
    st.session_state.libs_checked = True
if 'messages' not in st.session_state:
    st.session_state.messages = load_messages()
if 'user_passwords' not in st.session_state:
    st.session_state.user_passwords = load_passwords()
if 'last_message_count' not in st.session_state:
    st.session_state.last_message_count = 0
if 'current_user' not in st.session_state:
    st.session_state.current_user = None
if 'notification_enabled' not in st.session_state:
    st.session_state.notification_enabled = False
if 'counters' not in st.session_state:
    st.session_state.counters = load_counters()
if 'last_gps_location' not in st.session_state:
    st.session_state.last_gps_location = None
if 'gps_auto_enabled' not in st.session_state:
    st.session_state.gps_auto_enabled = True

def verify_human_body_simple(image):
    """Vérifie la présence d'un corps humain avec OpenCV + MediaPipe"""
    if not CV2_AVAILABLE:
        return True
    
    try:
        img_array = np.array(image)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        detections = []
        
        cascades = [
            ('haarcascade_frontalface_default.xml', 5, 30),
            ('haarcascade_profileface.xml', 5, 30),
            ('haarcascade_fullbody.xml', 3, 50),
            ('haarcascade_upperbody.xml', 3, 50),
            ('haarcascade_lowerbody.xml', 3, 30),
            ('haarcascade_eye.xml', 5, 20),
        ]
        
        for cascade_name, min_neighbors, min_size in cascades:
            try:
                cascade = cv2.CascadeClassifier(cv2.data.haarcascades + cascade_name)
                objects = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=min_neighbors, minSize=(min_size, min_size))
                if len(objects) > 0:
                    detections.append(cascade_name)
            except:
                pass
        
        if MEDIAPIPE_AVAILABLE:
            try:
                mp_hands = mp.solutions.hands
                with mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5) as hands:
                    results = hands.process(img_array)
                    if results.multi_hand_landmarks:
                        detections.append('hands')
            except:
                pass
            
            try:
                mp_pose = mp.solutions.pose
                with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
                    results = pose.process(img_array)
                    if results.pose_landmarks:
                        detections.append('pose')
            except:
                pass
        
        has_body_part = len(detections) > 0
        
        if not has_body_part:
            st.error("❌ Aucune partie du corps détectée")
        
        return has_body_part
        
    except Exception as e:
        st.error(f"Erreur détection: {str(e)}")
        return True

def add_text_to_image(image, text):
    """Ajoute du texte stylé sur l'image SANS perte de qualité"""
    if not text or text.strip() == "":
        return image
    
    img_copy = image.copy()
    
    if img_copy.mode != 'RGBA':
        img_copy = img_copy.convert('RGBA')
    
    txt_layer = Image.new('RGBA', img_copy.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(txt_layer)
    
    width, height = img_copy.size
    font_size = max(int(height * 0.04), 20)
    
    font = None
    font_paths = [
        "C:/Windows/Fonts/seguiemj.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "/System/Library/Fonts/Apple Color Emoji.ttc",
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    
    for font_path in font_paths:
        try:
            if os.path.exists(font_path):
                font = ImageFont.truetype(font_path, font_size)
                break
        except:
            continue
    
    if font is None:
        font = ImageFont.load_default()
    
    max_width = width * 0.85
    lines = []
    words = text.split()
    current_line = ""
    
    for word in words:
        test_line = current_line + " " + word if current_line else word
        try:
            bbox = draw.textbbox((0, 0), test_line, font=font)
            test_width = bbox[2] - bbox[0]
        except:
            test_width = len(test_line) * (font_size // 2)
        
        if test_width <= max_width:
            current_line = test_line
        else:
            if current_line:
                lines.append(current_line)
            current_line = word
    
    if current_line:
        lines.append(current_line)
    
    final_lines = []
    for line in lines:
        try:
            bbox = draw.textbbox((0, 0), line, font=font)
            line_width = bbox[2] - bbox[0]
        except:
            line_width = len(line) * (font_size // 2)
        
        if line_width > max_width:
            chars_per_line = int(len(line) * (max_width / line_width))
            for i in range(0, len(line), chars_per_line):
                final_lines.append(line[i:i+chars_per_line])
        else:
            final_lines.append(line)
    
    line_height = font_size * 1.4
    total_text_height = len(final_lines) * line_height
    
    if len(final_lines) > 5:
        font_size = max(int(height * 0.03), 16)
        try:
            for font_path in font_paths:
                if os.path.exists(font_path):
                    font = ImageFont.truetype(font_path, font_size)
                    break
        except:
            pass
        line_height = font_size * 1.4
        total_text_height = len(final_lines) * line_height
    
    padding = int(font_size * 0.8)
    
    max_line_width = 0
    for line in final_lines:
        try:
            bbox = draw.textbbox((0, 0), line, font=font)
            line_width = bbox[2] - bbox[0]
        except:
            line_width = len(line) * (font_size // 2)
        max_line_width = max(max_line_width, line_width)
    
    rect_width = max_line_width + padding * 2
    rect_height = total_text_height + padding * 2
    x = (width - rect_width) // 2
    y = height - rect_height - padding * 2
    
    rect = [x, y, x + rect_width, y + rect_height]
    radius = padding
    
    shadow_offset = 4
    shadow = Image.new('RGBA', img_copy.size, (0, 0, 0, 0))
    shadow_draw = ImageDraw.Draw(shadow)
    shadow_draw.rounded_rectangle(
        [r + shadow_offset for r in rect], 
        radius=radius, 
        fill=(0, 0, 0, 120)
    )
    shadow = shadow.filter(ImageFilter.GaussianBlur(6))
    txt_layer = Image.alpha_composite(txt_layer, shadow)
    draw = ImageDraw.Draw(txt_layer)
    
    draw.rounded_rectangle(rect, radius=radius, fill=(20, 20, 20, 230))
    draw.rounded_rectangle(rect, radius=radius, outline=(255, 255, 255, 180), width=2)
    
    current_y = y + padding
    for line in final_lines:
        try:
            bbox = draw.textbbox((0, 0), line, font=font)
            line_width = bbox[2] - bbox[0]
        except:
            line_width = len(line) * (font_size // 2)
        
        line_x = x + (rect_width - line_width) // 2
        
        for offset in [(1, 1), (-1, 1), (1, -1), (-1, -1), (0, 2), (2, 0)]:
            try:
                draw.text(
                    (line_x + offset[0], current_y + offset[1]), 
                    line, 
                    font=font, 
                    fill=(0, 0, 0, 200), 
                    embedded_color=True
                )
            except:
                draw.text(
                    (line_x + offset[0], current_y + offset[1]), 
                    line, 
                    font=font, 
                    fill=(0, 0, 0, 200)
                )
        
        try:
            draw.text(
                (line_x, current_y), 
                line, 
                font=font, 
                fill=(255, 255, 255, 255), 
                embedded_color=True
            )
        except:
            draw.text(
                (line_x, current_y), 
                line, 
                font=font, 
                fill=(255, 255, 255, 255)
            )
        
        current_y += line_height
    
    img_copy = Image.alpha_composite(img_copy, txt_layer)
    img_copy = img_copy.convert('RGB')
    
    return img_copy

def increment_counter(user):
    """Incrémente le compteur de l'utilisateur avec animation"""
    st.session_state.counters[user] = st.session_state.counters.get(user, 0) + 1
    counter_value = st.session_state.counters[user]
    
    st.balloons()
    
    if counter_value % 10 == 0:
        st.snow()
        st.success(f"🎉 **{counter_value} messages** ! Incroyable ! 🎉")
    elif counter_value % 5 == 0:
        st.success(f"🌟 **{counter_value} messages** ! Continue comme ça ! 🌟")

def save_message(image, text, original_image, sender, location=None):
    """Sauvegarde un message avec coordonnées GPS optionnelles"""
    message = {
        'timestamp': datetime.now().isoformat(),
        'text': text,
        'image_with_text': image,
        'original_image': original_image,
        'sender': sender,
        'id': int(datetime.now().timestamp() * 1000),
        'location': location
    }
    st.session_state.messages.append(message)
    increment_counter(sender)
    save_messages()
    send_telegram_notification(sender, bool(text))

def delete_message(message_id):
    """Supprime un message et décrémente le compteur"""
    message_to_delete = None
    for msg in st.session_state.messages:
        if msg['id'] == message_id:
            message_to_delete = msg
            break
    
    st.session_state.messages = [msg for msg in st.session_state.messages if msg['id'] != message_id]
    
    if message_to_delete:
        sender = message_to_delete['sender']
        if sender in st.session_state.counters and st.session_state.counters[sender] > 0:
            st.session_state.counters[sender] -= 1
    
    save_messages()

def check_new_messages():
    """Vérifie les nouveaux messages"""
    current_count = len(st.session_state.messages)
    
    if current_count > st.session_state.last_message_count:
        last_msg = st.session_state.messages[-1]
        if last_msg['sender'] != st.session_state.current_user:
            st.toast("📬 Nouveau message !", icon="📬")
    
    st.session_state.last_message_count = current_count

def display_counters():
    """Affiche les compteurs avec style"""
    st.markdown("""
    <style>
        .counter-container {
            background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%);
            border-radius: 20px;
            padding: 1.5rem;
            margin: 1rem 0;
            border: 2px solid rgba(255,255,255,0.2);
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        }
        .counter-title {
            color: white;
            font-size: 1.1rem;
            font-weight: 500;
            margin-bottom: 0.5rem;
            text-align: center;
        }
        .counter-value {
            color: #f5576c;
            font-size: 3rem;
            font-weight: 700;
            text-align: center;
            text-shadow: 0 2px 10px rgba(245, 87, 108, 0.5);
            animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0%, 100% { transform: scale(1); }
            50% { transform: scale(1.05); }
        }
        .counter-label {
            color: rgba(255,255,255,0.8);
            text-align: center;
            font-size: 0.9rem;
            margin-top: 0.3rem;
        }
    </style>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        admin_count = st.session_state.counters.get("admin", 0)
        st.markdown(f"""
        <div class="counter-container">
            <div class="counter-title">Le grand, beau, magnifique, merveilleux, grandiose, splendide, humble cousin</div>
            <div class="counter-value">{admin_count}</div>
            <div class="counter-label">messages envoyés</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        user_count = st.session_state.counters.get("user", 0)
        st.markdown(f"""
        <div class="counter-container">
            <div class="counter-title">La cousine 😘</div>
            <div class="counter-value">{user_count}</div>
            <div class="counter-label">messages envoyés</div>
        </div>
        """, unsafe_allow_html=True)

def login_page():
    """Page de connexion"""
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("<h1 style='font-size: 4rem; margin-bottom: 1rem;'>📸</h1>", unsafe_allow_html=True)
    st.title("Messagerie Photo")
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        password = st.text_input("", type="password", key="login_input", placeholder="Code d'accès", label_visibility="collapsed")
        
        if st.button("Se connecter", type="primary", use_container_width=True):
            if password == "RLC":
                st.session_state.authenticated = True
                st.session_state.is_admin = True
                st.session_state.current_user = "admin"
                st.session_state.last_message_count = len(st.session_state.messages)
                st.rerun()
            elif password in st.session_state.user_passwords:
                st.session_state.authenticated = True
                st.session_state.is_admin = False
                st.session_state.current_user = "user"
                st.session_state.last_message_count = len(st.session_state.messages)
                st.rerun()
            else:
                st.error("❌ Code incorrect")

def admin_panel():
    """Panel admin"""
    st.sidebar.title("Panel Admin")
    st.sidebar.subheader("Mots de passe")
    
    for idx, pwd in enumerate(st.session_state.user_passwords):
        col1, col2 = st.sidebar.columns([3, 1])
        col1.text(pwd)
        if col2.button("🗑️", key=f"del_pwd_{idx}"):
            st.session_state.user_passwords.pop(idx)
            save_messages()
            st.rerun()
    
    new_pwd = st.sidebar.text_input("Nouveau mot de passe", key="new_pwd")
    if st.sidebar.button("➕ Ajouter"):
        if new_pwd and new_pwd not in st.session_state.user_passwords:
            st.session_state.user_passwords.append(new_pwd)
            save_messages()
            st.sidebar.success("✅ Ajouté")
            st.rerun()

def messagerie_tab():
    """Onglet de messagerie"""
    
    check_new_messages()
    
    st.header("📤 Nouveau message")
    
    # Géolocalisation automatique via JavaScript
    st.markdown("""
    <div id="gps-status" style="padding: 10px; background: rgba(255,255,255,0.1); border-radius: 10px; color: white; text-align: center; font-size: 14px; margin-bottom: 15px;">
        📍 <span id="gps-text">Tentative de géolocalisation...</span>
    </div>
    <script>
    (function() {
        if (navigator.geolocation) {
            navigator.geolocation.getCurrentPosition(
                function(position) {
                    const lat = position.coords.latitude;
                    const lon = position.coords.longitude;
                    const acc = Math.round(position.coords.accuracy);
                    
                    document.getElementById('gps-text').innerHTML = 
                        '✅ Position capturée : ' + lat.toFixed(4) + ', ' + lon.toFixed(4) + ' (±' + acc + 'm)';
                    
                    // Stocker dans localStorage pour que Streamlit puisse le récupérer
                    localStorage.setItem('gps_latitude', lat);
                    localStorage.setItem('gps_longitude', lon);
                    localStorage.setItem('gps_timestamp', Date.now());
                },
                function(error) {
                    let errorMsg = '';
                    switch(error.code) {
                        case error.PERMISSION_DENIED:
                            errorMsg = '❌ Permission refusée. Autorisez la géolocalisation dans votre navigateur.';
                            break;
                        case error.POSITION_UNAVAILABLE:
                            errorMsg = '⚠️ Position indisponible. Vérifiez votre GPS.';
                            break;
                        case error.TIMEOUT:
                            errorMsg = '⏱️ Délai expiré. Réessayez.';
                            break;
                    }
                    document.getElementById('gps-text').innerHTML = errorMsg;
                    localStorage.removeItem('gps_latitude');
                    localStorage.removeItem('gps_longitude');
                },
                {
                    enableHighAccuracy: true,
                    timeout: 15000,
                    maximumAge: 60000
                }
            );
        } else {
            document.getElementById('gps-text').innerHTML = '❌ Géolocalisation non supportée par ce navigateur.';
        }
    })();
    </script>
    """, unsafe_allow_html=True)
    
    # Option pour désactiver ou entrer manuellement
    with st.expander("⚙️ Options de géolocalisation", expanded=False):
        st.info("""
        **Mode automatique** : La position GPS est capturée automatiquement par votre navigateur.
        
        **Mode manuel** : Entrez vos coordonnées manuellement (clic droit sur Google Maps).
        
        **Désactivé** : Aucune position ne sera enregistrée.
        """)
        
        gps_option = st.radio(
            "Mode GPS",
            ["Automatique (recommandé)", "Manuel", "Désactivé"],
            key="gps_mode_radio",
            horizontal=True
        )
        
        if gps_option == "Manuel":
            col_lat, col_lon = st.columns(2)
            with col_lat:
                manual_lat = st.number_input("Latitude", value=0.0, format="%.6f", step=0.000001, key="manual_lat_input")
            with col_lon:
                manual_lon = st.number_input("Longitude", value=0.0, format="%.6f", step=0.000001, key="manual_lon_input")
    
    camera_photo = st.camera_input("📸 Prendre une photo", label_visibility="collapsed", key="camera_main")
    
    if camera_photo is not None:
        image = Image.open(camera_photo)
        
        has_human = True
        if CV2_AVAILABLE:
            with st.spinner("🔍 Vérification..."):
                has_human = verify_human_body_simple(image)
        
        if not has_human:
            st.error("❌ La photo doit contenir une partie du corps humain")
        else:
            text_input = st.text_input("", key="text_msg_input", placeholder="💬 Ajouter un message...", label_visibility="collapsed")
            
            # Composant pour lire le localStorage JavaScript
            st.markdown("""
            <script>
            // Envoyer les coordonnées GPS à Streamlit via query params
            const lat = localStorage.getItem('gps_latitude');
            const lon = localStorage.getItem('gps_longitude');
            if (lat && lon) {
                // Créer un événement personnalisé
                const event = new CustomEvent('gps-ready', {
                    detail: {latitude: parseFloat(lat), longitude: parseFloat(lon)}
                });
                window.dispatchEvent(event);
            }
            </script>
            """, unsafe_allow_html=True)
            
            if st.button("✉️ Envoyer", type="primary", use_container_width=True, key="send_msg_btn"):
                image_with_text = add_text_to_image(image, text_input) if text_input else image
                
                # Récupérer la position GPS selon le mode choisi
                location = None
                gps_mode = st.session_state.get('gps_mode_radio', 'Automatique (recommandé)')
                
                if gps_mode == "Manuel":
                    manual_lat = st.session_state.get('manual_lat_input', 0.0)
                    manual_lon = st.session_state.get('manual_lon_input', 0.0)
                    if manual_lat != 0.0 or manual_lon != 0.0:
                        location = {'latitude': manual_lat, 'longitude': manual_lon}
                
                elif gps_mode == "Automatique (recommandé)":
                    # Note: JavaScript localStorage n'est pas directement accessible depuis Python
                    # Solution: L'utilisateur voit sa position dans le bandeau vert
                    # On stocke via un input caché qui lit localStorage
                    st.info("""
                    ℹ️ **Géolocalisation automatique** : 
                    Si vous voyez "✅ Position capturée" ci-dessus, votre position a été détectée.
                    
                    ⚠️ **Limitation technique** : Streamlit ne peut pas accéder directement au GPS du navigateur.
                    Pour enregistrer votre position, utilisez le **mode Manuel** et copiez les coordonnées affichées ci-dessus.
                    
                    💡 **Astuce** : Sur mobile, activez le GPS et autorisez le navigateur. Les coordonnées s'afficheront automatiquement.
                    """)
                
                # Mode "Désactivé" : location reste None
                
                save_message(image_with_text, text_input, image, st.session_state.current_user, location)
                
                st.success("✅ Envoyé !")
                st.rerun()
    
    st.header("💬 Messages")
    
    if st.session_state.messages:
        for msg in reversed(st.session_state.messages):
            is_admin = msg['sender'] == "admin"
            container_class = "message-container-admin" if is_admin else "message-container-user"
            
            st.markdown(f'<div class="{container_class}"><div class="message-content">', unsafe_allow_html=True)
            
            timestamp = datetime.fromisoformat(msg['timestamp']).strftime('%d/%m %H:%M')
            st.write(f"**{timestamp}**")
            
            # Afficher l'icône GPS si géolocalisé
            if msg.get('location'):
                lat = msg['location']['latitude']
                lon = msg['location']['longitude']
                st.caption(f"📍 {lat:.4f}, {lon:.4f}")
            
            st.image(msg['image_with_text'], use_container_width=True)
            
            col1, col2 = st.columns([1, 1])
            with col1:
                img_bytes = io.BytesIO()
                msg['original_image'].save(img_bytes, format='PNG')
                st.download_button("📥", img_bytes.getvalue(), f"photo_{msg['id']}.png", "image/png", key=f"dl_{msg['id']}")
            with col2:
                if st.button("🗑️", key=f"del_{msg['id']}"):
                    delete_message(msg['id'])
                    st.rerun()
            
            st.markdown('</div></div>', unsafe_allow_html=True)
            st.divider()
    else:
        st.info("Aucun message")

def main_app():
    """Application principale"""
    st.title("📸 Messagerie Photo")

    display_counters()
    
    # Bouton de déconnexion en haut à droite
    col1, col2 = st.columns([5, 1])
    with col2:
        if st.button("🚪", key="logout_btn"):
            st.session_state.authenticated = False
            st.session_state.is_admin = False
            st.session_state.current_user = None
            st.rerun()
    
    # Système d'onglets
    tab1, tab2 = st.tabs(["💬 Messagerie", "🗺️ Carte"])
    
    with tab1:
        messagerie_tab()
    
    with tab2:
        create_map_view()

    # Sidebar
    with st.sidebar:
        if st.session_state.is_admin:
            admin_panel()
            st.markdown("---")
        
        st.write("### 📊 État du système")
        st.write(f"Messages en mémoire : **{len(st.session_state.messages)}**")
        st.write(f"GitHub : **{'✅ Configuré' if GITHUB_TOKEN and GITHUB_REPO else '❌ Non configuré'}**")
        st.write(f"OpenCV : **{'✅' if CV2_AVAILABLE else '❌'}**")
        st.write(f"MediaPipe : **{'✅' if MEDIAPIPE_AVAILABLE else '❌'}**")

        if not CV2_AVAILABLE or not MEDIAPIPE_AVAILABLE:
            st.warning("⚠️ Bibliothèques non chargées")
            if st.button("🔄 Recharger les bibliothèques", key="reload_libs_btn"):
                with st.spinner("Rechargement..."):
                    if reload_heavy_libraries():
                        st.success("✅ Rechargées avec succès !")
                        st.rerun()
                    else:
                        st.error("❌ Échec du rechargement")
        
        if st.button("🔄 Recharger depuis GitHub", key="reload_github_btn"):
            st.session_state.messages = load_messages()
            st.session_state.user_passwords = load_passwords()
            st.session_state.counters = load_counters()
            st.rerun()

if not st.session_state.authenticated:
    login_page()
else:
    main_app()