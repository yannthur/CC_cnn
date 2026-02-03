import streamlit as st
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np
import cv2
from datetime import datetime

# --- CONFIGURATION ---
st.set_page_config(
    page_title="Classification de Déchets 🗑️",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé pour améliorer le design
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(120deg, #84fab0 0%, #8fd3f4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 1rem 0;
    }
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .prediction-card {
        border-radius: 15px;
        padding: 25px;
        margin: 15px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease;
    }
    .prediction-card:hover {
        transform: translateY(-5px);
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        text-align: center;
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3rem;
        font-weight: bold;
        font-size: 1.1rem;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
    }
    div[data-testid="stExpander"] {
        border-radius: 10px;
        border: 1px solid #e0e0e0;
    }
    .upload-section {
        border: 2px dashed #84fab0;
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        background-color: #f8fff9;
    }
</style>
""", unsafe_allow_html=True)

# --- CONSTANTES ---
IMG_SIZE = 224
CLASS_NAMES = [
    'battery', 'biological', 'cardboard', 'clothes', 
    'glass', 'metal', 'paper', 'plastic', 'shoes', 'trash'
]

# Informations de recyclage pour chaque classe
RECYCLING_INFO = {
    'battery': {
        'icon': '🔋',
        'couleur': '#FF6B6B',
        'poubelle': 'Points de collecte spéciaux',
        'recyclable': True,
        'conseil': 'Ne jamais jeter à la poubelle normale. Danger pour l\'environnement.',
        'description': 'Les piles contiennent des métaux lourds toxiques.'
    },
    'biological': {
        'icon': '🥬',
        'couleur': '#51CF66',
        'poubelle': 'Poubelle marron (compost)',
        'recyclable': True,
        'conseil': 'Peut être composté à la maison ou dans les déchets organiques.',
        'description': 'Déchets alimentaires et végétaux biodégradables.'
    },
    'cardboard': {
        'icon': '📦',
        'couleur': '#FFD43B',
        'poubelle': 'Poubelle jaune (recyclage)',
        'recyclable': True,
        'conseil': 'Plier les cartons pour gagner de l\'espace.',
        'description': 'Le carton est recyclable à 100%.'
    },
    'clothes': {
        'icon': '👕',
        'couleur': '#748FFC',
        'poubelle': 'Conteneurs à vêtements',
        'recyclable': True,
        'conseil': 'Donner les vêtements en bon état à des associations.',
        'description': 'Les textiles peuvent être réutilisés ou recyclés.'
    },
    'glass': {
        'icon': '🍾',
        'couleur': '#20C997',
        'poubelle': 'Conteneur à verre',
        'recyclable': True,
        'conseil': 'Retirer les bouchons avant de jeter.',
        'description': 'Le verre se recycle à l\'infini.'
    },
    'metal': {
        'icon': '🥫',
        'couleur': '#ADB5BD',
        'poubelle': 'Poubelle jaune (recyclage)',
        'recyclable': True,
        'conseil': 'Les boîtes de conserve doivent être vides et rincées.',
        'description': 'Les métaux sont précieux et hautement recyclables.'
    },
    'paper': {
        'icon': '📄',
        'couleur': '#74C0FC',
        'poubelle': 'Poubelle jaune (recyclage)',
        'recyclable': True,
        'conseil': 'Pas de papier gras ou souillé dans le recyclage.',
        'description': 'Le papier peut être recyclé 5 à 7 fois.'
    },
    'plastic': {
        'icon': '🧴',
        'couleur': '#FF8787',
        'poubelle': 'Poubelle jaune (recyclage)',
        'recyclable': True,
        'conseil': 'Vider et rincer les contenants en plastique.',
        'description': 'Tous les plastiques ne sont pas recyclables de la même manière.'
    },
    'shoes': {
        'icon': '👟',
        'couleur': '#845EF7',
        'poubelle': 'Conteneurs à chaussures',
        'recyclable': True,
        'conseil': 'Même usées, les chaussures peuvent être recyclées.',
        'description': 'Les chaussures sont souvent revalorisées.'
    },
    'trash': {
        'icon': '🗑️',
        'couleur': '#495057',
        'poubelle': 'Poubelle noire (ordures ménagères)',
        'recyclable': False,
        'conseil': 'Déchets non recyclables. Essayez de réduire ce type de déchet.',
        'description': 'Ces déchets finiront en incinération ou enfouissement.'
    }
}

# --- FONCTIONS ---

@st.cache_resource
def load_model():
    """Charge le modèle CNN pré-entraîné"""
    try:
        model = keras.models.load_model('best_model.keras')
        return model
    except Exception as e:
        st.error(f"⚠️ Erreur lors du chargement du modèle : {e}")
        st.info("💡 Le modèle 'best_model.keras' n'a pas été trouvé. Veuillez l'ajouter dans le répertoire de l'application.")
        return None

def preprocess_image(image):
    """Prétraite l'image pour le modèle"""
    img_array = np.array(image)
    img_resized = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    img_normalized = img_resized / 255.0
    img_expanded = np.expand_dims(img_normalized, axis=0)
    return img_expanded

def predict_waste(model, image):
    """Effectue la prédiction sur une image"""
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image, verbose=0)
    
    # Obtenir les probabilités pour chaque classe
    probabilities = predictions[0]
    
    # Classe prédite
    predicted_idx = np.argmax(probabilities)
    predicted_class = CLASS_NAMES[predicted_idx]
    confidence = float(probabilities[predicted_idx] * 100)  # Convertir en float Python
    
    # Top 3 prédictions
    top_3_idx = np.argsort(probabilities)[-3:][::-1]
    top_3_predictions = [
        (CLASS_NAMES[idx], float(probabilities[idx] * 100))  # Convertir en float Python
        for idx in top_3_idx
    ]
    
    return predicted_class, confidence, top_3_predictions

def display_result_card(predicted_class, confidence):
    """Affiche une carte de résultat stylisée"""
    info = RECYCLING_INFO[predicted_class]
    
    st.markdown(f"""
    <div class="prediction-card" style="
        background: linear-gradient(135deg, {info['couleur']}22 0%, {info['couleur']}11 100%);
        border-left: 5px solid {info['couleur']};
    ">
        <div style="text-align: center;">
            <div style="font-size: 4rem; margin-bottom: 10px;">{info['icon']}</div>
            <h1 style="color: {info['couleur']}; margin: 10px 0; font-size: 2.5rem;">
                {predicted_class.upper()}
            </h1>
            <div style="font-size: 1.8rem; color: #666; font-weight: bold;">
                Confiance: {confidence:.1f}%
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# --- INITIALISATION SESSION STATE ---
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

if 'current_image' not in st.session_state:
    st.session_state.current_image = None

# --- INTERFACE STREAMLIT ---

def main():
    # Header
    st.markdown('<h1 class="main-header">♻️ Classification Intelligente de Déchets</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Identifiez vos déchets et apprenez à les trier correctement grâce à l\'IA</p>', unsafe_allow_html=True)
    
    # Charger le modèle
    model = load_model()
    
    if model is None:
        st.error("❌ L'application ne peut pas fonctionner sans le modèle. Veuillez charger 'best_model.keras'.")
        return
    
    # Barre latérale
    with st.sidebar:
        st.image("https://img.icons8.com/3d-fluency/94/recycle-sign.png", width=100)
        st.title("📖 Guide")
        
        st.markdown("""
        ### 🎯 Comment utiliser ?
        
        1. **📸 Téléchargez** une photo de déchet
        2. **🔍 Cliquez** sur "Classifier"
        3. **✅ Consultez** les résultats
        4. **♻️ Triez** correctement !
        """)
        
        st.divider()
        
        # Statistiques
        st.subheader("📊 Statistiques")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size: 2rem; color: #51CF66;">10</div>
                <div style="color: #666;">Classes</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size: 2rem; color: #748FFC;">{len(st.session_state.prediction_history)}</div>
                <div style="color: #666;">Prédictions</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.divider()
        
        # Types de déchets
        with st.expander("🗑️ Types reconnus", expanded=False):
            for class_name in CLASS_NAMES:
                info = RECYCLING_INFO[class_name]
                st.markdown(f"{info['icon']} **{class_name.capitalize()}**")
        
        st.divider()
        
        # Historique
        if st.session_state.prediction_history:
            st.subheader("📜 Historique")
            for i, pred in enumerate(reversed(st.session_state.prediction_history[-5:])):
                info = RECYCLING_INFO[pred['class']]
                st.markdown(f"{info['icon']} {pred['class']} - {pred['confidence']:.1f}%")
            
            if st.button("🗑️ Effacer l'historique"):
                st.session_state.prediction_history = []
                st.rerun()
    
    # Zone principale
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown("### 📸 Upload de l'image")
        
        # Zone d'upload stylisée
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Glissez-déposez ou cliquez pour choisir une image",
            type=['png', 'jpg', 'jpeg'],
            label_visibility="collapsed"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        if uploaded_file is not None:
            # Charger et afficher l'image
            image = Image.open(uploaded_file)
            st.session_state.current_image = image
            
            st.markdown("#### 🖼️ Aperçu")
            st.image(image, use_container_width=True, caption="Image à classifier")
            
            # Bouton de prédiction
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🔍 Classifier ce déchet", type="primary", use_container_width=True):
                with st.spinner("🔄 Classification en cours..."):
                    predicted_class, confidence, all_predictions = predict_waste(model, image)
                    
                    # Ajouter à l'historique
                    st.session_state.prediction_history.append({
                        'class': predicted_class,
                        'confidence': confidence,
                        'timestamp': datetime.now().strftime("%H:%M:%S"),
                        'all_predictions': all_predictions
                    })
                    
                    # Stocker les résultats pour affichage
                    st.session_state['last_prediction'] = {
                        'class': predicted_class,
                        'confidence': confidence,
                        'all_predictions': all_predictions,
                        'timestamp': datetime.now()
                    }
                    
                    st.success("✅ Classification terminée !")
                    st.rerun()
        else:
            st.info("👆 Veuillez télécharger une image pour commencer")
    
    with col2:
        st.markdown("### 📊 Résultats de classification")
        
        if 'last_prediction' in st.session_state and st.session_state.get('current_image') is not None:
            pred = st.session_state['last_prediction']
            predicted_class = pred['class']
            confidence = pred['confidence']
            all_predictions = pred['all_predictions']
            
            # Carte de résultat principale
            display_result_card(predicted_class, confidence)
            
            # Informations détaillées
            info = RECYCLING_INFO[predicted_class]
            
            st.markdown("#### ♻️ Informations de tri")
            
            # Grille d'informations
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.markdown(f"""
                <div class="metric-card" style="background-color: {info['couleur']}22;">
                    <div style="font-size: 1.5rem;">{info['icon']}</div>
                    <div style="font-weight: bold; margin-top: 10px;">Destination</div>
                    <div style="color: #666;">{info['poubelle']}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col_b:
                recyclable_color = "#51CF66" if info['recyclable'] else "#FF6B6B"
                recyclable_text = "✅ Recyclable" if info['recyclable'] else "❌ Non recyclable"
                st.markdown(f"""
                <div class="metric-card" style="background-color: {recyclable_color}22;">
                    <div style="font-size: 1.5rem;">{"♻️" if info['recyclable'] else "🚫"}</div>
                    <div style="font-weight: bold; margin-top: 10px;">Statut</div>
                    <div style="color: {recyclable_color}; font-weight: bold;">{recyclable_text}</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Conseils
            st.info(f"💡 **Conseil :** {info['conseil']}")
            st.markdown(f"ℹ️ {info['description']}")
            
            # Barre de confiance
            st.markdown("#### 📈 Niveau de confiance")
            st.progress(confidence / 100)
            st.caption(f"Le modèle est sûr à {confidence:.1f}% de sa prédiction")
            
            # Top 3 prédictions
            with st.expander("🔍 Voir les 3 meilleures prédictions", expanded=False):
                st.markdown("##### Prédictions alternatives")
                for i, (class_name, prob) in enumerate(all_predictions, 1):
                    class_info = RECYCLING_INFO[class_name]
                    
                    # Créer une barre de progression pour chaque prédiction
                    st.markdown(f"""
                    <div style="margin: 15px 0;">
                        <div style="display: flex; align-items: center; margin-bottom: 5px;">
                            <span style="font-size: 1.5rem; margin-right: 10px;">{class_info['icon']}</span>
                            <span style="font-weight: bold; flex-grow: 1;">{class_name.capitalize()}</span>
                            <span style="color: #666;">{prob:.1f}%</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    # Convertir explicitement en float Python pour st.progress
                    st.progress(float(prob) / 100.0)
        
        else:
            st.info("👈 Téléchargez une image et cliquez sur 'Classifier' pour voir les résultats")
            
            # Afficher des exemples
            st.markdown("#### 💡 Exemples de déchets")
            example_cols = st.columns(3)
            examples = ['battery', 'cardboard', 'plastic', 'glass', 'paper', 'biological']
            
            for idx, example in enumerate(examples[:3]):
                with example_cols[idx]:
                    info = RECYCLING_INFO[example]
                    st.markdown(f"""
                    <div style="text-align: center; padding: 10px;">
                        <div style="font-size: 3rem;">{info['icon']}</div>
                        <div style="font-size: 0.9rem; color: #666;">{example.capitalize()}</div>
                    </div>
                    """, unsafe_allow_html=True)
    
    # Section informative
    st.divider()
    
    col_info1, col_info2, col_info3 = st.columns(3)
    
    with col_info1:
        st.markdown("""
        <div class="metric-card">
            <div style="font-size: 2rem;">🎯</div>
            <h4>Précis</h4>
            <p style="color: #666; font-size: 0.9rem;">Modèle entraîné sur des milliers d'images</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_info2:
        st.markdown("""
        <div class="metric-card">
            <div style="font-size: 2rem;">⚡</div>
            <h4>Rapide</h4>
            <p style="color: #666; font-size: 0.9rem;">Classification en quelques secondes</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_info3:
        st.markdown("""
        <div class="metric-card">
            <div style="font-size: 2rem;">🌍</div>
            <h4>Écologique</h4>
            <p style="color: #666; font-size: 0.9rem;">Contribuez à un meilleur tri des déchets</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
