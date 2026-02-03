import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import cv2
import os
from datetime import datetime
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# --- CONFIGURATION ---
st.set_page_config(
    page_title="Classificateur de Déchets IA",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Classes de déchets (basées sur vos dossiers)
CLASSES = ['battery', 'biological', 'cardboard', 'clothes', 'glass', 
           'metal', 'paper', 'plastic', 'shoes', 'trash']

# Descriptions et conseils de recyclage
RECYCLING_INFO = {
    'battery': {
        'emoji': '🔋',
        'nom': 'Batterie',
        'description': 'Déchets électroniques contenant des batteries',
        'recyclage': 'À déposer dans les points de collecte spécialisés. Ne jamais jeter avec les ordures ménagères.',
        'danger': 'Contient des substances toxiques et inflammables'
    },
    'biological': {
        'emoji': '🥬',
        'nom': 'Déchets biologiques',
        'description': 'Déchets organiques et compostables',
        'recyclage': 'Compostage domestique ou collecte des déchets organiques',
        'danger': 'Aucun danger, mais peut produire du méthane en décharge'
    },
    'cardboard': {
        'emoji': '📦',
        'nom': 'Carton',
        'description': 'Emballages en carton',
        'recyclage': 'Bac de recyclage papier/carton. Aplatir avant de jeter.',
        'danger': 'Aucun danger'
    },
    'clothes': {
        'emoji': '👕',
        'nom': 'Vêtements',
        'description': 'Textiles et vêtements',
        'recyclage': 'Conteneurs de collecte textile ou associations caritatives',
        'danger': 'Aucun danger'
    },
    'glass': {
        'emoji': '🍾',
        'nom': 'Verre',
        'description': 'Bouteilles et contenants en verre',
        'recyclage': 'Conteneur à verre. Retirer les bouchons.',
        'danger': 'Risque de coupure'
    },
    'metal': {
        'emoji': '🔩',
        'nom': 'Métal',
        'description': 'Objets métalliques et canettes',
        'recyclage': 'Bac de recyclage ou déchetterie selon la taille',
        'danger': 'Risque de coupure pour certains objets'
    },
    'paper': {
        'emoji': '📄',
        'nom': 'Papier',
        'description': 'Documents et papiers',
        'recyclage': 'Bac de recyclage papier',
        'danger': 'Aucun danger'
    },
    'plastic': {
        'emoji': '🥤',
        'nom': 'Plastique',
        'description': 'Emballages et objets en plastique',
        'recyclage': 'Bac de recyclage selon le type de plastique',
        'danger': 'Pollution environnementale importante'
    },
    'shoes': {
        'emoji': '👟',
        'nom': 'Chaussures',
        'description': 'Chaussures et accessoires',
        'recyclage': 'Conteneurs spécialisés ou associations',
        'danger': 'Aucun danger'
    },
    'trash': {
        'emoji': '🗑️',
        'nom': 'Déchets non recyclables',
        'description': 'Déchets ménagers non recyclables',
        'recyclage': 'Poubelle des ordures ménagères',
        'danger': 'Variable selon le contenu'
    }
}

# --- CHARGEMENT DU MODÈLE ---
@st.cache_resource
def load_model():
    """Charge le modèle pré-entraîné"""
    try:
        model = keras.models.load_model('best_model.keras')
        return model
    except Exception as e:
        st.error(f"⚠️ Erreur lors du chargement du modèle : {e}")
        st.info("Assurez-vous que le fichier 'model.h5' est présent dans le répertoire.")
        return None

# --- FONCTIONS DE TRAITEMENT D'IMAGE ---
def preprocess_image(image, target_size=(64, 64)):
    """Prétraite l'image pour la prédiction"""
    # Convertir en RGB si nécessaire
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Redimensionner
    image = image.resize(target_size)
    
    # Convertir en array et normaliser
    img_array = np.array(image)
    img_array = img_array / 255.0
    
    # Ajouter la dimension batch
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def predict_waste(model, image):
    """Effectue la prédiction sur une image"""
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image, verbose=0)
    
    # Obtenir les probabilités pour chaque classe
    probabilities = predictions[0]
    
    # Classe prédite
    predicted_class_idx = np.argmax(probabilities)
    predicted_class = CLASSES[predicted_class_idx]
    confidence = probabilities[predicted_class_idx] * 100
    
    # Créer un dictionnaire de toutes les prédictions
    all_predictions = {CLASSES[i]: float(probabilities[i] * 100) for i in range(len(CLASSES))}
    
    return predicted_class, confidence, all_predictions

# --- HISTORIQUE DES PRÉDICTIONS ---
def init_history():
    """Initialise l'historique des prédictions"""
    if 'prediction_history' not in st.session_state:
        st.session_state.prediction_history = []

def add_to_history(image_name, predicted_class, confidence):
    """Ajoute une prédiction à l'historique"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.prediction_history.append({
        'timestamp': timestamp,
        'image': image_name,
        'classe': predicted_class,
        'confiance': confidence
    })

# --- CSS PERSONNALISÉ ---
def load_custom_css():
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #2E7D32;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #555;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #f0f7ff;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #2196F3;
        margin: 1rem 0;
    }
    .recycling-box {
        background-color: #e8f5e9;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #4CAF50;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3e0;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #FF9800;
        margin: 1rem 0;
    }
    .stat-card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# --- INTERFACE PRINCIPALE ---
def main():
    load_custom_css()
    init_history()
    
    # En-tête
    st.markdown('<h1 class="main-header">♻️ Classificateur de Déchets IA</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Identifiez automatiquement le type de déchet et apprenez comment le recycler correctement</p>', unsafe_allow_html=True)
    
    # Charger le modèle
    model = load_model()
    
    if model is None:
        st.stop()
    
    # Barre latérale
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Mode de saisie
        input_mode = st.radio(
            "Mode d'entrée",
            ["📤 Télécharger une image", "📸 Utiliser la caméra", "🖼️ Images d'exemple"],
            index=0
        )
        
        st.divider()
        
        # Options d'affichage
        st.subheader("Options d'affichage")
        show_probabilities = st.checkbox("Afficher toutes les probabilités", value=True)
        show_recycling_info = st.checkbox("Afficher les infos de recyclage", value=True)
        
        st.divider()
        
        # Statistiques
        if st.session_state.prediction_history:
            st.subheader("📊 Statistiques")
            st.metric("Prédictions totales", len(st.session_state.prediction_history))
            
            # Classe la plus prédite
            df_history = pd.DataFrame(st.session_state.prediction_history)
            most_common = df_history['classe'].value_counts().index[0]
            st.metric("Type le plus fréquent", RECYCLING_INFO[most_common]['emoji'] + " " + RECYCLING_INFO[most_common]['nom'])
        
        st.divider()
        
        # À propos
        with st.expander("ℹ️ À propos"):
            st.write("""
            **Classificateur de Déchets IA**
            
            Ce système utilise un réseau de neurones convolutifs (CNN) 
            pour identifier automatiquement 10 types de déchets différents.
            
            **Classes supportées :**
            - 🔋 Batteries
            - 🥬 Déchets biologiques
            - 📦 Carton
            - 👕 Vêtements
            - 🍾 Verre
            - 🔩 Métal
            - 📄 Papier
            - 🥤 Plastique
            - 👟 Chaussures
            - 🗑️ Déchets non recyclables
            
            **Précision du modèle :** Optimisé avec Keras Tuner
            """)
    
    # Zone principale
    image = None
    image_name = None
    
    # Gestion des différents modes d'entrée
    if input_mode == "📤 Télécharger une image":
        uploaded_file = st.file_uploader(
            "Choisissez une image de déchet",
            type=['jpg', 'jpeg', 'png', 'webp'],
            help="Formats supportés : JPG, JPEG, PNG, WEBP"
        )
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            image_name = uploaded_file.name
    
    elif input_mode == "📸 Utiliser la caméra":
        camera_image = st.camera_input("Prenez une photo du déchet")
        if camera_image is not None:
            image = Image.open(camera_image)
            image_name = f"camera_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    
    else:  # Images d'exemple
        st.info("💡 Sélectionnez une image d'exemple pour tester le classificateur")
        
        # Créer des colonnes pour les exemples
        example_cols = st.columns(5)
        
        # Ici vous pouvez ajouter des images d'exemple si vous en avez
        st.warning("⚠️ Fonctionnalité en cours de développement. Veuillez utiliser le mode 'Télécharger une image'.")
    
    # Traitement de l'image
    if image is not None:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📷 Image téléchargée")
            st.image(image, use_container_width=True, caption=image_name)
            
            # Informations sur l'image
            st.info(f"**Taille :** {image.size[0]} x {image.size[1]} pixels")
        
        with col2:
            st.subheader("🤖 Analyse en cours...")
            
            # Prédiction
            with st.spinner("Classification en cours..."):
                predicted_class, confidence, all_predictions = predict_waste(model, image)
            
            # Affichage du résultat principal
            info = RECYCLING_INFO[predicted_class]
            
            st.markdown(f"""
            <div class="prediction-box">
                <h2>{info['emoji']} {info['nom']}</h2>
                <h3>Confiance : {confidence:.2f}%</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Barre de progression pour la confiance
            st.progress(confidence / 100)
            
            # Ajouter à l'historique
            add_to_history(image_name, predicted_class, confidence)
        
        # Informations détaillées
        st.divider()
        
        if show_probabilities:
            st.subheader("📊 Probabilités détaillées")
            
            # Trier les prédictions par probabilité décroissante
            sorted_predictions = sorted(all_predictions.items(), key=lambda x: x[1], reverse=True)
            
            # Créer un graphique à barres
            fig = go.Figure(data=[
                go.Bar(
                    x=[RECYCLING_INFO[cls]['emoji'] + " " + RECYCLING_INFO[cls]['nom'] for cls, _ in sorted_predictions],
                    y=[prob for _, prob in sorted_predictions],
                    marker_color=['#4CAF50' if i == 0 else '#90CAF9' for i in range(len(sorted_predictions))],
                    text=[f'{prob:.2f}%' for _, prob in sorted_predictions],
                    textposition='outside'
                )
            ])
            
            fig.update_layout(
                title="Distribution des probabilités",
                xaxis_title="Type de déchet",
                yaxis_title="Probabilité (%)",
                height=400,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        if show_recycling_info:
            st.subheader("♻️ Informations de recyclage")
            
            col_info1, col_info2 = st.columns(2)
            
            with col_info1:
                st.markdown(f"""
                <div class="info-box">
                    <h4>📝 Description</h4>
                    <p>{info['description']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="recycling-box">
                    <h4>♻️ Comment recycler</h4>
                    <p>{info['recyclage']}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col_info2:
                st.markdown(f"""
                <div class="warning-box">
                    <h4>⚠️ Précautions</h4>
                    <p>{info['danger']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Conseils supplémentaires
                st.success("💡 **Astuce :** Pensez toujours à nettoyer vos contenants avant de les recycler !")
    
    else:
        # Message d'accueil
        st.info("👆 Veuillez télécharger ou prendre une photo d'un déchet pour commencer l'analyse")
        
        # Afficher des statistiques si disponibles
        if st.session_state.prediction_history:
            st.subheader("📈 Historique des prédictions")
            
            df_history = pd.DataFrame(st.session_state.prediction_history)
            
            # Graphique de distribution
            fig = px.histogram(
                df_history,
                x='classe',
                title="Distribution des types de déchets identifiés",
                labels={'classe': 'Type de déchet', 'count': 'Nombre'},
                color='classe'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Tableau détaillé
            with st.expander("📋 Voir l'historique détaillé"):
                # Formater l'affichage
                df_display = df_history.copy()
                df_display['Emoji'] = df_display['classe'].map(lambda x: RECYCLING_INFO[x]['emoji'])
                df_display['Type'] = df_display['classe'].map(lambda x: RECYCLING_INFO[x]['nom'])
                df_display['Confiance (%)'] = df_display['confiance'].round(2)
                
                st.dataframe(
                    df_display[['timestamp', 'Emoji', 'Type', 'Confiance (%)', 'image']],
                    use_container_width=True,
                    hide_index=True
                )
                
                # Bouton pour télécharger l'historique
                csv = df_history.to_csv(index=False)
                st.download_button(
                    label="📥 Télécharger l'historique (CSV)",
                    data=csv,
                    file_name=f"historique_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
                
                # Bouton pour effacer l'historique
                if st.button("🗑️ Effacer l'historique", type="secondary"):
                    st.session_state.prediction_history = []
                    st.rerun()

if __name__ == "__main__":
    main()
