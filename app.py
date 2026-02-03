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
    layout="wide"
)

# --- CONSTANTES ---
IMG_SIZE = 224  # Taille d'image utilisée lors de l'entraînement
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
        'conseil': 'Ne jamais jeter à la poubelle normale. Danger pour l\'environnement.'
    },
    'biological': {
        'icon': '🥬',
        'couleur': '#51CF66',
        'poubelle': 'Poubelle marron (compost)',
        'recyclable': True,
        'conseil': 'Peut être composté à la maison ou dans les déchets organiques.'
    },
    'cardboard': {
        'icon': '📦',
        'couleur': '#FFD43B',
        'poubelle': 'Poubelle jaune (recyclage)',
        'recyclable': True,
        'conseil': 'Plier les cartons pour gagner de l\'espace.'
    },
    'clothes': {
        'icon': '👕',
        'couleur': '#748FFC',
        'poubelle': 'Conteneurs à vêtements',
        'recyclable': True,
        'conseil': 'Donner les vêtements en bon état à des associations.'
    },
    'glass': {
        'icon': '🍾',
        'couleur': '#20C997',
        'poubelle': 'Conteneur à verre',
        'recyclable': True,
        'conseil': 'Retirer les bouchons avant de jeter.'
    },
    'metal': {
        'icon': '🥫',
        'couleur': '#ADB5BD',
        'poubelle': 'Poubelle jaune (recyclage)',
        'recyclable': True,
        'conseil': 'Les boîtes de conserve doivent être vides et rincées.'
    },
    'paper': {
        'icon': '📄',
        'couleur': '#74C0FC',
        'poubelle': 'Poubelle jaune (recyclage)',
        'recyclable': True,
        'conseil': 'Pas de papier gras ou souillé dans le recyclage.'
    },
    'plastic': {
        'icon': '🧴',
        'couleur': '#FF8787',
        'poubelle': 'Poubelle jaune (recyclage)',
        'recyclable': True,
        'conseil': 'Vider et rincer les contenants en plastique.'
    },
    'shoes': {
        'icon': '👟',
        'couleur': '#845EF7',
        'poubelle': 'Conteneurs à chaussures',
        'recyclable': True,
        'conseil': 'Même usées, les chaussures peuvent être recyclées.'
    },
    'trash': {
        'icon': '🗑️',
        'couleur': '#495057',
        'poubelle': 'Poubelle noire (ordures ménagères)',
        'recyclable': False,
        'conseil': 'Déchets non recyclables. Essayez de réduire ce type de déchet.'
    }
}

# --- FONCTIONS ---

@st.cache_resource
def load_model():
    """Charge le modèle CNN pré-entraîné"""
    try:
        # Essayer de charger le modèle depuis le fichier
        model = keras.models.load_model('best_model.keras')
        return model
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle : {e}")
        st.info("Le modèle 'best_model.keras' n'a pas été trouvé. Veuillez l'ajouter dans le répertoire de l'application.")
        return None

def build_model_architecture():
    """Crée l'architecture du modèle (pour affichage ou si le modèle n'est pas chargé)"""
    model = keras.models.Sequential([
        # Bloc 1
        keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(224, 224, 3)),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2, 2)),
        
        # Bloc 2
        keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2, 2)),
        
        # Bloc 3
        keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2, 2)),
        
        # Bloc 4
        keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2, 2)),
        
        # Bloc 5
        keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2, 2)),
        
        # Bloc 6
        keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2, 2)),
        
        # Bloc 7
        keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2, 2)),
        
        # Classificateur
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        
        keras.layers.Dense(128, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        
        keras.layers.Dense(128, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        
        # Sortie
        keras.layers.Dense(10, activation='softmax')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def preprocess_image(image):
    """Prétraite l'image pour le modèle"""
    # Convertir PIL Image en array numpy
    img_array = np.array(image)
    
    # Redimensionner à 224x224
    img_resized = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    
    # Normaliser (0-1)
    img_normalized = img_resized / 255.0
    
    # Ajouter la dimension batch
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
    confidence = probabilities[predicted_idx] * 100
    
    # Top 3 prédictions
    top_3_idx = np.argsort(probabilities)[-3:][::-1]
    top_3_predictions = [
        (CLASS_NAMES[idx], probabilities[idx] * 100)
        for idx in top_3_idx
    ]
    
    return predicted_class, confidence, top_3_predictions

# --- INTERFACE STREAMLIT ---

def main():
    # Titre et description
    st.title("♻️ Classification Intelligente de Déchets")
    st.markdown("""
    Cette application utilise un réseau de neurones convolutif (CNN) pour classifier automatiquement 
    vos déchets et vous indiquer dans quelle poubelle les jeter.
    """)
    
    # Charger le modèle
    model = load_model()
    
    if model is None:
        st.warning("⚠️ Le modèle n'est pas disponible. L'application ne peut pas effectuer de prédictions.")
        
        # Afficher l'architecture du modèle
        with st.expander("📊 Voir l'architecture du modèle CNN"):
            model_arch = build_model_architecture()
            
            # Créer un buffer pour capturer le résumé
            from io import StringIO
            import sys
            
            buffer = StringIO()
            model_arch.summary(print_fn=lambda x: buffer.write(x + '\n'))
            summary_str = buffer.getvalue()
            
            st.code(summary_str, language='text')
        
        return
    
    # Barre latérale
    with st.sidebar:
        st.header("📖 Guide d'utilisation")
        st.markdown("""
        1. **Téléchargez** une photo de votre déchet
        2. **Attendez** la classification automatique
        3. **Consultez** les instructions de tri
        
        ### Types de déchets reconnus :
        """)
        
        for class_name in CLASS_NAMES:
            info = RECYCLING_INFO[class_name]
            st.markdown(f"{info['icon']} **{class_name.capitalize()}**")
        
        st.divider()
        
        # Statistiques du modèle
        st.header("📈 Informations du modèle")
        total_params = model.count_params()
        st.metric("Paramètres totaux", f"{total_params:,}")
        st.metric("Taille d'entrée", f"{IMG_SIZE}x{IMG_SIZE}")
        st.metric("Classes", len(CLASS_NAMES))
    
    # Zone principale
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📸 Télécharger une image")
        uploaded_file = st.file_uploader(
            "Choisissez une image de déchet",
            type=['png', 'jpg', 'jpeg'],
            help="Formats acceptés : PNG, JPG, JPEG"
        )
        
        if uploaded_file is not None:
            # Afficher l'image
            image = Image.open(uploaded_file)
            st.image(image, caption="Image téléchargée", use_container_width=True)
            
            # Bouton de prédiction
            if st.button("🔍 Classifier ce déchet", type="primary", use_container_width=True):
                with st.spinner("Classification en cours..."):
                    predicted_class, confidence, all_predictions = predict_waste(model, image)
                    
                    # Stocker les résultats dans session_state
                    st.session_state['last_prediction'] = {
                        'class': predicted_class,
                        'confidence': confidence,
                        'all_predictions': all_predictions,
                        'timestamp': datetime.now()
                    }
    
    with col2:
        st.subheader("📊 Résultats de classification")
        
        if 'last_prediction' in st.session_state:
            pred = st.session_state['last_prediction']
            predicted_class = pred['class']
            confidence = pred['confidence']
            all_predictions = pred['all_predictions']
            
            # Affichage du résultat principal
            info = RECYCLING_INFO[predicted_class]
            
            # Card de résultat
            st.markdown(f"""
            <div style="
                background-color: {info['couleur']}22;
                border-left: 5px solid {info['couleur']};
                padding: 20px;
                border-radius: 10px;
                margin: 10px 0;
            ">
                <h1 style="margin: 0; color: {info['couleur']};">
                    {info['icon']} {predicted_class.upper()}
                </h1>
                <h3 style="margin: 10px 0 0 0;">
                    Confiance: {confidence:.1f}%
                </h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Instructions de tri
            st.markdown("### ♻️ Instructions de tri")
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.info(f"**Poubelle:** {info['poubelle']}")
            
            with col_b:
                recyclable_text = "✅ Recyclable" if info['recyclable'] else "❌ Non recyclable"
                st.success(recyclable_text) if info['recyclable'] else st.error(recyclable_text)
            
            st.warning(f"💡 **Conseil:** {info['conseil']}")
            
            # Top 3 prédictions
            with st.expander("📈 Voir toutes les prédictions (Top 3)"):
                for i, (class_name, prob) in enumerate(all_predictions, 1):
                    class_info = RECYCLING_INFO[class_name]
                    st.markdown(f"""
                    **{i}. {class_info['icon']} {class_name.capitalize()}** - {prob:.1f}%
                    """)
                    st.progress(prob / 100)
        
        else:
            st.info("👆 Téléchargez une image et cliquez sur 'Classifier' pour voir les résultats.")
    
    # Section d'information supplémentaire
    st.divider()
    
    with st.expander("ℹ️ À propos du modèle"):
        st.markdown("""
        ### Architecture du modèle
        
        Ce modèle utilise une architecture CNN profonde avec :
        - **7 blocs convolutionnels** avec BatchNormalization et MaxPooling
        - **3 couches denses** avec Dropout pour la classification
        - **Input shape:** 224x224x3 (images RGB)
        - **10 classes** de déchets
        
        ### Entraînement
        - **Optimiseur:** Adam (learning rate: 0.001)
        - **Fonction de perte:** Categorical Crossentropy
        - **Augmentation des données:** Rotation, zoom, flip, etc.
        - **Early Stopping** avec patience de 5 epochs
        
        ### Performance
        Le modèle a été entraîné sur le dataset Kaggle "Garbage Classification v2" 
        avec une division 70/20/10 (train/test/validation).
        """)

if __name__ == "__main__":
    main()
