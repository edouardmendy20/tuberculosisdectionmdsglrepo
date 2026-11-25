import os

import joblib
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

st.set_page_config(
    page_title="Démo Deep Learning",
    page_icon="🫁",
    layout="centered",
)

IMG_SIZE = (150, 150)

MALADIES = {
    "Tuberculose": {
        "model_path": "vgg16_model_finetuned.h5",
        "classes": ["Normal", "Tuberculosis"],
    },
    "Pneumonie": {
        "model_path": "vgg16_pneumonia.h5",
        "classes": ["Normal", "Pneumonia"],
    },
}


@st.cache_resource(max_entries=1)
def load_model(model_path):
    if not os.path.exists(model_path):
        st.error(f"Fichier du modèle introuvable : {model_path}.")
        st.stop()
    return tf.keras.models.load_model(model_path, compile=False)


@st.cache_resource
def load_performance():
    for f in ["modele_performance.keras", "robust_scaler.joblib"]:
        if not os.path.exists(f):
            st.error(f"Fichier introuvable : {f}.")
            st.stop()
    model = tf.keras.models.load_model("modele_performance.keras", compile=False)
    scaler = joblib.load("robust_scaler.joblib")
    return model, scaler


def predict_image(model, image, classes):
    image = image.convert("RGB").resize(IMG_SIZE, Image.NEAREST)
    x = np.asarray(image, dtype=np.float32) / 255.0
    x = np.expand_dims(x, axis=0)
    p = float(model.predict(x, verbose=0)[0][0])
    return {classes[1]: p, classes[0]: 1.0 - p}


def predire_performance(model, scaler, hours, previous, extra, sleep, papers):
    extra_encoded = 1 if extra == "Yes" else 0
    # Même ordre que df.drop('Performance Index', axis=1)
    X_input = np.array([[hours, previous, extra_encoded, sleep, papers]], dtype=float)
    X_scaled = scaler.transform(X_input)
    return round(float(model.predict(X_scaled, verbose=0)[0][0]), 2)


st.title("Démo Deep Learning")
tab1, tab2 = st.tabs(["🫁 Radiographies", "🎓 Performance élève"])

# ---------------- Onglet 1 : radiographies ----------------
with tab1:
    st.write("Choisissez la maladie, puis téléversez une radiographie thoracique.")
    st.warning("Outil pédagogique : il ne remplace pas un diagnostic médical.")

    maladie = st.radio("Maladie à détecter", list(MALADIES.keys()), horizontal=True)
    config = MALADIES[maladie]
    classes = config["classes"]

    uploaded_file = st.file_uploader(
        "Radiographie thoracique", type=["png", "jpg", "jpeg"]
    )

    if uploaded_file is not None:
        model = load_model(config["model_path"])
        image = Image.open(uploaded_file)
        col_image, col_result = st.columns(2)
        with col_image:
            st.image(image, caption="Image téléversée")
        with col_result:
            with st.spinner("Analyse en cours..."):
                probas = predict_image(model, image, classes)
            prediction = max(probas, key=probas.get)
            st.subheader(f"Résultat : {maladie}")
            if prediction == classes[1]:
                st.error(f"**{prediction}** ({probas[prediction]:.1%})")
            else:
                st.success(f"**{prediction}** ({probas[prediction]:.1%})")
            for classe, proba in sorted(probas.items(), key=lambda x: x[1], reverse=True):
                st.write(f"{classe} : {proba:.1%}")
                st.progress(proba)
    else:
        st.info("Aucune image téléversée pour le moment.")

# ---------------- Onglet 2 : performance élève ----------------
with tab2:
    st.write("Renseignez les variables de l'étudiant pour estimer son Performance Index.")

    hours = st.number_input("Hours Studied (heures d'étude)", value=7.0)
    previous = st.number_input("Previous Scores (notes précédentes)", value=85.0)
    extra = st.radio("Extracurricular Activities", ["No", "Yes"], index=1, horizontal=True)
    sleep = st.number_input("Sleep Hours (heures de sommeil)", value=8.0)
    papers = st.number_input("Sample Question Papers Practiced", value=5.0)

    if st.button("Prédire la performance"):
        perf_model, scaler = load_performance()
        resultat = predire_performance(perf_model, scaler, hours, previous, extra, sleep, papers)
        st.metric("Performance Index prédit", resultat)

st.caption("DIT & AfriAI Solutions")