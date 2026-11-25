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

CAT_COLS = [
    "job", "marital", "education", "default", "housing",
    "loan", "contact", "month", "day_of_week", "poutcome"
]

CIFAR_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


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


@st.cache_resource
def load_souscription():
    fichiers = ["DNN_model.h5", "scaler.joblib"] + [f"{c}_encoder.joblib" for c in CAT_COLS]
    for f in fichiers:
        if not os.path.exists(f):
            st.error(f"Fichier introuvable : {f}.")
            st.stop()
    model = tf.keras.models.load_model("DNN_model.h5", compile=False)
    scaler = joblib.load("scaler.joblib")
    encoders = {c: joblib.load(f"{c}_encoder.joblib") for c in CAT_COLS}
    return model, scaler, encoders


@st.cache_resource
def load_cifar():
    if not os.path.exists("cnn_model.h5"):
        st.error("Fichier introuvable : cnn_model.h5.")
        st.stop()
    return tf.keras.models.load_model("cnn_model.h5", compile=False)


def predict_image(model, image, classes):
    image = image.convert("RGB").resize(IMG_SIZE, Image.NEAREST)
    x = np.asarray(image, dtype=np.float32) / 255.0
    x = np.expand_dims(x, axis=0)
    p = float(model.predict(x, verbose=0)[0][0])
    return {classes[1]: p, classes[0]: 1.0 - p}


def predire_performance(model, scaler, hours, previous, extra, sleep, papers):
    extra_encoded = 1 if extra == "Yes" else 0
    X_input = np.array([[hours, previous, extra_encoded, sleep, papers]], dtype=float)
    X_scaled = scaler.transform(X_input)
    return round(float(model.predict(X_scaled, verbose=0)[0][0]), 2)


def predict_cifar(model, image):
    image = image.convert("RGB").resize((32, 32))
    image_array = np.array(image, dtype=np.float32) / 255.0
    image_array = image_array.reshape(1, 32, 32, 3)
    predictions = model.predict(image_array, verbose=0)[0]
    return {CIFAR_CLASSES[i]: float(predictions[i]) for i in range(10)}


st.title("Démo Deep Learning")
tab1, tab2, tab3, tab4 = st.tabs(
    ["🫁 Radiographies", "🎓 Performance élève", "🏦 Souscription", "🖼️ CIFAR-10"]
)

# ---------------- Onglet 1 : radiographies ----------------
with tab1:
    st.write("Choisissez la maladie, puis téléversez une radiographie thoracique.")
    st.warning("Outil pédagogique : il ne remplace pas un diagnostic médical.")

    maladie = st.radio("Maladie à détecter", list(MALADIES.keys()), horizontal=True)
    config = MALADIES[maladie]
    classes = config["classes"]

    uploaded_file = st.file_uploader(
        "Radiographie thoracique", type=["png", "jpg", "jpeg"], key="radio_uploader"
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

# ---------------- Onglet 3 : souscription ----------------
with tab3:
    st.subheader("Bank Marketing Prediction")
    st.write("Renseignez les informations du client pour prédire s'il va souscrire.")

    sous_model, sous_scaler, sous_encoders = load_souscription()

    def choix(col, label):
        return st.selectbox(label, list(sous_encoders[col].classes_), key=f"sous_{col}")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("age", value=40, step=1, key="sous_age")
        job = choix("job", "job")
        marital = choix("marital", "marital")
        education = choix("education", "education")
        default = choix("default", "default")
        housing = choix("housing", "housing")
        loan = choix("loan", "loan")
        contact = choix("contact", "contact")

    with col2:
        month = choix("month", "month")
        day_of_week = choix("day_of_week", "day_of_week")
        duration = st.number_input("duration", value=200, step=1, key="sous_duration")
        campaign = st.number_input("campaign", value=2, step=1, key="sous_campaign")
        pdays = st.number_input("pdays", value=999, step=1, key="sous_pdays")
        previous_c = st.number_input("previous", value=0, step=1, key="sous_previous")
        poutcome = choix("poutcome", "poutcome")

    if st.button("Prédire la souscription"):
        vals = {
            "job": job, "marital": marital, "education": education,
            "default": default, "housing": housing, "loan": loan,
            "contact": contact, "month": month,
            "day_of_week": day_of_week, "poutcome": poutcome,
        }
        enc = {c: sous_encoders[c].transform([vals[c]])[0] for c in CAT_COLS}

        x_new = np.array([
            age, enc["job"], enc["marital"], enc["education"], enc["default"],
            enc["housing"], enc["loan"], enc["contact"], enc["month"],
            enc["day_of_week"], duration, campaign, pdays, previous_c,
            enc["poutcome"],
        ], dtype=float).reshape(1, -1)

        x_new = sous_scaler.transform(x_new)
        proba = float(sous_model.predict(x_new, verbose=0)[0][0])
        y_pred = int(np.round(proba))

        if y_pred == 1:
            st.success(f"Souscrire (Probabilité : {proba:.1%})")
        else:
            st.error(f"Pas souscrire (Probabilité : {proba:.1%})")

# ---------------- Onglet 4 : CIFAR-10 ----------------
with tab4:
    st.subheader("Classificateur CIFAR-10")
    st.write("Téléverse une image pour prédire sa classe parmi les 10 catégories CIFAR-10.")

    cifar_file = st.file_uploader(
        "Choisir une image...", type=["png", "jpg", "jpeg"], key="cifar_uploader"
    )

    if cifar_file is not None:
        cifar_model = load_cifar()
        image = Image.open(cifar_file)
        col_img, col_res = st.columns(2)
        with col_img:
            st.image(image, caption="Image téléversée")
        with col_res:
            with st.spinner("Classification en cours..."):
                probas = predict_cifar(cifar_model, image)
            top_4 = sorted(probas.items(), key=lambda x: x[1], reverse=True)[:4]
            best_class, best_prob = top_4[0]
            st.success(f"Prédiction : **{best_class}** ({best_prob:.1%})")
            st.write("Top 4 des probabilités :")
            for classe, proba in top_4:
                st.write(f"{classe} : {proba:.1%}")
                st.progress(proba)
    else:
        st.info("Aucune image téléversée pour le moment.")

st.caption("DIT & AfriAI Solutions")