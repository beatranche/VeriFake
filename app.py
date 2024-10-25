import streamlit as st
import pandas as pd
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import openai
from dotenv import load_dotenv
from textblob import TextBlob
from utils.functions import TextPreprocessor
from scraper.scraper import main as scraper_main
from streamlit_option_menu import option_menu

# Configuración para cargar variables de entorno (API_KEY de OpenAI)
load_dotenv()  
openai.api_key = os.getenv("API_KEY")  

# ---------------------------
# Funciones auxiliares
# ---------------------------

def load_models():
    """Carga los modelos de clasificación desde archivos pickle."""
    with open("models/logreg_model.pkl", "rb") as f:
        logistic_regression_model = pickle.load(f)
    with open("models/rf_model.pkl", "rb") as f:
        random_forest_model = pickle.load(f)
    with open("models/svm_model.pkl", "rb") as f:
        svm_model = pickle.load(f)
    with open("models/xgb_model.pkl", "rb") as f:
        xgb_model = pickle.load(f)
    with open("models/voting_classifier.pkl", "rb") as f:
        voting_classifier = pickle.load(f)
   
    return logistic_regression_model, random_forest_model, svm_model, xgb_model, voting_classifier

def load_vectorizer_svd():
    """Carga el vectorizador TF-IDF y el modelo de reducción de dimensionalidad SVD desde archivos."""
    with open("models/tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open("models/svd_model.pkl", "rb") as f:
        svd = pickle.load(f)
    return vectorizer, svd

def predict_news(text, model, vectorizer, svd):
    """
    Realiza una predicción de si una noticia es falsa o real.
    Preprocesa el texto de entrada, lo vectoriza y lo transforma mediante SVD.
    """
    preprocessor = TextPreprocessor()
    cleaned_text = preprocessor.clean_text(text)
    lemmatized_text = preprocessor.lemmatize_text(cleaned_text)
    text_vectorized = vectorizer.transform([lemmatized_text])
    text_svd = svd.transform(text_vectorized)
    prediction = model.predict(text_svd)
    try:
        probabilidad = model.predict_proba(text_svd)
    except AttributeError:
        probabilidad = None
    return prediction[0], probabilidad

# ---------------------------
# Configuración de la Aplicación
# ---------------------------

# Configuración del menú de navegación lateral con iconos
with st.sidebar:
    selected = option_menu(
        menu_title="Menú de Navegación",
        options=["Predicción de Fake News", "Scraping de Noticias", "Análisis de Datos", "Generador de Titulares"],
        icons=["newspaper", "cloud-download", "bar-chart", "lightbulb"],  # Iconos del menú
        menu_icon="cast",
        default_index=0,
    )

# ---------------------------
# Opción 1: Predicción de Fake News
# ---------------------------

if selected == "Predicción de Fake News":
    st.title("Detección de Noticias Falsas")
    logistic_regression_model, random_forest_model, svm_model, xgb_model, voting_classifier = load_models()
    vectorizer, svd = load_vectorizer_svd()

    # Entrada del usuario con límite de caracteres
    user_input = st.text_area("Introduce el texto de la noticia que deseas analizar:", max_chars=1000)
    st.write(f"Longitud del texto actual: {len(user_input)} caracteres (máximo permitido: 1000).")

    # Selección del modelo de clasificación
    model_option = st.selectbox(
        "Elige el modelo que deseas utilizar para la predicción:",
        ("Logistic Regression", "Random Forest", "SVM", "XGBoost", "Voting Classifier")
    )

    # Botón para iniciar el análisis de la noticia
    if st.button("Analizar Noticia"):
        if user_input:
            # Selección del modelo
            model = {
                "Logistic Regression": logistic_regression_model,
                "Random Forest": random_forest_model,
                "SVM": svm_model,
                "XGBoost": xgb_model,
                "Voting Classifier": voting_classifier,
            }.get(model_option, logistic_regression_model)

            # Predicción y resultado
            prediction, probabilidad = predict_news(user_input, model, vectorizer, svd)
            if prediction == 1:
                st.success(f"✅ Esta noticia podría ser verdadera. (Confianza: {probabilidad[0][1]*100:.2f}%)")
            else:
                st.error(f"⚠️ Esta noticia parece falsa. (Confianza: {probabilidad[0][0]*100:.2f}%)")
                
            if probabilidad is not None:
                confianza = max(probabilidad[0]) * 100
                st.write(f"El modelo está {confianza:.2f}% seguro de su predicción.")
            else:
                st.write("El modelo seleccionado no proporciona un porcentaje de confianza.")
        else:
            st.warning("Por favor, introduce el texto de una noticia.")

# ---------------------------
# Opción 2: Scraping de Noticias
# ---------------------------

if selected == "Scraping de Noticias":
    st.title("Scraping de Noticias Falsas")

    # Configuración de parámetros
    pages = st.number_input("Número de páginas", min_value=1, max_value=20, value=3)
    workers = st.number_input("Número de workers (threads)", min_value=1, max_value=10, value=3)

    # Opción para guardar resultados
    guardar_archivo = st.checkbox("¿Deseas guardar los resultados en un archivo CSV?")
    if guardar_archivo:
        file_name_input = st.text_input("Nombre del archivo (sin extensión):")

    # Botón para ejecutar el scraping
    if st.button("Ejecutar Scraping"):
        with st.spinner("Realizando scraping, por favor espera..."):
            # Ejecutar la función de scraping y procesar datos
            scraped_data = scraper_main(pages, workers)
            df_scraped = pd.DataFrame(scraped_data)

            # Agrupar datos y crear un DataFrame combinado
            data_combined = []
            grouped = df_scraped.groupby('Index')
            for name, group in grouped:
                headlines = group['Headline'].dropna().tolist()
                links = group['Link'].dropna().tolist()
                for i in range(min(len(headlines), len(links))):
                    data_combined.append({'Index': name, 'Headline': headlines[i], 'Link': links[i]})

            df_combined = pd.DataFrame(data_combined)
            st.success("Scraping completado.")
            st.dataframe(df_combined.head())

            # Guardar archivo si el usuario elige esta opción
            if guardar_archivo and file_name_input:
                output_path = f"data/{file_name_input}.csv"
                df_combined.to_csv(output_path, index=False)
                st.success(f"Archivo guardado como: {output_path}")
                st.write(f"Ubicación del archivo: {os.path.abspath(output_path)}")

# ---------------------------
# Opción 3: Análisis de Datos
# ---------------------------

if selected == "Análisis de Datos":
    st.title("Análisis de Noticias Falsas y Reales")
    df = pd.read_csv('data/new_data.csv')

    # Distribución de noticias falsas vs reales
    st.subheader("Distribución de Noticias Falsas vs Reales")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(x='Category', data=df, ax=ax)
    st.pyplot(fig)

    # Distribución de la longitud del texto
    st.subheader("Distribución de la Longitud del Texto (Fake vs Real)")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data=df, x='Text_Length', hue='Category', kde=True, bins=30, ax=ax)
    st.pyplot(fig)

    # Fuentes más comunes para noticias falsas y reales
    st.subheader("Principales Fuentes de Noticias Falsas y Reales")
    fake_sources = df[df['Category'] == 0]['Source'].value_counts().head(10)
    real_sources = df[df['Category'] == 1]['Source'].value_counts().head(10)
    st.markdown("##### Fuentes de Noticias Falsas")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=fake_sources.index, y=fake_sources.values, ax=ax)
    st.pyplot(fig)
    st.markdown("##### Fuentes de Noticias Reales")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=real_sources.index, y=real_sources.values, ax=ax)
    st.pyplot(fig)

    # Análisis de sentimiento
    st.subheader("Distribución del Sentimiento en Noticias Falsas vs Reales")
    def get_sentiment(text):
        analysis = TextBlob(text)
        return analysis.sentiment.polarity
    df['Sentiment'] = df['Text'].apply(get_sentiment)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data=df, x='Sentiment', hue='Category', kde=True, ax=ax)
    st.pyplot(fig)

# ---------------------------
# Opción 4: Generador de Titulares Clickbait
# ---------------------------

if selected == "Generador de Titulares":
    st.title("Generador de Titulares Clickbait")
    noticia = st.text_input("Introduce la noticia para el titular clickbait:")
    if st.button("Generar titular clickbait"):
        if noticia:
            prompt = f"Escribe un titular de clickbait sobre el tema: '{noticia}'"
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=50,
                    temperature=0.8,
                    top_p=0.9,
                    n=1
                )
                titular_clickbait = response['choices'][0]['message']['content']
                st.write("Titular generado:")
                st.success(titular_clickbait)
            except Exception as e:
                st.error(f"Error al generar el titular: {e}")
        else:
            st.warning("Introduce un tema para generar un titular.")

