import streamlit as st
import pandas as pd
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from textblob import TextBlob
from functions import TextPreprocessor
from scraper import main as scraper_main
from streamlit_option_menu import option_menu
from datetime import datetime


# Cargar modelos guardados
def load_models():
    with open("logreg_model.pkl", "rb") as f:
        logistic_regression_model = pickle.load(f)
    with open("rf_model.pkl", "rb") as f:
        random_forest_model = pickle.load(f)
    with open("svm_model.pkl", "rb") as f:
        svm_model = pickle.load(f)
    with open("xgb_model.pkl", "rb") as f:
        xgb_model = pickle.load(f)
    with open("voting_classifier.pkl", "rb") as f:
        voting_classifier = pickle.load(f)
   
    return logistic_regression_model, random_forest_model, svm_model, xgb_model, voting_classifier

def load_vectorizer_svd():
    with open("tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open("svd_model.pkl", "rb") as f:
        svd = pickle.load(f)
    return vectorizer, svd

# Función para predecir si la noticia es falsa o real
def predict_news(text, model, vectorizer, svd):
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

# Configuración de la aplicación con iconos
with st.sidebar:
    selected = option_menu(
        menu_title="Menú de Navegación",
        options=["Predicción de Fake News", "Scraping de Noticias", "Análisis de datos"],
        icons=["newspaper", "cloud-download", "bar-chart"],  # Iconos del menú
        menu_icon="cast",
        default_index=0,
    )

# Opción 1: Análisis de Fake News
if selected == "Predicción de Fake News":
    st.title("Detección de Noticias Falsas")

    logistic_regression_model, random_forest_model, svm_model, xgb_model, voting_classifier = load_models()
    vectorizer, svd = load_vectorizer_svd()

    user_input = st.text_area("Introduce el texto de la noticia que deseas analizar:", max_chars=1000)
    st.write(f"Longitud del texto actual: {len(user_input)} caracteres (máximo permitido: 1000).")

    model_option = st.selectbox(
        "Elige el modelo que deseas utilizar para la predicción:",
        ("Logistic Regression", "Random Forest", "SVM", "XGBoost", "Voting Classifier")
    )

    if st.button("Analizar Noticia"):
        if user_input:
            if model_option == "Logistic Regression":
                model = logistic_regression_model
            elif model_option == "Random Forest":
                model = random_forest_model
            elif model_option == "SVM":
                model = svm_model
            elif model_option == "XGBoost":
                model = xgb_model
            elif model_option == "Voting Classifier":
                model = voting_classifier
           

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
 
# Opción 2: Scraping de Noticias
if selected == "Scraping de Noticias":

    st.title("Scraping de Noticias Falsas")

    # Parámetros del scraping
    pages = st.number_input("Número de páginas a raspar", min_value=1, max_value=20, value=3)
    workers = st.number_input("Número de trabajadores (threads)", min_value=1, max_value=10, value=3)

    # Preguntar si el usuario desea guardar el archivo
    guardar_archivo = st.checkbox("¿Deseas guardar los resultados en un archivo CSV?")

    # Si elige guardar el archivo, permite seleccionar el nombre del archivo
    if guardar_archivo:
        file_name_input = st.text_input("Nombre del archivo (sin extensión):", value=f"scraped_news_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

    if st.button("Ejecutar Scraping"):
        with st.spinner("Realizando scraping, por favor espera..."):
            # Ejecutar el scraping
            scraped_data = scraper_main(pages, workers)
            df_scraped = pd.DataFrame(scraped_data)

            # Agrupar y combinar datos
            data_combined = []
            grouped = df_scraped.groupby('Index')

            for name, group in grouped:
                headlines = group['Headline'].dropna().tolist()
                links = group['Link'].dropna().tolist()
                for i in range(min(len(headlines), len(links))):
                    data_combined.append({
                        'Index': name,
                        'Headline': headlines[i],
                        'Link': links[i]
                    })

            df_combined = pd.DataFrame(data_combined)

            # Mostrar una vista previa de los resultados del scraping
            st.success("Scraping completado.")
            st.dataframe(df_combined.head())

            # Si el usuario desea guardar el archivo
            if guardar_archivo and file_name_input:
                # Guardar en el directorio actual con el nombre proporcionado por el usuario
                output_path = f"{file_name_input}.csv"
                df_combined.to_csv(output_path, index=False)
                st.success(f"Archivo guardado como: {output_path}")
                st.write(f"Ubicación del archivo: {os.path.abspath(output_path)}")

# Opción 3: Analisis de datos
if selected == "Análisis de datos":
    # Cargar datos
    df = pd.read_csv('../data/fake_news_clean.csv')

    # Título de la aplicación
    st.title("Análisis de Noticias Falsas y Reales")

    # Distribución de noticias falsas vs reales
    st.subheader("Distribución de Noticias Falsas vs Reales")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(x='Category', data=df, ax=ax)
    ax.set_title('Distribución de Noticias Falsas vs Reales')
    ax.set_xlabel('Categoría (0: Fake, 1: Real)')
    ax.set_ylabel('Número de Noticias')
    st.pyplot(fig)

    # Longitud del texto
    st.subheader("Distribución de la Longitud del Texto (Fake vs Real)")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data=df, x='Text_Length', hue='Category', kde=True, bins=30, ax=ax)
    ax.set_title('Distribución de la Longitud del Texto (Fake vs Real)')
    ax.set_xlabel('Número de Caracteres en el Texto')
    ax.set_ylabel('Frecuencia')
    st.pyplot(fig)

    # Fuentes más comunes
    st.subheader("Principales Fuentes de Noticias Falsas y Reales")
    fake_sources = df[df['Category'] == 0]['Source'].value_counts().head(10)
    real_sources = df[df['Category'] == 1]['Source'].value_counts().head(10)

    # Gráfico para fuentes de noticias falsas
    st.markdown("##### Fuentes de Noticias Falsas")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=fake_sources.index, y=fake_sources.values, ax=ax)
    ax.set_title('Principales Fuentes de Noticias Falsas')
    ax.set_xlabel('Fuente de Noticias')
    ax.set_ylabel('Número de Noticias')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    st.pyplot(fig)

    # Gráfico para fuentes de noticias reales
    st.markdown("##### Fuentes de Noticias Reales")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=real_sources.index, y=real_sources.values, ax=ax)
    ax.set_title('Principales Fuentes de Noticias Reales')
    ax.set_xlabel('Fuente de Noticias')
    ax.set_ylabel('Número de Noticias')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    st.pyplot(fig)

    # Análisis de Sentimiento
    st.subheader("Distribución del Sentimiento en Noticias Falsas vs Reales")
    def get_sentiment(text):
        analysis = TextBlob(text)
        return analysis.sentiment.polarity

    df['Sentiment'] = df['Text'].apply(get_sentiment)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data=df, x='Sentiment', hue='Category', kde=True, ax=ax)
    ax.set_title('Distribución del Sentimiento (Fake vs Real)')
    ax.set_xlabel('Polaridad del Sentimiento')
    ax.set_ylabel('Frecuencia')
    st.pyplot(fig)