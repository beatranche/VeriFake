import streamlit as st
import pickle
from fakenews_pred.functions import TextPreprocessor, vectorize_text

# Función para cargar el modelo
@st.cache(allow_output_mutation=True)
def load_model():
    with open("../models/fake_news_model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

# Título de la aplicación
st.title("Predicción de Noticias Falsas")

# Cargar el modelo de noticias falsas
model = load_model()

# Entrada de la noticia por parte del usuario
st.subheader("Introduce una noticia para verificar:")
user_input = st.text_area("Escribe la noticia aquí...")

# Botón para realizar la predicción
if st.button("Predecir"):
    if user_input:
        # Preprocesar el texto de entrada
        preprocessor = TextPreprocessor()
        preprocessed_text = preprocessor.preprocess_text(user_input)
        
        # Vectorizar el texto de entrada (usando el mismo vectorizador que el modelo entrenado)
        vectorizer = pickle.load(open("../models/vectorizer.pkl", "rb"))
        user_input_vec = vectorizer.transform([preprocessed_text])
        
        # Realizar la predicción
        prediction = model.predict(user_input_vec)
        probability = model.predict_proba(user_input_vec)

        # Mostrar el resultado
        if prediction[0] == 1:
            st.error(f"⚠️ Esta noticia podría ser falsa. (Confianza: {probability[0][1]*100:.2f}%)")
        else:
            st.success(f"✅ Esta noticia parece verdadera. (Confianza: {probability[0][0]*100:.2f}%)")
    else:
        st.warning("Por favor, introduce una noticia para hacer la predicción.")