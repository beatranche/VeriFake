# Proyecto de Análisis y Detección de Noticias Falsas

![image](https://github.com/user-attachments/assets/1ce0e450-ce64-4fb6-84b8-317675787ad8)

*Detección de Noticias Falsas en Español mediante Machine Learning y Procesamiento de Lenguaje Natural (NLP)*

Este proyecto permite analizar y detectar noticias falsas en español a través de modelos de *machine learning* y técnicas de *NLP*. Incluye funciones avanzadas de scraping para recolectar noticias desde diferentes fuentes, así como un generador de titulares tipo clickbait para probar el impacto de los encabezados en las noticias.

---

## Tabla de Contenidos
- [Características](#características)
- [Requisitos Previos](#requisitos-previos)
- [Instalación](#instalación)
- [Configuración del Entorno](#configuración-del-entorno)
- [Ejecución de la Aplicación](#ejecución-de-la-aplicación)
- [Uso de la Aplicación](#uso-de-la-aplicación)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Contribuciones](#contribuciones)

---

## Características

El proyecto está dividido en múltiples funciones interactivas que permiten:

1. **Clasificación de Noticias**: Analizar si una noticia es falsa o verdadera utilizando varios modelos de *machine learning*.
2. **Scraping de Noticias**: Recopilar noticias en español desde una fuente en línea, con opción de almacenamiento en archivos CSV.
3. **Análisis Exploratorio**: Visualizaciones de la longitud de las noticias, análisis de sentimiento, principales fuentes y otras características.
4. **Generador de Titulares**: Crear titulares tipo *clickbait* sobre un tema dado usando la API de OpenAI.

---

## Requisitos Previos

### Dependencias Principales:
- **Python 3.7 o superior**
- **Bibliotecas**:
    - `streamlit`, `pandas`, `matplotlib`, `seaborn`
    - `scikit-learn`, `xgboost`
    - `selenium` y un *WebDriver* compatible (como ChromeDriver)
    - `openai` para el generador de titulares

---

## Instalación

1. **Clonar el repositorio**:
    ```bash
    git clone https://github.com/tuusuario/fake-news-detection.git
    cd fake-news-detection
    ```

2. **Configurar entorno virtual**:
    ```bash
    python -m venv env
    source env/bin/activate  # En Linux o macOS
    env\Scripts\activate  # En Windows
    ```

3. **Instalar dependencias**:
    ```bash
    pip install -r requirements.txt
    ```

4. **Configurar API de OpenAI**:
   - Crear un archivo `.env` en el directorio raíz del proyecto y añadir tu clave de API de OpenAI:
    ```plaintext
    API_KEY=tu_clave_api_de_openai
    ```

5. **Configurar WebDriver**:
   - Instala un driver de Selenium compatible con tu navegador (ChromeDriver, GeckoDriver, etc.) y añade su ruta a tu PATH o colócala en el directorio del proyecto.

---

## Ejecución de la Aplicación

Para ejecutar la aplicación de Streamlit:
```bash
streamlit run app.py


