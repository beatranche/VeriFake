# Proyecto de Análisis y Detección de Noticias Falsas

![Fake News Detection](https://upload.wikimedia.org/wikipedia/commons/4/47/Fake-news.png)
*Imagen ilustrativa de detección de noticias falsas (Fuente: Wikimedia Commons)*

Este proyecto permite analizar y detectar noticias falsas en español utilizando técnicas de *machine learning* y procesamiento de lenguaje natural (NLP). La aplicación también incluye funciones de scraping para recolectar noticias de diferentes fuentes, así como un generador de titulares tipo clickbait.

## Tabla de Contenidos
- [Características](#características)
- [Instalación](#instalación)
- [Ejecución de la Aplicación](#ejecución-de-la-aplicación)
- [Descripción de la Aplicación](#descripción-de-la-aplicación)
- [Uso de la Aplicación](#uso-de-la-aplicación)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Contribuciones](#contribuciones)

## Características

- **Detección de Fake News**: Clasificación de noticias en verdaderas o falsas mediante modelos de machine learning.
- **Scraping de Noticias**: Recolección de noticias de una fuente en línea con posibilidad de almacenamiento en archivo.
- **Análisis de Datos**: Visualización de características de los datos, incluyendo análisis de sentimiento, longitud de texto y fuentes principales.
- **Generador de Titulares Clickbait**: Genera titulares llamativos basados en un tema dado mediante la API de OpenAI.

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

## Ejecución de la Aplicación

Para ejecutar la aplicación de Streamlit:
```bash
streamlit run app.py

