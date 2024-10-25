

import pandas as pd
import numpy as np
import nltk
import re
from nltk import ngrams
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.decomposition import TruncatedSVD
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
from textblob import TextBlob

# Descargar recursos de NLTK
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

class TextPreprocessor:
    def __init__(self):
        """Initialize TextPreprocessor with a lemmatizer and stop words."""
        # Initialize the WordNet lemmatizer for reducing words to their base form
        self.lemmatizer = WordNetLemmatizer()
        
        # Load the set of Spanish stop words to filter out common words
        self.stop_words = set(nltk.corpus.stopwords.words('spanish'))
    
    def clean_text(self, text):
        """Clean a text by converting to lowercase, removing URLs, mentions, special characters, and punctuation.

        Args:
            text (str): The text to clean.

        Returns:
            str: The cleaned text.
        """
        # Convertir todo a minúsculas
        text = text.lower()
        # Eliminar URLs (por ejemplo, http://example.com)
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        # Eliminar menciones (por ejemplo, @usuario)
        text = re.sub(r'@\w+', '', text)
        # Eliminar caracteres especiales y puntuación (excepto espacios)
        text = re.sub(r'[^\w\s]', '', text)
        # Eliminar espacios múltiples y dejar solo un espacio entre palabras
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def lemmatize_text(self, text):
        """Lemmatize a text by tokenizing it and reducing each word to its base form.

        Args:
            text (str): The text to lemmatize.

        Returns:
            str: The lemmatized text.
        """
        # Tokenize the text into words
        tokens = nltk.word_tokenize(text)
        # Lemmatize each word
        lemmatized_tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        # Join the lemmatized words back together
        return ' '.join(lemmatized_tokens)

    def preprocess(self, df):
        """Preprocess a DataFrame by lemmatizing its text column.

        Args:
            df (pd.DataFrame): The DataFrame to preprocess.

        Returns:
            pd.DataFrame: The preprocessed DataFrame.
        """
        # Lemmatize the text in the 'Text' column
        df['Text'] = df['Text'].apply(self.lemmatize_text)
        return df


class ModelTrainer:
    def __init__(self):
        """
        Initialize ModelTrainer with a set of classifiers to train and evaluate.

        The classifiers are:
            - Logistic Regression
            - Random Forest
            - Support Vector Machine (SVM)
            - XGBoost

        The hyperparameters for each classifier are set to reasonable defaults.
        """
        self.classifiers = {
            "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced'),
            "Random Forest": RandomForestClassifier(random_state=42, class_weight='balanced'),
            "SVM": SVC(kernel='linear', probability=True, class_weight='balanced'),
            "XGBoost": XGBClassifier(eval_metric='mlogloss', random_state=42, scale_pos_weight=1)  # Ajusta manualmente si se requiere
        }

    def train_models(self, X_train, y_train, X_test):
        """
        Train all the classifiers in the `self.classifiers` dictionary using the given data.

        Args:
            X_train (numpy.ndarray or pandas.DataFrame): The feature matrix for the training data.
            y_train (numpy.ndarray or pandas.Series): The target vector for the training data.
            X_test (numpy.ndarray or pandas.DataFrame): The feature matrix for the test data.

        Returns:
            dict: A dictionary with the predictions for each classifier.
            dict: A dictionary with the probabilities for the positive class for each classifier.
        """
        predictions = {}
        probabilities = {}
        for name, clf in self.classifiers.items():
            clf.fit(X_train, y_train)  # Train the classifier
            predictions[name] = clf.predict(X_test)  # Predict the labels for the test data
            probabilities[name] = clf.predict_proba(X_test)[:, 1]  # Get the probabilities for the positive class
        return predictions, probabilities

    def evaluate_models(self, predictions, y_test):
        """
        Evaluate the performance of all the classifiers in the `self.classifiers` dictionary.

        Args:
            predictions (dict): A dictionary with the predictions for each classifier.
            y_test (numpy.ndarray or pandas.Series): The target vector for the test data.

        Returns:
            dict: A dictionary with the classification reports for each classifier.
        """
        results = {}
        for name, y_pred in predictions.items():
            # Calculate the classification report for the current classifier
            results[name] = classification_report(y_test, y_pred, zero_division=0)
            cm = confusion_matrix(y_test, y_pred)
            print(f"\n{name} Report:")
            print(classification_report(y_test, y_pred, zero_division=0))
            plt.figure(figsize=(6, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1], yticklabels=[0, 1])
            plt.title(f'Matriz de Confusión: {name}')
            plt.xlabel('Predicción')
            plt.ylabel('Real')
            plt.show()
        return results
    
    def grid_search_optimization(self, X_train, y_train):
        """
        Perform a grid search to optimize the hyperparameters of the Random Forest, SVM, and XGBoost models.

        Args:
            X_train (numpy.ndarray or pandas.DataFrame): The feature matrix for the training data.
            y_train (numpy.ndarray or pandas.Series): The target vector for the training data.

        Returns:
            tuple: A tuple containing the best estimators for the Random Forest, SVM, and XGBoost models.
        """
        # Hiperparámetros para Random Forest
        rf_params = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10]
        }

        # Hiperparámetros para SVM
        svm_params = {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf'],
            'gamma': ['scale', 'auto']
        }

        # Hiperparámetros para XGBoost
        xgb_params = {
            'n_estimators': [100, 200],
            'max_depth': [3, 6, 10],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.7, 1]
        }

        # Crear los modelos base
        rf = RandomForestClassifier(random_state=42)
        svm = SVC(probability=True, random_state=42)
        xgb = XGBClassifier(eval_metric='mlogloss', random_state=42)

        # Realizar la búsqueda de hiperparámetros usando GridSearchCV
        rf_grid = GridSearchCV(rf, rf_params, cv=5, scoring='accuracy', n_jobs=-1)
        svm_grid = GridSearchCV(svm, svm_params, cv=5, scoring='accuracy', n_jobs=-1)
        xgb_grid = GridSearchCV(xgb, xgb_params, cv=5, scoring='accuracy', n_jobs=-1)

        # Entrenar con los datos
        print("Optimizing Random Forest...")
        rf_grid.fit(X_train, y_train)

        print("Optimizing SVM...")
        svm_grid.fit(X_train, y_train)

        print("Optimizing XGBoost...")
        xgb_grid.fit(X_train, y_train)

        # Guardar los mejores estimadores
        best_rf = rf_grid.best_estimator_
        best_svm = svm_grid.best_estimator_
        best_xgb = xgb_grid.best_estimator_

        return best_rf, best_svm, best_xgb
    
    
    def create_stacking_classifier(self, best_rf, best_svm, best_xgb):
        """
        Create a stacking classifier using the provided best models.

        Args:
            best_rf: Best estimator for Random Forest from GridSearchCV.
            best_svm: Best estimator for SVM from GridSearchCV.
            best_xgb: Best estimator for XGBoost from GridSearchCV.

        Returns:
            StackingClassifier: A stacking classifier with the given base models
            and a logistic regression as final estimator.
        """
        # Initialize the stacking classifier with the best models
        stacking_clf = StackingClassifier(
            estimators=[
                ('rf', best_rf),  # Random Forest
                ('svm', best_svm),  # SVM
                ('xgb', best_xgb)  # XGBoost
            ],
            final_estimator=LogisticRegression(max_iter=1000),  # Final estimator
            cv=5  # Cross-validation folds
        )
        return stacking_clf


def create_voting_classifier(models):
    """
    Creates a voting classifier from a list of models.

    Parameters
    ----------
    models : list
        A list of tuples, where each tuple contains a model name and an instance of the model.

    Returns
    -------
    VotingClassifier
        A voting classifier with the given models.
    """
    return VotingClassifier(estimators=models, voting='soft')

def resample_data(X, y, method='SMOTE'):
    """
    Resample the dataset using the specified method to address class imbalance.

    Args:
        X (numpy.ndarray or pandas.DataFrame): The feature matrix.
        y (numpy.ndarray or pandas.Series): The target vector.
        method (str): The resampling technique to use. Options are 'SMOTE', 'ADASYN', or 'undersampling'.

    Returns:
        tuple: A tuple containing the resampled feature matrix and target vector.

    Raises:
        ValueError: If the provided method is not supported.
    """
    if method == 'SMOTE':
        # Use SMOTE for oversampling the minority class
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
    elif method == 'ADASYN':
        # Use ADASYN for adaptive synthetic sampling
        adasyn = ADASYN(random_state=42)
        X_resampled, y_resampled = adasyn.fit_resample(X, y)
    elif method == 'undersampling':
        # Use random undersampling to reduce the majority class
        rus = RandomUnderSampler(random_state=42)
        X_resampled, y_resampled = rus.fit_resample(X, y)
    else:
        # Raise an error if the method is not recognized
        raise ValueError("Método no soportado. Usa 'SMOTE', 'ADASYN' o 'undersampling'.")
    
    return X_resampled, y_resampled

def vectorize_text(X_train, X_test, max_features=5000):
    """
    Vectoriza el texto de los conjuntos de entrenamiento y prueba
    utilizando el algoritmo de vectorización TF-IDF.

    Parameters
    ----------
    X_train : pandas.DataFrame
        Conjunto de entrenamiento con la columna 'Text'.
    X_test : pandas.DataFrame
        Conjunto de prueba con la columna 'Text'.
    max_features : int, optional
        Número de características máximas a considerar (por defecto, 5000).

    Returns
    -------
    tuple
        Un tuple con los siguientes elementos:
            - X_train_vec : numpy.ndarray
                Matriz de características del conjunto de entrenamiento.
            - X_test_vec : numpy.ndarray
                Matriz de características del conjunto de prueba.
            - vectorizer : TfidfVectorizer
                Instancia del vectorizador TF-IDF.

    """
    vectorizer = TfidfVectorizer(max_features=max_features)  # Limitar a 5000 características
    X_train_vec = vectorizer.fit_transform(X_train['Text']).toarray()  # Convertir a matriz
    X_test_vec = vectorizer.transform(X_test['Text']).toarray()  # Transformar conjunto de prueba
    return X_train_vec, X_test_vec, vectorizer



def plot_roc_curve(y_true, y_scores, model_name):
    """
    Plotea la curva ROC para un modelo dado.

    Parameters
    ----------
    y_true : array-like
        Etiquetas verdaderas del conjunto de prueba.
    y_scores : array-like
        Puntuaciones predichas del modelo para el conjunto de prueba.
    model_name : str
        Nombre del modelo para mostrar en la leyenda.

    Returns
    -------
    None
    """
    fpr, tpr, _ = roc_curve(y_true, y_scores)  # Calcula la curva ROC
    roc_auc = auc(fpr, tpr)  # Calcula el área bajo la curva ROC

    plt.figure()  # Crea una figura nueva
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'{model_name} AUC = {roc_auc:.2f}')  # Plotea la curva ROC
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # Plotea la diagonal
    plt.xlim([0.0, 1.0])  # Establece los límites en el eje x
    plt.ylim([0.0, 1.05])  # Establece los límites en el eje y
    plt.xlabel('False Positive Rate')  # Establece el título del eje x
    plt.ylabel('True Positive Rate')  # Establece el título del eje y
    plt.title('Curva ROC')  # Establece el título de la figura
    plt.legend(loc="lower right")  # Muestra la leyenda en la esquina inferior derecha
    plt.show()  # Muestra la figura


def analyze_misclassifications(X_test, y_test, y_pred, model_name):
    """
    Identifica los ejemplos mal clasificados por un modelo y los muestra en una tabla.

    Parameters
    ----------
    X_test : pd.DataFrame
        Conjunto de prueba.
    y_test : pd.Series
        Etiquetas verdaderas del conjunto de prueba.
    y_pred : np.ndarray
        Puntuaciones predichas del modelo para el conjunto de prueba.
    model_name : str
        Nombre del modelo para mostrar en la tabla.

    Returns
    -------
    pd.DataFrame
        DataFrame con los ejemplos mal clasificados, incluyendo las columnas 'Text', 'Actual' y 'Predicted'.
    """
    misclassified_indices = np.where(y_test != y_pred)[0]
    misclassified_examples = pd.DataFrame({
        'Text': X_test.iloc[misclassified_indices],
        'Actual': y_test.iloc[misclassified_indices],
        'Predicted': y_pred[misclassified_indices]
    })

    print(f'\nEjemplos mal clasificados para {model_name}:')
    print(misclassified_examples.head(10))  # Muestra los primeros 10 ejemplos mal clasificados
    return misclassified_examples



def extract_ngrams(texts, n):
    """
    Extracts n-grams from a list of text strings.

    Parameters
    ----------
    texts : list of str
        List of text strings to extract n-grams from.
    n : int
        The size of the n-grams to extract (e.g. 2 for bigrams, 3 for trigrams, etc.).

    Returns
    -------
    list of tuple
        A list of tuples, where each tuple contains an n-gram extracted from the input texts.
    """
    ngrams_list = []
    for text in texts:
        # Tokenize the text into individual words
        tokens = nltk.word_tokenize(text)
        # Extract n-grams from the tokens
        ngrams_list.extend(list(ngrams(tokens, n)))
    return ngrams_list


def analyze_sentiment(texts):
    """
    Analyze the sentiment of a list of text strings using TextBlob.

    Parameters
    ----------
    texts : list of str
        List of text strings to analyze.

    Returns
    -------
    list of tuple
        A list of tuples, where each tuple contains the text string and its sentiment polarity.
    """
    sentiments = []
    for text in texts:
        # Create a TextBlob object from the text
        analysis = TextBlob(text)
        # Append a tuple containing the text and its sentiment polarity to the list
        sentiments.append((text, analysis.sentiment.polarity))
    return sentiments


def calculate_sentiment_percentages(sentiment_results):
    """
    Calculate the percentage of positive, negative and neutral sentiment in a list of sentiment results.

    Parameters
    ----------
    sentiment_results : list of tuple
        A list of tuples, where each tuple contains the text and its sentiment polarity.

    Returns
    -------
    dict
        A dictionary containing the percentages of positive, negative and neutral sentiment.
    """
    total = len(sentiment_results)
    positive = sum(1 for _, polarity in sentiment_results if polarity > 0)
    negative = sum(1 for _, polarity in sentiment_results if polarity < 0)
    neutral = total - (positive + negative)

    # Calculate the percentages
    percentages = {
        'positive': (positive / total) * 100,
        'negative': (negative / total) * 100,
        'neutral': (neutral / total) * 100
    }
    return percentages


