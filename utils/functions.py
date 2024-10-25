

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
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(nltk.corpus.stopwords.words('spanish'))
    
    def clean_text(self, text):
        # Convertir todo a minúsculas
        text = text.lower()
        # Eliminar URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        # Eliminar menciones (por ejemplo, @usuario)
        text = re.sub(r'@\w+', '', text)
        # Eliminar caracteres especiales y puntuación
        text = re.sub(r'\W', ' ', text)
        text = re.sub(r'\s+', ' ', text)  # Eliminar espacios múltiples
        return text.strip()

    def lemmatize_text(self, text):
        tokens = nltk.word_tokenize(text)
        lemmatized_tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        return ' '.join(lemmatized_tokens)

    def preprocess(self, df):
        df['Text'] = df['Text'].apply(self.lemmatize_text)
        return df


class ModelTrainer:
    def __init__(self):
        self.classifiers = {
            "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced'),
            "Random Forest": RandomForestClassifier(random_state=42, class_weight='balanced'),
            "SVM": SVC(kernel='linear', probability=True, class_weight='balanced'),
            "XGBoost": XGBClassifier(eval_metric='mlogloss', random_state=42, scale_pos_weight=1)  # Ajusta manualmente si se requiere
        }

    def train_models(self, X_train, y_train, X_test):
        predictions = {}
        probabilities = {}
        for name, clf in self.classifiers.items():
            clf.fit(X_train, y_train)
            predictions[name] = clf.predict(X_test)  # Predicciones
            probabilities[name] = clf.predict_proba(X_test)[:, 1]  # Probabilidades para la clase positiva
        return predictions, probabilities

    def evaluate_models(self, predictions, y_test):
        results = {}
        for name, y_pred in predictions.items():
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
    
    # Función para crear el Stacking Classifier
    def create_stacking_classifier(self, best_rf, best_svm, best_xgb):
        stacking_clf = StackingClassifier(
            estimators=[
                ('rf', best_rf),
                ('svm', best_svm),
                ('xgb', best_xgb)
            ],
            final_estimator=LogisticRegression(max_iter=1000),
            cv=5
        )
        return stacking_clf


def create_voting_classifier(models):
    return VotingClassifier(estimators=models, voting='soft')

def resample_data(X, y, method='SMOTE'):
    if method == 'SMOTE':
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
    elif method == 'ADASYN':
        adasyn = ADASYN(random_state=42)
        X_resampled, y_resampled = adasyn.fit_resample(X, y)
    elif method == 'undersampling':
        rus = RandomUnderSampler(random_state=42)
        X_resampled, y_resampled = rus.fit_resample(X, y)
    else:
        raise ValueError("Método no soportado. Usa 'SMOTE', 'ADASYN' o 'undersampling'.")
    return X_resampled, y_resampled


# Definición de la función para vectorizar texto
def vectorize_text(X_train, X_test, max_features=5000):
    vectorizer = TfidfVectorizer(max_features=max_features)  # Limitar a 5000 características
    X_train_vec = vectorizer.fit_transform(X_train['Text']).toarray()  # Convertir a matriz
    X_test_vec = vectorizer.transform(X_test['Text']).toarray()  # Transformar conjunto de prueba
    return X_train_vec, X_test_vec, vectorizer


def plot_roc_curve(y_true, y_scores, model_name):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'{model_name} AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Curva ROC')
    plt.legend(loc="lower right")
    plt.show()


def analyze_misclassifications(X_test, y_test, y_pred, model_name):
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
    ngrams_list = []
    for text in texts:
        tokens = nltk.word_tokenize(text)
        ngrams_list.extend(list(ngrams(tokens, n)))
    return ngrams_list


def analyze_sentiment(texts):
    sentiments = []
    for text in texts:
        analysis = TextBlob(text)
        sentiments.append((text, analysis.sentiment.polarity))
    return sentiments


def calculate_sentiment_percentages(sentiment_results):
    total = len(sentiment_results)
    positive = sum(1 for _, polarity in sentiment_results if polarity > 0)
    negative = sum(1 for _, polarity in sentiment_results if polarity < 0)
    neutral = total - (positive + negative)

    percentages = {
        'positive': (positive / total) * 100,
        'negative': (negative / total) * 100,
        'neutral': (neutral / total) * 100
    }
    return percentages


