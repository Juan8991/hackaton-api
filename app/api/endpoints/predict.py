import nltk
from nltk.corpus import stopwords
import unicodedata
import re
import joblib
from fastapi import APIRouter

# Descargar las stopwords si no están ya descargadas
nltk.download('stopwords')

# Definir las stopwords en español
stop_words = set(stopwords.words('spanish'))

# Crear un router de FastAPI
router = APIRouter()

# Cargar modelos una vez al inicio
modelo_nlp = joblib.load("app/utils/predict/npl_data_science.pkl")
tfidf_new = joblib.load('app/utils/predict/tfidf_vectorizer.joblib')
mlb_new = joblib.load('app/utils/predict/multi_label_binarizer.joblib')

# Función para predecir etiquetas
def predict_labels(text, modelo_nlp, tfidf, mlb):
    preprocessed_text = preprocess_text(text)
    text_features = tfidf.transform([preprocessed_text])
    prediction = modelo_nlp.predict(text_features)
    return mlb.inverse_transform(prediction)

# Función para preprocesar texto
def preprocess_text(text):
    tokens = [word for word in text.split() if word.lower() not in stop_words]
    return ' '.join(tokens)

# Función para eliminar tildes
def delete_tildes(texto):
    texto = texto.lower()
    texto_normalize = unicodedata.normalize('NFD', texto)
    texto_sin_tildes = ''.join(char for char in texto_normalize if unicodedata.category(char) != 'Mn')
    texto_sin_tildes = unicodedata.normalize('NFC', texto_sin_tildes)
    return texto_sin_tildes

# Función para obtener nivel de experiencia
def get_experience_level(texto):
    pattern = r'(?:experiencia/s*de/s*)?(\d+)/s*años/s*de/s*experiencia'
    pattern_frases = r'(amplia experiencia|mucha experiencia|gran experiencia|sólida experiencia|amplia trayectoria|extensa experiencia|experiencia considerable|experiencia sustancial|experiencia profunda|amplia formación)'
    pattern_executive = r'(ejecutivo|directivo|manager|gerente)'

    match_level = re.search(pattern, texto)
    match_frases = re.search(pattern_frases, texto)
    match_executive = re.search(pattern_executive, texto)

    if match_executive:
        nivel_experiencia = "Executive"
    elif match_level:
        anos_experiencia = int(match_level.group(1))
        if anos_experiencia < 2:
            nivel_experiencia = "Entry-level"
        elif 2 <= anos_experiencia < 5:
            nivel_experiencia = "Mid-level"
        else:
            nivel_experiencia = "Senior"
    elif match_frases:
        nivel_experiencia = "Senior"
    else:
        nivel_experiencia = "NO aun"

    return nivel_experiencia

# Función para obtener tipo de empleo
def get_employment_type(texto):
    pattern_full_time = r'(tiempo completo|jornada completa|full-time)'
    pattern_part_time = r'(tiempo parcial|media jornada|part-time)'
    pattern_freelance = r'(freelance|independiente)'
    pattern_contractor = r'(contractor|contratista)'

    match_full_time = re.search(pattern_full_time, texto)
    match_part_time = re.search(pattern_part_time, texto)
    match_freelance = re.search(pattern_freelance, texto)
    match_contractor = re.search(pattern_contractor, texto)

    if match_full_time:
        return "Full-time"
    elif match_part_time:
        return "Part-time"
    elif match_freelance:
        return "Freelance"
    elif match_contractor:
        return "Contract"
    else:
        return "Full-time"
    
# Función para obtener entorno de trabajo
def get_work_setting(texto):
    pattern_remote = r'(remoto|remote|teletrabajo|distancia)'
    pattern_hybrid = r'(híbrido|hybrid)'
    pattern_in_person = r'(presencial|in-person|en oficina|on-site)'

    match_remote = re.search(pattern_remote, texto)
    match_hybrid = re.search(pattern_hybrid, texto)
    match_in_person = re.search(pattern_in_person, texto)

    if match_remote:
        return "Remote"
    elif match_hybrid:
        return "Hybrid"
    elif match_in_person:
        return "In-person"
    else:
        return "Remote"

# Ruta de predicción en FastAPI
@router.post("/nlp")
def predict(text: str) -> dict:
    text = delete_tildes(text)
    experiencia = get_experience_level(text)
    tipo_empleo = get_employment_type(text)
    work_setting = get_work_setting(text)
    etiqueta_predicha = predict_labels(text, modelo_nlp, tfidf_new, mlb_new)
    return {
        "experience_level": experiencia,
        "employment_type": tipo_empleo,
        "work_setting": work_setting,
        "job_category": etiqueta_predicha[0]
    }

nltk.download('stopwords')
stop_words = set(stopwords.words('spanish'))


router = APIRouter()



def predict_labels(text, modelo_nlp, tfidf, mlb):
    preprocessed_text = preprocess_text(text)
    text_features = tfidf.transform([preprocessed_text])
    prediction = modelo_nlp.predict(text_features)
    return mlb.inverse_transform(prediction)

def preprocess_text(text):
    tokens = [word for word in text.split() if word.lower() not in stop_words]
    return ' '.join(tokens)



def delete_tildes(texto):
    texto = texto.lower()
    texto_normalize = unicodedata.normalize('NFD', texto)
    texto_sin_tildes = ''.join(char for char in texto_normalize if unicodedata.category(char) != 'Mn')
    texto_sin_tildes = unicodedata.normalize('NFC', texto_sin_tildes)

    return texto_sin_tildes


def get_experience_level(texto):

    pattern = r'(?:experiencia/s*de/s*)?(/d+)/s*anos/s*de/s*experiencia'
    pattern_frases = r'(amplia experiencia|mucha experiencia|gran experiencia|solida experiencia|amplia trayectoria|extensa experiencia|experiencia considerable|experiencia sustancial|experiencia profunda|amplia formacion)'
    pattern_executive = r'(ejecutivo|directivo|manager|gerente)'


    match_level = re.search(pattern, texto)
    print(match_level)
    match_frases = re.search(pattern_frases, texto)
    match_executive = re.search(pattern_executive, texto)

    if match_executive:
      nivel_experiencia = "Executive"

    elif match_level:
        anos_experiencia = int(match_level.group(1))
        print(anos_experiencia)

        if anos_experiencia < 2:
            nivel_experiencia = "Entry-level"
        elif 2 <= anos_experiencia < 5:
            nivel_experiencia = "Mid-level"
        else:
            nivel_experiencia = "Senior"

    elif match_frases:
        nivel_experiencia = "Senior"
    else:
        nivel_experiencia = "NO aun"

    return nivel_experiencia

def get_employment_type(texto):
    pattern_full_time = r'(tiempo completo|jornada completa|full-time)'
    pattern_part_time = r'(tiempo parcial|media jornada|part-time)'
    pattern_freelance = r'(freelance|independiente)'
    pattern_contractor = r'(contractor|contratista)'

    match_full_time = re.search(pattern_full_time, texto)
    match_part_time = re.search(pattern_part_time, texto)
    match_freelance = re.search(pattern_freelance, texto)
    match_contractor = re.search(pattern_contractor, texto)

    if match_full_time:
        return "Full-time"
    elif match_part_time:
        return "Part-time"
    elif match_freelance:
        return "Freelance"
    elif match_contractor:
        return "Contract"
    else:
        return "Full-time"
    
def get_work_setting(texto):

    pattern_remote = r'(remoto|remote|teletrabajo|distancia)'
    pattern_hybrid = r'(hibrido|hybrid)'
    pattern_in_person = r'(presencial|in-person|en oficina|on-site)'

    match_remote = re.search(pattern_remote, texto)
    match_hybrid = re.search(pattern_hybrid, texto)
    match_in_person = re.search(pattern_in_person, texto)

    if match_remote:
        return "Remote"
    elif match_hybrid:
        return "Hybrid"
    elif match_in_person:
        return "In-person"
    else:
        return "Remote"
    

@router.post("/nlp")
def predict(text: str) -> dict:
    text = delete_tildes(text)
    experiencia = get_experience_level(text)
    # return func.HttpResponse(f"experiencia, {experiencia}")
    tipo_empleo = get_employment_type(text)
    work_setting = get_work_setting(text)
    modelo_nlp = joblib.load("app/utils/predict/npl_data_science.pkl")
    tfidf_new = joblib.load('app/utils/predict/tfidf_vectorizer.joblib')
    mlb_new = joblib.load('app/utils/predict/multi_label_binarizer.joblib')
    etiqueta_predicha = predict_labels(text, modelo_nlp, tfidf_new, mlb_new)
    return {
        "experience_level": experiencia,
        "employment_type": tipo_empleo,
        "work_setting": work_setting,
        "job_category": etiqueta_predicha[0]

    }