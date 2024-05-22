import nltk
import unicodedata
import re
import joblib
import fitz  # PyMuPDF
import docx
import numpy as np
from nltk.corpus import stopwords
from sklearn.preprocessing import StandardScaler
from fastapi import APIRouter, File, UploadFile
from app.models.profile import Profile
from app.models.profilepydantic import Profilepy
from app.db.data_load import getDataFrameOriginal


# Descargar las stopwords si no están ya descargadas
nltk.download('stopwords')

# Definir las stopwords en español
stop_words = set(stopwords.words('spanish'))

# Crear un router de FastAPI
router = APIRouter()

# Cargar modelos una vez al inicio
#Modelo NLP, vectorizador y label converter
modelo_nlp = joblib.load("app/utils/predict/npl_data_science.pkl")
tfidf_new = joblib.load('app/utils/predict/tfidf_vectorizer.joblib')
mlb_new = joblib.load('app/utils/predict/multi_label_binarizer.joblib')
#Modelo de Regresión
model_regressor = joblib.load(r'app/utils/predict/xgboost_regressor.pkl')
encoder_regressor = joblib.load(r'app/utils/predict/label_encoders_regressor.joblib')
data = getDataFrameOriginal()
# Seleccionar la columna que deseas escalar
Y = data['salary_in_usd'].values.reshape(-1, 1)  # Reformar a (n_samples, 1)

# Crear el StandardScaler
scaler = StandardScaler()

# Ajustar y transformar los datos
Y_scaled = scaler.fit_transform(Y)
#Modelo de Clasificación
model_classifier= joblib.load(r'app/utils/predict/xgboost_classifier.pkl')
encoder_clasifier = joblib.load(r'app/utils/predict/label_encoders_classifier.joblib')

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

    # pattern = r'(\d+)\s+anos\s+de\s+experiencia|experiencia\s+de\s+(\d+)\s+anos'
    pattern = r'(\d+)\s+(?:anos|años)\s+de\s+experiencia|experiencia\s+de\s+(\d+)\s+(?:anos|años)'
    pattern_frases = r'(amplia experiencia|mucha experiencia|gran experiencia|solida experiencia|amplia trayectoria|extensa experiencia|experiencia considerable|experiencia sustancial|experiencia profunda|amplia formacion)'
    pattern_executive = r'(ejecutivo|directivo|manager|gerente)'

    match_level = re.search(pattern, texto)
    match_frases = re.search(pattern_frases, texto)
    match_executive = re.search(pattern_executive, texto)

    if match_executive:
        nivel_experiencia = "Executive"

    elif match_level:
        if match_level.group(1) is not None:
            anos_experiencia = int(match_level.group(1))
        elif match_level.group(2) is not None:
            anos_experiencia = int(match_level.group(2))

        if anos_experiencia < 2:
            nivel_experiencia = "Entry-level"
        elif 2 <= anos_experiencia < 5:
            nivel_experiencia = "Mid-level"
        else:
            nivel_experiencia = "Senior"

    elif match_frases:
        nivel_experiencia = "Senior"
    else:
        nivel_experiencia = "Mid-level"

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
# @router.post("/nlp")
def predict(text: str) -> dict:
    print(text)
    text = delete_tildes(text)
    experiencia = get_experience_level(text)
    tipo_empleo = get_employment_type(text)
    work_setting = get_work_setting(text)
    etiqueta_predicha = predict_labels(text, modelo_nlp, tfidf_new, mlb_new)
    profile =  Profile()
    profile.experience_level= experiencia
    profile.employment_type= tipo_empleo
    profile.work_setting= work_setting
    profile.job_category= etiqueta_predicha[0]
    profile.employee_residence=''
    return profile
    
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text()
    return text

def extract_text_from_word(docx_path):
    doc = docx.Document(docx_path)
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text

def predictLocation(x_prueba, modelo,encoder,salario=np.nan):
    transformed_array = []
    for value,column in zip(x_prueba,['job_category','employee_residence','experience_level','employment_type', 'work_setting','company_size']):
        transformed_value = encoder[column].transform([value])[0]
        transformed_array.append(transformed_value)

    transformed_array.insert(1,salario)
    y_pred = modelo.predict(np.array(transformed_array).reshape(1, -1))
    return encoder['company_location'].inverse_transform(y_pred)[0]

def predictSalary(x_prueba, model,encoder):
    transformed_array = []
    for value,column in zip(x_prueba,['job_category','employee_residence','experience_level','employment_type','company_location','work_setting','company_size']):
        transformed_value = encoder[column].transform([value])[0]
        transformed_array.append(transformed_value)


    transformed_array
    y_pred = model.predict(np.array(transformed_array).reshape(1, -1))
    return scaler.inverse_transform([y_pred])



@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if file.content_type == "application/pdf":
        with open(file.filename, "wb") as f:
            f.write(file.file.read())
        text = extract_text_from_pdf(file.filename)
        text = preprocess_text(text)
    elif file.content_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        with open(file.filename, "wb") as f:
            f.write(file.file.read())
        text = extract_text_from_word(file.filename)
        text = preprocess_text(text)
    else:
        return {"error": "Formato de archivo no soportado"}
    
    return predict(text)
    


@router.post("/result")
def save_profile(prof: Profilepy):
    text_classification = [prof.job_category,prof.employee_residence,prof.experience_level,prof.employment_type,prof.work_setting,prof.company_size]
    location = predictLocation(text_classification,model_classifier,encoder_clasifier,0)
    text_regression = [prof.job_category,prof.employee_residence,prof.experience_level,prof.employment_type,prof.employee_residence,prof.work_setting,prof.company_size]
    salary= predictSalary(text_regression,model_regressor,encoder_regressor)
    print(salary)
    return {"location":location,
            "salary":salary[0][0]}

    """ text_classification = [prof.job_category,prof.employee_residence,prof.experience_level,prof.employment_type,prof.work_setting,prof.company_size]
    text_regression = [prof.job_category,prof.employee_residence,prof.experience_level,prof.employment_type,"",prof.work_setting,prof.company_size]
    
    return {
        "salary": predictSalary(text_regression,model_regressor,encoder_regressor),
        "location": predictLocation(text_classification,model_classifier,encoder_clasifier,0)
    } """