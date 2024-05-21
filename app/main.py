""" from fastapi import FastAPI, File, UploadFile
import fitz  # PyMuPDF
import docx
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering
from fastapi.middleware.cors import CORSMiddleware
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
import re
from models.profile import Profile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import datetime as dt
import warnings
import plotly.graph_objects as go """
from fastapi import FastAPI
from app.api.api import api_router
from app.core.cors import configure_cors
from app.db.data_load import load_data

app = FastAPI()

#Configuración 
configure_cors(app)

#Rutas
app.include_router(api_router)

# Cargar y procesar datos al inicio
load_data()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


