
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


