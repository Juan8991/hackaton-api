from fastapi import APIRouter
from app.api.endpoints import graph, predict

api_router = APIRouter()
api_router.include_router(predict.router, prefix="/predict", tags=["predict"])
api_router.include_router(graph.router, prefix="/graph", tags=["graph"])