from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import predict

app = FastAPI()

# CORS 미들웨어 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터 등록
app.include_router(predict.router, prefix="/predict", tags=["Prediction"])
