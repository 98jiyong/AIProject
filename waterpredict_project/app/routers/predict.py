from fastapi import APIRouter, HTTPException
from app.services.waterPredict02 import predict_water_electricity, get_forecast_data
import pandas as pd

router = APIRouter()

@router.get("/")
async def predict():
    """
    예측 데이터를 조회하고 반환합니다.
    """
    try:
        # 예측 수행
        predict_water_electricity()

        # 예측 데이터 조회
        forecast_data = get_forecast_data()

        # 데이터가 올바르게 반환되었는지 확인
        print(forecast_data.head())  # 디버깅을 위한 출력

        # pandas DataFrame을 JSON으로 변환
        return forecast_data.to_dict(orient='records')  # [{column -> value}, ...] 형식으로 변환
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
