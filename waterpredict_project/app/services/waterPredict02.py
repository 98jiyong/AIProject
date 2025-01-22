import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
import sqlalchemy
from sqlalchemy import create_engine, text

def predict_water_electricity():
    # 모델 로드
    model = load_model('app/model/lstm_model.h5')

    # 데이터 로드
    df_concat = pd.read_csv('app/dataset/watertot.csv', encoding="CP949")
    df_concat['일자'] = pd.to_datetime(df_concat['일자'])

    # 데이터 전처리
    df_concat = df_concat[(df_concat['총유입수량'] - df_concat['총유입수량'].mean()).abs() < 3 * df_concat['총유입수량'].std()]
    df_concat = df_concat[(df_concat['전력량'] - df_concat['전력량'].mean()).abs() < 3 * df_concat['전력량'].std()]
    df_concat = df_concat.sort_values(by='일자')

    # 표준화
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_concat[['총유입수량', '전력량']])

    # 시계열 데이터 생성 파라미터
    seq_length = 7

    # 마지막 시퀀스 추출
    last_sequence = scaled_data[-seq_length:, :-1]
    last_sequence = np.expand_dims(last_sequence, axis=0)

    # 1주일 예측
    future_predictions = []
    for _ in range(7):
        pred_scaled = model.predict(last_sequence)
        future_predictions.append(pred_scaled[0, 0])
        new_sequence = np.append(last_sequence[:, 1:, :], pred_scaled.reshape(1, 1, -1), axis=1)
        last_sequence = new_sequence

    # 예측값 역변환
    future_predictions_scaled = []
    for pred in future_predictions:
        scaled_value = scaler.inverse_transform(
            np.concatenate((scaled_data[-1:, :-1], np.array([[pred]])), axis=1)
        )[:, -1][0]
        future_predictions_scaled.append(scaled_value)

    # 1주일 예측 결과를 DataFrame으로 정리
    future_dates = pd.date_range(start=df_concat['일자'].iloc[-1] + pd.Timedelta(days=1), periods=7, freq='D')
    future_df = pd.DataFrame({
        'Date': future_dates,
        'Predicted_wattage': future_predictions_scaled
    })

    # DBMS에 저장
    engine = create_engine("mysql+pymysql://root:8948864a@localhost:3306/test?charset=utf8")
    
    # Date 형식을 'YYYY-MM-DD'로 변환
    future_df['Date'] = future_df['Date'].dt.strftime('%Y-%m-%d')

    # 데이터 타입 정의
    dtypesql = {
        'Date': sqlalchemy.types.VARCHAR(20),
        'Predicted_wattage': sqlalchemy.types.DECIMAL(10, 2)            
    }

    # 데이터 저장
    future_df.to_sql(name="elec_forecast", con=engine, if_exists='replace', index=False, dtype=dtypesql)

    return future_df

def get_forecast_data():
    engine = create_engine("mysql+pymysql://root:8948864a@localhost:3306/test?charset=utf8")
    conn = engine.connect()
    
    result = conn.execute(text("SELECT * FROM elec_forecast")).fetchall()
    conn.close()
    
    return pd.DataFrame(result, columns=['Date', 'Predicted_wattage'])

