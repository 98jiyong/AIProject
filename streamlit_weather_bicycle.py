import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

# 한글 폰트 설정
mpl.rcParams['font.family'] = 'Malgun Gothic'  # 또는 'Apple Gothic' (macOS의 경우)
mpl.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

# Streamlit 애플리케이션 제목
st.title("서울시 공공자전거 이용 건수 예측")

# 데이터 로드
file_path = 'dataset/weather_bicycle.csv'
df = pd.read_csv(file_path, encoding='cp949')


# 독립변수와 종속변수 분리
X = df[['avg_temperature', 'avg_humidity', 'solar_radiation']]
y = df['count']

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 모델 학습
model = XGBRegressor(random_state=42)
model.fit(X_train, y_train)

# 예측
y_pred = model.predict(X_test)

# 평가
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# 결과 출력
st.markdown("<h5>사용한 모델 : XGBoost</h5>", unsafe_allow_html=True)
st.markdown(f"<h5>평균 제곱 오차 (MSE): {mse:.2f}</h5>", unsafe_allow_html=True)
st.markdown(f"<h5>결정 계수 (R² Score): {r2:.2f}</h5>", unsafe_allow_html=True)

st.subheader("선 그래프")
# 예측값과 실제값 시각화
plt.figure(figsize=(12, 6))
plt.plot(y_test.reset_index(drop=True), label='실제값', linestyle='-', color='blue')
plt.plot(y_pred, label='예측값', linestyle='-', color='orange')
plt.title('실제값 vs 예측값')
plt.xlabel('샘플')
plt.ylabel('대여건수')
plt.legend(loc='upper left')
plt.grid()
st.pyplot(plt)

st.subheader("산점도")
# 산점도
plt.figure(figsize=(12, 6))
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'b-', label='y=x 선')
plt.scatter(y_test.reset_index(drop=True), y_pred, color='orange', alpha=0.5, label='예측값')
plt.title('실제값 vs 예측값 산점도')
plt.xlabel('실제값')
plt.ylabel('예측값')
plt.legend(loc='upper left')
plt.grid()
st.pyplot(plt)

st.subheader("잔차 그래프")
# 잔차 그래프
residuals = y_test.reset_index(drop=True) - y_pred
plt.figure(figsize=(12, 6))
plt.axhline(0, color='blue', linewidth=2, label='기준선')
plt.scatter(y_pred, residuals, color='orange', label='잔차')
plt.title('잔차 그래프')
plt.xlabel('예측 값')
plt.ylabel('잔차')
plt.legend(loc='upper left')
plt.grid()
st.pyplot(plt)

# 새로운 데이터를 입력받기
st.subheader("기상정보에 따른 이용건수 예측하기")
avg_temperature = st.number_input("평균 기온 (°C)", min_value=-30.0, max_value=50.0, value=28.2)
avg_humidity = st.number_input("평균 습도 (%)", min_value=0.0, max_value=100.0, value=64.9)
solar_radiation = st.number_input("일사량 (MJ/m2)", min_value=0.0, value=23.62)

if st.button("예측하기"):
    # 새로운 데이터 스케일링
    new_data_scaled = np.array([[avg_temperature, avg_humidity, solar_radiation]])
    
    # 예측
    predicted_count = model.predict(new_data_scaled)
    
    # 결과 출력
    st.write(f"예상되는 공공자전거 이용 건수: {predicted_count[0]:.0f}건")

# CSV 파일로 저장
if st.button("CSV 파일로 저장"):
    df.to_csv('dataset/weather_bicycle.csv', index=False)
    st.success("CSV 파일이 저장되었습니다: weather_bicycle.csv")

