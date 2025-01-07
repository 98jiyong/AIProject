#!/usr/bin/env python
# coding: utf-8

# ### 필요한 라이브러리 설치

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


# 폰트지정
plt.rc('font', family='Malgun Gothic')

# 마이너스 부호 깨짐 지정
plt.rcParams['axes.unicode_minus'] = False


# ### 데이터 로드

# In[3]:


# 2023년 1월 ~ 12월 서울시 공공자전거 이용 데이터
# 2023년 1월 ~ 12월 서울시 기상 데이터
file_path1 = 'dataset/seoul_bicycle_daily_2023.csv'
file_path2 = 'dataset/OBS_ASOS_DD_2023.csv'
df1 = pd.read_csv(file_path1, encoding='cp949')
df2 = pd.read_csv(file_path2, encoding='cp949')


# ### 데이터셋 확인
# - 데이터타입
# - 결측치 
# - 이상치

# In[4]:


# 데이터타입 및 결측치 확인
df1.info()


# In[5]:


df1.head()


# In[6]:


df1.tail()


# In[7]:


# 데이터타입 확인(자세히)
# 대여일시, 대여건수 -> object타입
print(df1.dtypes)


# In[8]:


# 결측치 확인(자세히) -> 결측치 없음
df1.isnull().sum()


# In[9]:


# 이상치 확인을 위해 [대여건수] 컬럼을 정수 타입으로 변경
# 쉼표 제거 및 정수형 변환
df1['대여건수'] = df1['대여건수'].str.replace(',','').astype(int)


# In[10]:


# 변환된 데이터타입 확인
print(df1.dtypes)


# In[11]:


# 이상치 확인 -> 이상치 없음
df1.boxplot(column=['대여건수'])
plt.show()


# In[12]:


# 데이터타입 및 결측치 확인
df2.info()


# In[13]:


df2.head()


# In[14]:


df2.tail()


# In[15]:


# 데이터타입 확인(자세히)
# 대여일시, 대여건수 -> object타입
print(df2.dtypes)


# In[16]:


# 결측치 확인
# 기상 데이터에는 일강수량, 일 최심적설에 결측치가 존재
df2.isnull().sum()


# In[17]:


# 이상치 확인
df2.boxplot(column=['평균기온(°C)','일강수량(mm)','평균 상대습도(%)','합계 일사량(MJ/m2)','일 최심적설(cm)'])
plt.show()


# ### 데이터 전처리
# - 불필요한 컬럼 제거
# - 컬럼명 변경
# - 데이터타입 변환
# - 결측치 처리
# - 이상치 제거

# In[18]:


# 컬럼명 변경
df1.rename(columns={'대여일시': 'date', '대여건수': 'count'}, inplace=True)
print(df1.columns)


# In[19]:


# 대여일 데이터타입 변경
df1['date'] = pd.to_datetime(df1['date'])
print(df1.dtypes)


# In[20]:


# 불필요한 데이터 컬럼 제거(지점, 지점명 불필요)
df2 = df2.iloc[: , 2: ]
df2.info()


# In[21]:


# 컬럼명 변경
# date = 일시, avg_temperature = 평균기온(°C), precipitation = 일강수량(mm), 
# avg_humidity = 평균 상대습도(%), solar_radiation = 합계 일사량(MJ/m2), snow_depth = 일 최심적설(cm)

df2.rename(columns={
    '일시': 'date',
    '평균기온(°C)': 'avg_temperature',
    '일강수량(mm)': 'precipitation',
    '평균 상대습도(%)': 'avg_humidity',
    '합계 일사량(MJ/m2)': 'solar_radiation',
    '일 최심적설(cm)': 'snow_depth'
}, inplace=True)
print(df2.columns)


# In[22]:


# 데이터타입 변경
df2['date'] = pd.to_datetime(df2['date'])


# In[23]:


df2.info()


# In[24]:


# 결측지 확인
df2.isna().sum()


# In[25]:


# 결측치 처리(평균값 대체)
df2.fillna({
    'precipitation': df2['precipitation'].mean(),
    'snow_depth': df2['snow_depth'].mean()
}, inplace=True)

# 0값 처리
df2['precipitation'] = df2['precipitation'].replace(0, df2['precipitation'].mean())
df2['snow_depth'] = df2['snow_depth'].replace(0, df2['snow_depth'].mean())


# In[26]:


# 결측치 확인
df2.isna().sum()


# In[27]:


# 이상치 확인
# 상대습도를 제외한 강수량, 적설량 데이터는 계절적 특성으로인해 이상치로 분류됨됨

df2.boxplot(column=['avg_temperature','precipitation','avg_humidity','solar_radiation','snow_depth'])
plt.show()


# In[28]:


# 6월부터 8월까지의 데이터 추출
filtered_data = df2[(df2['date'].dt.month >= 6) & (df2['date'].dt.month <= 8)]

# 날짜별로 강수량 집계
daily_precipitation = filtered_data.groupby('date')['precipitation'].sum().reset_index()

# 평균값보다 큰 데이터만 추출
above_average_data = daily_precipitation[daily_precipitation['precipitation'] > df2['precipitation'].mean()]

# 시각화 (막대그래프)
plt.figure(figsize=(12, 6))
precipitation_bars = plt.bar(above_average_data['date'], above_average_data['precipitation'], color='blue', alpha=0.7)
plt.title('6월 ~ 8월 일별 강수량 (단위 : mm)')
plt.xlabel('날짜')
plt.ylabel('강수량(mm)')
plt.grid(axis='y')

# 막대 위 레이블 추가
for bar in precipitation_bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), ha='center', va='bottom')

# 막대 아레 날짜 레이블 설정
plt.xticks(above_average_data['date'], rotation=45, ha='right')
plt.tight_layout()
plt.show()


# In[29]:


# 이상치 확인
# 계절적 패턴으로 인해 생기는 이상치였음을 확인
above_average_data.boxplot(column = ['precipitation'])
plt.show()


# In[30]:


# 12, 1, 2월의 데이터 추출
filtered_snow_data = df2[(df2['date'].dt.month == 12) | (df2['date'].dt.month == 1) | (df2['date'].dt.month == 2)]

# 날짜별로 눈 깊이 집계
daily_snow_depth = filtered_snow_data.groupby('date')['snow_depth'].sum().reset_index()

# 평균값보다 큰 데이터만 추출
above_average_snow_data = daily_snow_depth[daily_snow_depth['snow_depth'] > daily_snow_depth['snow_depth'].mean() + 0.1]

# 시각화 (막대그래프)
plt.figure(figsize=(12, 6))

# 균일한 간격으로 x축 설정
x_positions = np.arange(len(above_average_snow_data))  # x축 위치 설정

# 막대그래프 생성
snow_depth_bars = plt.bar(x_positions, above_average_snow_data['snow_depth'], color='blue', alpha=0.7)

# x축 레이블 설정
plt.xticks(x_positions, above_average_snow_data['date'].dt.strftime('%Y-%m-%d'), rotation=45, ha='right')

plt.title('12, 1, 2월 일별 적설량 (단위 : cm)')
plt.xlabel('날짜')
plt.ylabel('적설량(cm)')
plt.grid(axis='y')

# 막대 위 레이블 추가
for bar in snow_depth_bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), ha='center', va='bottom')
    
plt.tight_layout()
plt.show()


# In[31]:


# 12, 1, 2월 데이터 추출
filtered_data = df2[(df2['date'].dt.month == 12) | (df2['date'].dt.month == 1) | (df2['date'].dt.month == 2)]

# 평균값 수치 제외
filtered_data = filtered_data[filtered_data['snow_depth'] > df2['snow_depth'].mean() + 0.1]

# 기초 통계량 계산
statistics = filtered_data['snow_depth'].describe()

# 결과 출력
print(statistics)

# 시각화 (히스토그램)
plt.figure(figsize=(10, 6))
plt.hist(filtered_data['snow_depth'], bins=30, color='blue', alpha=0.7)
plt.title('12, 1, 2월 적설량 빈도수')
plt.xlabel('적설량(cm)')
plt.ylabel('빈도')
plt.grid()
plt.show()


# In[32]:


filtered_data.boxplot(column = ['snow_depth'])
plt.show()


# In[33]:


# 이상치 처리(상대습도)
Q1 = df2['avg_humidity'].quantile(0.25)
Q3 = df2['avg_humidity'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# 이상치를 상한 및 하한으로 클리핑
df2['avg_humidity'] = df2['avg_humidity'].clip(lower_bound, upper_bound)


# ## 특성 중요도 평가
# ##### 사용하고자 하는 특성이 예측 결과에 얼마나 영향을 미치는지 확인

# In[34]:


# 데이터셋 병합(기상 데이터 + 공공자전거 데이터)
merge_data = pd.merge(df1, df2, on='date', how='inner')


# In[35]:


merge_data.info()


# In[36]:


merge_data.head()


# In[37]:


merge_data.tail()


# In[38]:


corr = merge_data.corr()

# 히트맵 그리기
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap='Blues', fmt='.2f')
plt.title('상관관계 히트맵')
plt.show()


# In[39]:


# 독립변수와 종속변수 분리
# 독립변수 (기온, 습도, 일사량, 강수량, 적설량)
# 종속변수 (이용건수)
X = merge_data.drop(['count','date'], axis=1)
y = merge_data['count']

# 데이터 분할 (비율 7:3)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[40]:


# 랜덤 포레스트 모델 생성 및 학습
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)


# In[41]:


# 특성 중요도 생성
importances = model.feature_importances_

# 특성 이름과 중요도를 데이터프레임으로 변환
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importances
})


# In[42]:


# 중요도를 내림차순으로 정렬
feature_importance = feature_importance.sort_values(by='Importance', ascending=False)

# 시각화
plt.figure(figsize=(10, 6))
sns.barplot(x='Feature',y='Importance', data=feature_importance)
plt.title('특성 중요도')
plt.xlabel('특성')
plt.ylabel('중요도')
plt.show()


# In[43]:


# 특성중요도가 낮은 강수량, 적설량을 데이터셋에서 제거
df = merge_data.drop(['precipitation', 'snow_depth'], axis=1)


# In[44]:


df.info()


# In[45]:


df.head()


# In[46]:


df.tail()


# ### ML
# - 선형 회귀
# - 랜덤 포레스트 회귀
# - XGBoost

# ### 선형 회귀

# In[47]:


# 데이터 전처리
X = df[['avg_temperature', 'avg_humidity', 'solar_radiation']]
y = df['count']

# 데이터 스케일링
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)


# In[48]:


# 모델 정의
models = {
    '선형 회귀': LinearRegression(),
    '랜덤 포레스트': RandomForestRegressor(random_state=42),
    'XGBoost': XGBRegressor(random_state=42)
}

# R-squared 점수 저장
r2_scores = {}

# 각 모델 학습 및 R-squared 점수 계산
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    r2_scores[name] = r2


# In[49]:


# 시각화(선형 회귀 / 랜덤 포레스트 회귀 / XGBoost의 R-squared 값)
# XGBoost모델 채택
plt.figure(figsize=(6, 5))
bars = plt.bar(r2_scores.keys(), r2_scores.values(), color=['blue', 'orange', 'green'])
plt.title('예측모델 결정계수 비교')
plt.ylabel('r-squared')
plt.ylim(0, 1) 
plt.grid(axis='y')

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.2f}', ha='center', va='bottom')
    
plt.show()


# ### XGBoost

# In[50]:


# 데이터 전처리
X = df[['avg_temperature', 'avg_humidity', 'solar_radiation']]
y = df['count']

# 데이터 스케일링
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)


# In[51]:


# XGBoost 모델 생성 및 학습
model = XGBRegressor(random_state=42)
model.fit(X_train, y_train)

# 예측
y_pred = model.predict(X_test)

# 평가
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse:.2f}')
print(f'R² Score: {r2:.2f}')


# In[52]:


# 시각화 (선 그래프)
# 예측값 선 그래프가 실제값 선 그래프와 일치할수록 예측이 정확함
plt.figure(figsize=(12, 6))
plt.plot(y_test.reset_index(drop=True), label='실제값', linestyle='-', color='blue')
plt.plot(y_pred, label='예측값', linestyle='-', color='orange')
plt.title('실제값 vs 예측값')
plt.xlabel('독립 변수')
plt.ylabel('종속 변수')
plt.legend()
plt.grid()
plt.show()


# In[53]:


# 시각화 (산점도)
# 실제값 선에 점들이 가깝게 위치할수록 데이터를 예측이 정확함
plt.figure(figsize=(12, 6))
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'b-', label='실제값')  # y=x 선
plt.scatter(y_test, y_pred, color='orange', label='예측값')
plt.title('실제값 vs 예측값')
plt.xlabel('x')
plt.ylabel('y')
plt.grid()
plt.legend()
plt.show()


# In[54]:


# 시각화 (잔차 그래프)
# 잔차가 0을 중심으로 고르르게 분포하면 모델이 잘 작동하고 있음을 의미함
residuals = y_test.reset_index(drop=True) - y_pred
plt.figure(figsize=(12, 6))
plt.axhline(0, color='blue', linewidth=2, label='기준선')
plt.scatter(y_pred, residuals, color='orange', label='잔차')
plt.title('잔차 그래프')
plt.xlabel('예측 값')
plt.ylabel('잔차')
plt.legend()
plt.grid()
plt.show()


# In[55]:


# 새로운 데이터 입력받기 (2023-06-25 데이터 입력)
# 기온 : 28.2 / 습도 : 64.9 / 일사량 : 23.62
# 실제 이용건수 : 133,556 / 예측 이용건수 : 188243
new_data = {
    'avg_temperature': float(28.2),
    'avg_humidity': float(64.9),
    'solar_radiation': float(23.62)
}

# 새로운 데이터 스케일링
new_data_scaled = scaler.transform([[new_data['avg_temperature'], new_data['avg_humidity'], new_data['solar_radiation']]])

# 예측
predicted_count = model.predict(new_data_scaled)

# 결과 출력
print("실제 공공자전거 이용 건수 : 133556건")
print(f"예상되는 공공자전거 이용 건수: {predicted_count[0]:.0f}건")


# In[56]:


# CSV 파일로 저장
df.to_csv('dataset/weather_bicycle.csv')

print("저장완료.")

