#!/usr/bin/env python
# coding: utf-8

# In[4]:


import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# In[5]:


# 타이틀 설정
st.title('Iris 꽃 분류기')


# In[6]:


# 데이터 로드
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target_names[iris.target], name='species')


# In[7]:


# 사이드바 생성
st.sidebar.header('사용자 입력 파라미터')


# In[8]:


# 슬라이더로 하이퍼파라미터 조정
n_estimators = st.sidebar.slider('트리 개수', 1, 100, 10)
max_depth = st.sidebar.slider('최대 깊이', 1, 10, 3)


# In[9]:


# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[10]:


# 모델 학습
clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
clf.fit(X_train, y_train)


# In[11]:


# 정확도 계산
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)


# In[12]:


# 결과 표시
st.write(f'모델 정확도: {accuracy * 100:.2f}%')


# In[13]:


# 특징 중요도 시각화
st.subheader('특징 중요도')
feature_importance = pd.DataFrame({
    '특징': X.columns,
    '중요도': clf.feature_importances_
}).sort_values('중요도', ascending=False)

st.bar_chart(feature_importance.set_index('특징'))


# In[ ]:




