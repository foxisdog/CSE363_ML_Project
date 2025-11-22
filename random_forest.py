import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np

# 1. 데이터 불러오기
df = pd.read_csv('DED_Dataset_preprocessed.csv')

# 2. 학습용 데이터와 정답 데이터 나누기
X = df.drop('Dry Eye Disease', axis=1) # 예측에 쓸 데이터
y = df['Dry Eye Disease']              # 맞혀야 할 정답

# 3. 랜덤 포레스트 모델 생성
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# 4. k-fold 교차 검증으로 성능 확인 (k=5)
scores = cross_val_score(rf, X, y, cv=5, scoring='accuracy')

# 5. 결과 출력
print(f"각 fold 정확도: {scores}")
print(f"평균 정확도: {scores.mean():.2f}")
print(f"표준편차: {scores.std():.2f}")