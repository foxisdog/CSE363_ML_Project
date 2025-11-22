import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load dataset
file_path = "Dry_Eye_Dataset.csv"
df = pd.read_csv(file_path)

# Encode categorical features
categorical_features = ['Gender', 'Sleep disorder', 'Wake up during night', 'Feel sleepy during day',
                        'Caffeine consumption', 'Alcohol consumption', 'Smoking', 'Medical issue',
                        'Ongoing medication', 'Smart device before bed', 'Blue-light filter',
                        'Discomfort Eye-strain', 'Redness in eye', 'Itchiness/Irritation in eye', 'Dry Eye Disease']
label_encoders = {}
for col in categorical_features:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Convert Blood Pressure into two separate numerical columns
df[['Systolic_BP', 'Diastolic_BP']] = df['Blood pressure'].str.split('/', expand=True).astype(float)
df.drop(columns=['Blood pressure'], inplace=True)

# Define numerical features
numerical_features = ['Age', 'Sleep duration', 'Sleep quality', 'Stress level', 'Heart rate', 'Daily steps',
                      'Physical activity', 'Height', 'Weight', 'Average screen time', 'Systolic_BP', 'Diastolic_BP']

# Standardize numerical features
scaler = StandardScaler()
df[numerical_features] = scaler.fit_transform(df[numerical_features])

# Split data into features and target
#X = df.drop(columns=['Dry Eye Disease'])
#y = df['Dry Eye Disease']

df.to_csv("Dry_Eye_Dataset_preprocessed.csv", index=False)

# 1. 범주형 변수들을 숫자로 Label Encoding 
# 2. 혈압 데이터를 수축/이완 두 개의 열로 분리 (’Blood pressure’ → ‘Systolic_BP', 'Diastolic_BP’)
# 3. 숫자형 변수들을 StandardScaler로 표준화

# 아 여기서 3번은 0~1범위로 normalize 한것은 아니고 평균값을 0으로 하여 표준화한 값입니다! 대략 -2~2 정도의 값을 가집니다