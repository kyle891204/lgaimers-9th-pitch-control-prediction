import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import os

def main():
    print("1. 평가 데이터 불러오기...")
    # 🚨 평가 서버 환경에 맞춰 상대 경로 지정
    test = pd.read_csv('./data/test.csv')

    # 제출 양식에 넣을 row_id를 따로 빼두고, 모델에 넣을 피처(X_test)만 남깁니다.
    row_ids = test['row_id']
    X_test = test.drop(columns=['row_id'])

    print("2. 데이터 전처리 진행 중...")
    # 숫자형(Numeric) 결측치 처리: 에러 방지를 위해 단순 0으로 채웁니다.
    # (원래는 Train 데이터의 평균을 쓰는 것이 정석입니다.)
    numeric_cols = X_test.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if X_test[col].isnull().any():
            X_test[col] = X_test[col].fillna(0)

    # 범주형(Object) 결측치 처리 및 LightGBM용 category 타입 변환
    object_cols = X_test.select_dtypes(include=['object']).columns
    for col in object_cols:
        if X_test[col].isnull().any():
            X_test[col] = X_test[col].fillna('Unknown')
        X_test[col] = X_test[col].astype('category')

    print("3. 나만의 LightGBM 모델 불러오기...")
    # 우리가 훈련시켜서 ./model/ 폴더에 저장해둔 모델을 깨웁니다.
    model = joblib.load('./model/model.pkl')

    print("4. 제구 성공 확률 예측 중...")
    # predict_proba를 사용하여 0~1 사이의 확률값을 뽑아냅니다.
    # [:, 1]은 '제구 실패(0)', '성공(1)' 중 '성공(1)'에 대한 확률만 가져온다는 뜻입니다.
    pred_proba = model.predict_proba(X_test)[:, 1]

    print("5. 예측 결과 저장 중...")
    # 서버 요구사항에 맞게 output 폴더를 만들고 그 안에 submission.csv를 저장합니다.
    os.makedirs('./output', exist_ok=True)
    submission = pd.DataFrame({
        'row_id': row_ids,
        'control_success': pred_proba
    })
    submission.to_csv('./output/submission.csv', index=False)
    print("모든 추론 과정이 완료되었습니다")

if __name__ == '__main__':
    main()
