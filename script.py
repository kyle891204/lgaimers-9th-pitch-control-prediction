import pandas as pd
import numpy as np
import joblib
import os

def main():
    print("1. 평가 데이터 불러오기...")
    test = pd.read_csv('./data/test.csv')

    row_ids = test['row_id']
    X_test = test.drop(columns=['row_id'])

    print("2. 데이터 전처리 중 (Train과 완벽히 동일하게!)...")
    # 💡결측치 불일치 버그 해결: Test 데이터도 단순 -1로 채워 모델의 혼동을 막습니다. 
    # (실무에서는 LightGBM이 -1을 결측치 그룹으로 잘 묶어서 판단합니다)
    numeric_cols = X_test.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if X_test[col].isnull().any():
            X_test[col] = X_test[col].fillna(-1)

    # 💡투수별 과거 제구 성공률(Target Encoding) 적용
    pitcher_stat = joblib.load('./model/pitcher_stat.pkl')
    overall_mean = joblib.load('./model/overall_mean.pkl')

    X_test = pd.merge(X_test, pitcher_stat, on='pitcher_id', how='left')
    X_test['pitcher_control_rate'] = X_test['pitcher_control_rate'].fillna(overall_mean)

    # 범주형(Object) 결측치 및 타입 변환
    object_cols = X_test.select_dtypes(include=['object']).columns
    for col in object_cols:
        if X_test[col].isnull().any():
            X_test[col] = X_test[col].fillna('Unknown')
        X_test[col] = X_test[col].astype('category')

    print("3. 보정된 모델(Calibrated) 불러오기...")
    model = joblib.load('./model/model.pkl')

    print("4. 제구 성공 확률 추론 중...")
    pred_proba = model.predict_proba(X_test)[:, 1]

    print("5. 결과 저장 중...")
    os.makedirs('./output', exist_ok=True)
    pd.DataFrame({
        'row_id': row_ids,
        'control_success': pred_proba
    }).to_csv('./output/submission.csv', index=False)
    print("✅ 추론 파이프라인 무사 통과!")

if __name__ == '__main__':
    main()
