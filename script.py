import pandas as pd
import numpy as np
import joblib
import os

def main():
    test = pd.read_csv('./data/test.csv')
    row_ids = test['row_id']
    X_test = test.drop(columns=['row_id'])

    # Trackman 물리 데이터(속도, 회전수) 장착
    speed_dict = joblib.load('./model/speed_dict.pkl')
    spin_dict = joblib.load('./model/spin_dict.pkl')
    mean_speed = joblib.load('./model/mean_speed.pkl')
    mean_spin = joblib.load('./model/mean_spin.pkl')

    primary_pitch_dict = joblib.load('./model/primary_pitch_dict.pkl')
    mean_primary_ratio = joblib.load('./model/mean_primary_ratio.pkl')

    X_test['avg_speed'] = X_test['pitcher_id'].map(speed_dict).fillna(mean_speed)
    X_test['avg_spin'] = X_test['pitcher_id'].map(spin_dict).fillna(mean_spin)
    X_test['primary_pitch_ratio'] = X_test['pitcher_id'].map(primary_pitch_dict).fillna(mean_primary_ratio)

    # 🌟 [NEW] 야구 도메인 파생 변수 추가 블록
    X_test['pressure_index'] = 0
    X_test['pressure_index'] += np.where(X_test['inning'] >= 7, 1, 0)
    X_test['pressure_index'] += np.where(X_test['balls_before'] == 3, 2, 0)
    X_test['pressure_index'] += np.where((X_test['outs_before'] == 2) & (X_test['balls_before'] >= 2), 1, 0)
    X_test['count_advantage'] = X_test['strikes_before'] - X_test['balls_before']
    X_test['platoon_advantage'] = np.where(X_test['pitcher_hand'] == X_test['batter_hand'], 1, 0)

    # 결측치 처리 (0으로 다시 통일하여 베이스라인 안정성 확보)
    numeric_cols = X_test.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if X_test[col].isnull().any():
            X_test[col] = X_test[col].fillna(0)

    object_cols = X_test.select_dtypes(include=['object']).columns
    for col in object_cols:
        if X_test[col].isnull().any():
            X_test[col] = X_test[col].fillna('Unknown')
        X_test[col] = X_test[col].astype('category')

    model = joblib.load('./model/model.pkl')
    pred_proba = model.predict_proba(X_test)[:, 1]

    os.makedirs('./output', exist_ok=True)
    pd.DataFrame({
        'row_id': row_ids,
        'control_success': pred_proba
    }).to_csv('./output/submission.csv', index=False)

if __name__ == '__main__':
    main()
