# lgaimers-9th-pitch-control-prediction
260805 - 260902 LG Aimers 9th Hackerthon: Pitch Control Preddiction

![LG Aimers](https://img.shields.io/badge/LG-Aimers_9th-crimson?style=flat-square&logo=LG)
![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=Python)
![LightGBM](https://img.shields.io/badge/LightGBM-Ready-green?style=flat-square)

## Directory Structure
데이터, 모델 가중치 파일은 Git 트래킹에서 제외함.

```text
lgaimers-pitch-control-prediction/
│
├── data/                    # 📂 [데이터] 대회 제공 csv 파일 (Git 제외)
├── model/                   # 📂 [모델] 학습된 .pkl 가중치 파일 (Git 제외)
├── output/                  # 📂 [결과물] 자체 채점용 csv 및 제출용 zip (Git 제외)
│
├── baseline.ipynb           # 📝 EDA, Feature Engineering 및 모델 학습 노트북
├── script.py                # 📝 주최 측 서버 평가용 추론(Inference) 스크립트
├── requirements.txt         # 📝 제출 환경 패키지 목록
└── README.md                # 📄 프로젝트 설명서
```

## 파이프라인 및 모델 구성
1. 전처리/모델링(Baseline)
2. Feature Engineering
3. Probability Calibration
4. 모델 추론 규칙
   - test.csv 내 다른 행을 이용한 통계(평균, 분포, ...) 생성 금지
   - 예측 확률에 대한 사후 임의 보정 금지
   - 현재 행의 정보 및 사전에 Train, Trackman 데이터를 통해 생성된 규칙만을 사용하여 추론 진행
  
## 실행방법
1. 환경세팅
```text
pip install pandas numpy scikit-learn lightbgm joblib
```
2. 데이터 배치
   - 데이터 data/ 폴더내에 위치
3. 모델 학습
   - baseline.ipynb 노트북의 모든 셀을 실행하여 전처리, 모델학습, 로컬자체평가 진행 및 model.pkl과 script.py 생성
4. 서버제출
   - model폴더, script.py, requirements.txt를 하나의 .zip으로 압축하여 제출
