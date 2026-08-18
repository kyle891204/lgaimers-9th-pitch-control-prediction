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
