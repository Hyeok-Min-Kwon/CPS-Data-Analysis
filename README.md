# 🏭 Milling Machine Failure Analysis  

> 밀링 머신 공정 시뮬레이션 데이터를 기반으로  
> **5가지 설비 고장 유형의 물리적 발생 메커니즘을 분석**한 프로젝트입니다.

---

## 📌 Project Overview

본 프로젝트는 밀링 머신 공정에서 수집된 **예측 유지보수(Predictive Maintenance)** 센서 데이터를 활용하여,  
설비 고장을 단순 이진 분류가 아닌 **고장 유형별 물리적 원인 관점**에서 분석하는 것을 목표로 합니다.

---

## 📎 Data Source

- 📥 **Predictive Maintenance Dataset (AI4I 2020)**  
  데이터 출처: https://www.kaggle.com/datasets/stephanmatzka/predictive-maintenance-dataset-ai4i-2020  

  이 데이터는 산업 환경에서 설비 이상 상태를 예측하기 위해 수집된 센서 및 운영 데이터로,  
  다양한 고장 유형과 정상 운영 상태의 센서 값이 함께 포함되어 있습니다.

---

## 👥 Team

- **권혁민** (Team Leader)  
  Data Preprocessing · EDA · Feature Engineering · Presentation  
- **장서윤**  
  Data Loading · EDA · Feature Engineering · Presentation  
- **조예은**  
  Data Preprocessing · EDA · Feature Engineering · Presentation  

---

## 📊 Dataset Description

### 🔹 Failure Types (Target)

| Code | Failure Type | Description |
|------|--------------|-------------|
| TWF  | Tool Wear Failure | 공구 마모가 원인인 고장 |
| HDF  | Heat Dissipation Failure | 열 방출 불량으로 인한 과열 |
| PWF  | Power Failure | 전력 이상 상황 |
| OSF  | Overstrain Failure | 과부하 / 무리한 공정 |
| RNF  | Random Failure | 확률적 원인 고장 |

---

### 🔹 Sensor Features

- **Air temperature [K]** : 공정 외부 온도  
- **Process temperature [K]** : 공구 접촉부 온도  
- **Rotational speed [rpm]** : 스핀들 회전 속도  
- **Torque [Nm]** : 스핀들 토크  
- **Tool wear [min]** : 누적 공구 사용 시간  
- **Type (L / M / H)** : 제품 등급

---

## 🔍 Analysis Strategy

분석 접근 방식:

- 고장 유형별로 **중요 변수 및 분포 특성 파악**
- **물리적으로 의미 있는 파생 변수 생성**
- 고장/정상 개체 간 분포의 **결정적 차이 확인**

---

## 🧠 Failure-wise Analysis Summary

### 🔧 TWF – Tool Wear Failure
- **핵심 변수**: `Tool wear`  
- 고장 samples의 거의 모든 경우가 **높은 마모 수준**에 집중  
- ▶ Tool wear가 사실상 고장을 설명하는 대표 지표

---

### 🔥 HDF – Heat Dissipation Failure
- **파생 변수**: `Temp_diff = Process Temp - Air Temp`  
- 고장에서는 일정 범위의 Temp_diff에 **가깝게 집중**  
- ▶ 열 방출 불균형이 주요 메커니즘으로 드러남

---

### ⚡ PWF – Power Failure
- **파생 변수**: `Power = Rotational Speed × Torque`  
- 단일 변수로 구분하기 어려웠던 고장을  
  Power 값으로 구간화함으로써 **명확한 영역 분리** 확인

---

### 🛠 OSF – Overstrain Failure
- **파생 변수**: `Strain = Tool wear × Torque`  
- 고장 그룹에서 Strain의 평균이 정상 대비 **월등히 큼**  
- 제품 등급(L/M/H)에 따라 분포 차이를 추가 확인

---

### 🎲 RNF – Random Failure
- 전체 데이터 중 비중이 극히 낮으며 분포적 특징 부재  
- ▶ 통계적 의미로 해석하기 어려운 **확률적 고장**

---

## 📌 Key Takeaways

- 고장 유형별로 **물리적 원인 메커니즘이 다름**
- **파생 변수 생성**은 노이즈를 제거하고 의미 있는 차이 식별에 효과적
- 단순 통계 평균보다 **분포와 임계 구간 중심 분석**이 중요

---

## 🚀 Significance

이 분석은:

- 실제 스마트 제조 환경에서 **Rule-based 진단 로직** 설계가 가능한 분석
- 향후 **예측 모델, 실시간 모니터링 시스템** 구축의 기반 제공

---

## 🗂 Reference

Predictive Maintenance Dataset (AI4I 2020) – Kaggle  
https://www.kaggle.com/datasets/stephanmatzka/predictive-maintenance-dataset-ai4i-2020

