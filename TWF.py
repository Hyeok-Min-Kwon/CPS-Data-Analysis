import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from kaggle.api.kaggle_api_extended import KaggleApi

# ---------------------------------------------------------
# 1. 데이터 준비 (파생 변수 생성 포함)
# ---------------------------------------------------------
api = KaggleApi()
api.authenticate()
dataset_name = "stephanmatzka/predictive-maintenance-dataset-ai4i-2020"

if not os.path.exists("ai4i2020.csv"):
    api.dataset_download_files(dataset_name, path="./", unzip=True)

data = pd.read_csv("ai4i2020.csv")
data = data.drop(columns=["Product ID", "UDI"]) # Type은 Strain 분석 참고용으로 유지 가능하나 여기선 제거

# -- Feature Engineering (고장 조건 구현) --
# [HDF용] 온도 차이
data['Temp diff [K]'] = data['Process temperature [K]'] - data['Air temperature [K]']
# [PWF용] 전력 (W)
data['Power [W]'] = data['Torque [Nm]'] * (data['Rotational speed [rpm]'] * (2 * np.pi / 60))
# [OSF용] 과부하 (Strain)
data['Strain [minNm]'] = data['Tool wear [min]'] * data['Torque [Nm]']

# ---------------------------------------------------------
# 2. 통합 시각화 함수
# ---------------------------------------------------------
# 기본적으로 확인할 5대 피처
BASIC_FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

def plot_failure_mode_details(failure_code, additional_features=None, thresholds=None):
    """
    failure_code: 고장 유형 (예: 'HDF')
    additional_features: 기본 5개 외에 추가로 볼 파생 변수 리스트 (예: ['Temp diff [K]'])
    thresholds: 피처별 임계값 딕셔너리 (예: {'Rotational speed [rpm]': [1380]})
    """
    # 1. 해당 고장 데이터 필터링
    subset = data[data[failure_code] == 1]
    if len(subset) == 0:
        print(f"[Skip] {failure_code} 데이터 없음")
        return

    # 2. 그릴 피처 목록 합치기 (기본 + 추가)
    features_to_plot = BASIC_FEATURES.copy()
    if additional_features:
        features_to_plot.extend(additional_features)
    
    # 3. 그래프 설정 (2행 3열 or 2행 4열 등 유동적 조정)
    n_features = len(features_to_plot)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    plt.figure(figsize=(5 * n_cols, 4 * n_rows))
    
    for i, col in enumerate(features_to_plot, start=1):
        plt.subplot(n_rows, n_cols, i)
        
        # 전체 데이터 기준 범위 (스케일 고정)
        g_min = data[col].min()
        g_max = data[col].max()
        
        # 히스토그램 그리기
        # (전체 범위 내에서 고장 데이터가 어디 있는지 표시)
        subset[col].hist(bins=30, range=(g_min, g_max), 
                         edgecolor='black', color='salmon', alpha=0.9, label=f'{failure_code}')
        
        # 임계값 그리기 (해당 피처에 임계값이 설정되어 있다면)
        if thresholds and col in thresholds:
            th_vals = thresholds[col]
            # 리스트가 아니면 리스트로 변환
            if not isinstance(th_vals, list):
                th_vals = [th_vals]
                
            for th in th_vals:
                plt.axvline(th, color='blue', linestyle='--', linewidth=2, label=f'Threshold {th}')
                # 임계값 근처 텍스트 표시
                plt.text(th, plt.ylim()[1]*0.9, f'{th}', color='blue', ha='right', va='top', rotation=90)

        plt.title(col)
        plt.xlim(g_min, g_max)
        plt.legend(loc='upper right')

    plt.suptitle(f"Analysis of {failure_code} (Basic Features + Causes)", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# ---------------------------------------------------------
# 3. 고장 유형별 실행 (설정 적용)
# ---------------------------------------------------------

# === 1. TWF (공구 마모) ===
# 추가 변수: 없음 (Tool wear가 기본에 포함됨)
# 임계값: Tool wear 200, 240
print("\n=== 1. TWF Analysis ===")
plot_failure_mode_details(
    'TWF', 
    additional_features=[], 
    thresholds={'Tool wear [min]': [200, 240]}
)

# === 2. HDF (열 방출 실패) ===
# 추가 변수: Temp diff
# 임계값: Temp diff < 8.6, Speed < 1380
print("\n=== 2. HDF Analysis ===")
plot_failure_mode_details(
    'HDF', 
    additional_features=['Temp diff [K]'], 
    thresholds={
        'Temp diff [K]': 8.6,
        'Rotational speed [rpm]': 1380
    }
)

# === 3. PWF (전력 고장) ===
# 추가 변수: Power
# 임계값: Power 3500, 9000
print("\n=== 3. PWF Analysis ===")
plot_failure_mode_details(
    'PWF', 
    additional_features=['Power [W]'], 
    thresholds={'Power [W]': [3500, 9000]}
)

# === 4. OSF (과부하 고장) ===
# 추가 변수: Strain
# 임계값: Strain 11000, 12000, 13000 (Type별 임계값 모두 표시)
print("\n=== 4. OSF Analysis ===")
plot_failure_mode_details(
    'OSF', 
    additional_features=['Strain [minNm]'], 
    thresholds={'Strain [minNm]': [11000, 12000, 13000]}
)

# === 5. RNF (랜덤 고장) ===
# 추가 변수 없음, 임계값 없음 (분포만 확인)
print("\n=== 5. RNF Analysis ===")
plot_failure_mode_details('RNF')