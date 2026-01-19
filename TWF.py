import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from kaggle.api.kaggle_api_extended import KaggleApi

# ==========================================
# 1. 데이터 로드 및 전처리
# ==========================================
def load_and_preprocess_data():
    # Kaggle API 인증 및 다운로드
    api = KaggleApi()
    api.authenticate()
    dataset_name = "stephanmatzka/predictive-maintenance-dataset-ai4i-2020"

    if not os.path.exists("ai4i2020.csv"):
        api.dataset_download_files(dataset_name, path="./", unzip=True)
        print("✅ 데이터셋 다운로드 완료.")
    else:
        print("✅ 데이터셋 파일이 이미 존재합니다.")

    # CSV 파일 읽기
    df = pd.read_csv("ai4i2020.csv")
    
    # 불필요한 식별자 컬럼 제거 (Type은 OSF 분석에 필수이므로 유지!)
    df = df.drop(columns=["Product ID", "UDI"])
    
    # -- Feature Engineering (물리적 조건 반영) --
    # 1. [HDF용] 온도 차이 (Temp diff)
    df['Temp diff [K]'] = df['Process temperature [K]'] - df['Air temperature [K]']
    
    # 2. [PWF용] 전력 (Power) = 토크 * 회전속도(rad/s)
    # 1 rpm = 2*pi/60 rad/s
    df['Power [W]'] = df['Torque [Nm]'] * (df['Rotational speed [rpm]'] * (2 * np.pi / 60))
    
    # 3. [OSF용] 과부하 (Strain) = 공구 마모 * 토크
    df['Strain [minNm]'] = df['Tool wear [min]'] * df['Torque [Nm]']
    
    print(f"✅ 데이터 로드 및 파생 변수 생성 완료. (크기: {df.shape})")
    return df

# ==========================================
# 2. 시각화 함수 정의
# ==========================================

# 기본적으로 확인할 5대 피처
BASIC_FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

def plot_general_failure_analysis(df, failure_code, additional_features=None, thresholds=None):
    """
    OSF를 제외한 일반 고장(TWF, HDF, PWF, RNF) 분석용 함수
    """
    subset = df[df[failure_code] == 1]
    if len(subset) == 0:
        print(f"[Skip] {failure_code} 데이터가 없습니다.")
        return

    # 그릴 피처 목록 합치기
    features_to_plot = BASIC_FEATURES.copy()
    if additional_features:
        features_to_plot.extend(additional_features)
    
    # 그래프 레이아웃 설정
    n_features = len(features_to_plot)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    plt.figure(figsize=(5 * n_cols, 4 * n_rows))
    
    for i, col in enumerate(features_to_plot, start=1):
        plt.subplot(n_rows, n_cols, i)
        
        # 스케일 고정 (전체 데이터 기준)
        g_min = df[col].min()
        g_max = df[col].max()
        
        # 히스토그램
        subset[col].hist(bins=30, range=(g_min, g_max), 
                         edgecolor='black', color='salmon', alpha=0.9, label=f'{failure_code}')
        
        # 임계값 표시
        if thresholds and col in thresholds:
            th_vals = thresholds[col]
            if not isinstance(th_vals, list):
                th_vals = [th_vals]
            for th in th_vals:
                plt.axvline(th, color='blue', linestyle='--', linewidth=2, label=f'Threshold {th}')
                plt.text(th, plt.ylim()[1]*0.9, f'{th}', color='blue', ha='right', rotation=90)

        plt.title(col)
        plt.xlim(g_min, g_max)
        plt.legend(loc='upper right')

    plt.suptitle(f"Failure Analysis: {failure_code}", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def plot_osf_full_analysis(df):
    """
    OSF(과부하 고장) 전용 통합 분석 함수.
    - 상단: 기본 5대 피처 분포 (전체 OSF)
    - 하단: Type L, M, H 별 Strain 분포 (임계값 포함)
    """
    osf_data = df[df['OSF'] == 1]
    if len(osf_data) == 0:
        print("[Skip] OSF 데이터가 없습니다.")
        return
        
    # 레이아웃: 2행 4열 (총 8개 공간)
    # 1~5: 기본 피처
    # 6~8: Type별 Strain
    fig = plt.figure(figsize=(24, 10))
    
    # --- Part 1: 기본 5개 피처 (전체 OSF 데이터) ---
    for i, col in enumerate(BASIC_FEATURES, start=1):
        ax = fig.add_subplot(2, 4, i)
        
        g_min = df[col].min()
        g_max = df[col].max()
        
        osf_data[col].hist(bins=30, range=(g_min, g_max), 
                           edgecolor='black', color='salmon', alpha=0.9, ax=ax, label='OSF')
        
        ax.set_title(col)
        ax.set_xlim(g_min, g_max)
        ax.legend(loc='upper right')

    # --- Part 2: Type별 Strain (L, M, H) ---
    types = ['L', 'M', 'H']
    thresholds = {'L': 11000, 'M': 12000, 'H': 13000}
    
    # Strain 축 통일
    s_min = df['Strain [minNm]'].min()
    s_max = df['Strain [minNm]'].max()

    for i, t in enumerate(types, start=6): # 6, 7, 8번 위치
        ax = fig.add_subplot(2, 4, i)
        subset = osf_data[osf_data['Type'] == t]
        
        if len(subset) > 0:
            subset['Strain [minNm]'].hist(bins=20, range=(s_min, s_max), 
                                          edgecolor='black', color='salmon', alpha=0.9, ax=ax, label=f'OSF Type {t}')
        else:
            ax.text(0.5, 0.5, 'No Data', ha='center', transform=ax.transAxes)
            
        # 임계값 표시
        th = thresholds[t]
        ax.axvline(th, color='blue', linestyle='--', linewidth=2, label=f'Limit: {th}')
        ax.text(th, ax.get_ylim()[1]*0.9, f' {th}', color='blue', fontweight='bold', rotation=90)
        
        ax.set_title(f"[Type {t}] Strain Analysis")
        ax.set_xlabel("Strain [minNm]")
        ax.set_xlim(s_min, s_max)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle("OSF Analysis (Basic Features + Type-specific Strain)", fontsize=20)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# ==========================================
# 3. 메인 실행 블록
# ==========================================
if __name__ == "__main__":
    # 1. 데이터 준비
    data = load_and_preprocess_data()
    
    # 2. 고장 유형별 분석 실행
    
    # (1) TWF: 공구 마모 (200~240분)
    print("\n=== 1. TWF Analysis ===")
    plot_general_failure_analysis(data, 'TWF', thresholds={'Tool wear [min]': [200, 240]})

    # (2) HDF: 열 방출 (온도차 < 8.6, 회전수 < 1380)
    print("\n=== 2. HDF Analysis ===")
    plot_general_failure_analysis(
        data, 'HDF', 
        additional_features=['Temp diff [K]'],
        thresholds={'Temp diff [K]': 8.6, 'Rotational speed [rpm]': 1380}
    )

    # (3) PWF: 전력 고장 (3500 미만, 9000 초과)
    print("\n=== 3. PWF Analysis ===")
    plot_general_failure_analysis(
        data, 'PWF', 
        additional_features=['Power [W]'],
        thresholds={'Power [W]': [3500, 9000]}
    )

    # (4) OSF: 과부하 고장 (5개 피처 + Type별 Strain 분리)
    print("\n=== 4. OSF Analysis (Full Detail) ===")
    plot_osf_full_analysis(data)

    # (5) RNF: 랜덤 고장 (특이사항 없음)
    print("\n=== 5. RNF Analysis ===")
    plot_general_failure_analysis(data, 'RNF')