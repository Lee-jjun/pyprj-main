import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
from matplotlib import font_manager, rc

from bbs.utils.common_files import FileUtils
from bbs.utils.fonts import setup_matplotlib_fonts
# import json
import traceback

# LinearRegression(선형 회귀)
def LRegression():

    setup_matplotlib_fonts()
    ##################################################
    # 1. 데이터 불러오기 및 전처리
    ##################################################
    try:
        df = FileUtils().getCsv("아파트(전월세)_실거래가_all.csv")
    except UnicodeDecodeError as un:
        df = FileUtils().getCsv("아파트(전월세)_실거래가_all.csv")
    except FileNotFoundError as e:
        print(f"파일을 찾을 수 없습니다: {e.filename}")

    # 데이터 전처리
    df['거래일'] = pd.to_datetime(df['계약년월'].astype(str) + df['계약일'].astype(str).str.zfill(2), format='%Y%m%d')

    bins = [0, 40, 60, 85, 100, 135, 200]
    labels = ['~40㎡', '40~60㎡', '60~85㎡', '85~100㎡', '100~135㎡', '135㎡~']
    df['면적_구간'] = pd.cut(df['전용면적(㎡)'], bins=bins, labels=labels, right=False)

    df['월'] = df['거래일'].dt.to_period('M')
    monthly_demand = df.groupby(['월', '전월세구분', '면적_구간']).size().reset_index(name='수요량')
    monthly_demand['월'] = monthly_demand['월'].dt.to_timestamp()

    # 'month_index' 생성
    monthly_demand_full = monthly_demand.groupby('월').size().reset_index()
    monthly_demand_full['month_index'] = np.arange(len(monthly_demand_full))
    monthly_demand = pd.merge(monthly_demand, monthly_demand_full[['월', 'month_index']], on='월', how='left')

    # 예측할 월 (2025-09, 2025-10 두 달) 설정
    predict_dates = pd.date_range(start='2025-09', end='2025-10', freq='MS')
    min_date = monthly_demand['월'].min()
    predict_date_indices = [(d.year - min_date.year) * 12 + (d.month - min_date.month) for d in predict_dates]

    print("--- 면적 구간별 LinearRegression 예측 결과 및 성능 지표 ---")

    predicted_results = {'전세': {}, '월세': {}}
    performance_metrics = {'전세': {}, '월세': {}}

    # 실제/예측 리스트 저장
    time_series_results = {'전세': {}, '월세': {}}

    for rent_type in ['전세', '월세']:
        print(f"\n--- {rent_type} ---")
        for area_label in labels:
            subset_df = monthly_demand[(monthly_demand['전월세구분'] == rent_type) & (monthly_demand['면적_구간'] == area_label)].copy()

            if len(subset_df) < 2:
                print(f"  > {area_label} : 데이터 부족으로 예측 불가")
                continue

            X = subset_df[['month_index']]
            y = subset_df['수요량']

            model = LinearRegression()
            model.fit(X, y)

            # 두 달 예측 (2025-09, 2025-10)
            predicted_future = model.predict(np.array(predict_date_indices).reshape(-1, 1))

            # 과거 데이터 예측
            y_pred = model.predict(X)

            # === 길이 맞추기 ===
            combined_y_pred = np.concatenate([y_pred, predicted_future])
            combined_months = list(subset_df['월'].dt.strftime("%Y-%m")) + list(predict_dates.strftime("%Y-%m"))
            combined_y_actual = list(y.astype(int)) + [None] * len(predict_dates)

            # 성능 지표
            rmse = np.sqrt(mean_squared_error(y, y_pred))
            r2 = r2_score(y, y_pred)
            mae = mean_absolute_error(y, y_pred)

            # 결과 저장
            predicted_results[rent_type][area_label] = {d.strftime("%Y-%m"): int(v) for d, v in zip(predict_dates, predicted_future)}
            performance_metrics[rent_type][area_label] = {'RMSE': f"{rmse:.2f}", 'R2': f"{r2:.2f}", 'MAE': f"{mae:.2f}"}
            time_series_results[rent_type][area_label] = {
                "실제값 리스트": combined_y_actual,
                "예측값 리스트": [int(val) for val in combined_y_pred],
                "년-월": combined_months
            }

            print(f"  > {area_label} 예측: {predicted_results[rent_type][area_label]}")
            print(f"  > {area_label} 성능 지표: RMSE={rmse:.2f}, R²={r2:.2f}, MAE={mae:.2f}")

    # --- 결과 시각화 (라인 차트로 변경) ---
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for i, area_label in enumerate(labels):
        ax = axes[i]
        for rent_type, color in zip(['전세', '월세'], ['blue', 'red']):
            if area_label not in time_series_results[rent_type]:
                continue
            data = time_series_results[rent_type][area_label]
            months = pd.to_datetime(data["년-월"])
            actual = data["실제값 리스트"]
            pred = data["예측값 리스트"]

            ax.plot(months, pred, linestyle='--', marker='o', color=color, label=f"{rent_type} 예측")
            ax.plot(months, actual, linestyle='-', marker='x', color=color, alpha=0.5, label=f"{rent_type} 실제")

        ax.set_title(f"{area_label} 수요량 (2025-09~10 포함)")
        ax.legend()
        ax.grid(True)

    plt.tight_layout()

    # plt.show()
    filename = FileUtils().savePngToPath("linear-regression.png",closeFlag=True)
    # print(f"=== 생성된 이미지 파일 위치 {filename}")
    print(f"predicted_results {predicted_results}")

    return { "predicted":predicted_results
            ,"performance":performance_metrics
            ,"analysis":{}
            ,"img":filename}

# Decision Tree Regression (결정 트리 회귀): 
# 데이터를 특정 조건에 따라 나누는 '결정 트리' 구조를 사용하여 예측합니다.
from sklearn.tree import DecisionTreeRegressor
                         
def DTreeRegressor():

    # 운영체제에 맞는 한글 폰트 설정
    # Windows
    setup_matplotlib_fonts()
    # 1. CSV 파일 불러오기
    try:
        df = FileUtils().getCsv("아파트(전월세)_실거래가_all.csv")
    except UnicodeDecodeError as un:
        df = FileUtils().getCsv("아파트(전월세)_실거래가_all.csv")
    except FileNotFoundError as e:
        print(f"파일을 찾을 수 없습니다: {e.filename}")

    # 데이터 전처리
    df['거래일'] = pd.to_datetime(
        df['계약년월'].astype(str) + df['계약일'].astype(str).str.zfill(2),
        format='%Y%m%d'
    )

    bins = [0, 40, 60, 85, 100, 135, 200]
    labels = ['~40㎡', '40~60㎡', '60~85㎡', '85~100㎡', '100~135㎡', '135㎡~']
    df['면적_구간'] = pd.cut(df['전용면적(㎡)'], bins=bins, labels=labels, right=False)

    df['월'] = df['거래일'].dt.to_period('M')
    monthly_demand = df.groupby(['월', '전월세구분', '면적_구간']).size().reset_index(name='수요량')
    monthly_demand['월'] = monthly_demand['월'].dt.to_timestamp()

    # 'month_index'를 전체 데이터 기준으로 일관되게 생성
    monthly_demand_full = monthly_demand.groupby('월').size().reset_index()
    monthly_demand_full['month_index'] = np.arange(len(monthly_demand_full))
    monthly_demand = pd.merge(monthly_demand, monthly_demand_full[['월', 'month_index']], on='월', how='left')

    # 예측할 월 (2025-09, 2025-10 두 달로 확장)
    predict_dates = pd.date_range(start='2025-09', end='2025-10', freq='MS')
    min_date = monthly_demand['월'].min()
    predict_date_indices = [(d.year - min_date.year) * 12 + (d.month - min_date.month) for d in predict_dates]

    print("--- 면적 구간별 Decision Tree Regression 예측 결과 및 성능 지표 ---")

    # 예측 결과 저장 구조
    predicted_results = {'전세': {}, '월세': {}}
    performance_metrics = {'전세': {}, '월세': {}}
    analysis_metrics = {'전세': {}, '월세': {}}
    # 시계열 저장 구조 추가
    time_series_results = {'전세': {}, '월세': {}}

    # 전/월세 및 면적 구간별 반복
    for rent_type in ['전세', '월세']:
        print(f"\n--- {rent_type} ---")
        for area_label in labels:
            subset_df = monthly_demand[
                (monthly_demand['전월세구분'] == rent_type) & 
                (monthly_demand['면적_구간'] == area_label)
            ].copy()

            if len(subset_df) < 2:
                print(f"  > {area_label} : 데이터 부족으로 예측 불가")
                continue

            X = subset_df[['month_index']]
            y = subset_df['수요량']

            # DecisionTreeRegressor 모델 학습
            model = DecisionTreeRegressor(random_state=42)
            model.fit(X, y)

            # 두 달 예측
            predicted_future = model.predict(np.array(predict_date_indices).reshape(-1, 1))

            # 과거 데이터 예측
            y_pred = model.predict(X)

            # === 길이 맞추기 ===
            combined_y_pred = np.concatenate([y_pred, predicted_future])
            combined_months = list(subset_df['월'].dt.strftime("%Y-%m")) + list(predict_dates.strftime("%Y-%m"))
            combined_y_actual = list(y.astype(int)) + [None] * len(predict_dates)

            # 성능 지표
            rmse = np.sqrt(mean_squared_error(y, y_pred))
            r2 = r2_score(y, y_pred)
            mae = mean_absolute_error(y, y_pred)

            # 특성 중요도
            feature_importance = model.feature_importances_[0]

            # 결과 저장
            predicted_results[rent_type][area_label] = {
                d.strftime("%Y-%m"): int(v) for d, v in zip(predict_dates, predicted_future)
            }
            performance_metrics[rent_type][area_label] = {'RMSE': f"{rmse:.2f}", 'R2': f"{r2:.2f}", 'MAE': f"{mae:.2f}"}
            analysis_metrics[rent_type][area_label] = {'특성 중요도': feature_importance}
            time_series_results[rent_type][area_label] = {
                "실제값 리스트": combined_y_actual,
                "예측값 리스트": [int(val) for val in combined_y_pred],
                "년-월": combined_months
            }

            print(f"  > {area_label} 예측: {predicted_results[rent_type][area_label]}")
            print(f"  > {area_label} 성능 지표: RMSE={rmse:.2f}, R²={r2:.2f}, MAE={mae:.2f}")

    # --- 결과 시각화 (라인 차트로 변경) ---
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for i, area_label in enumerate(labels):
        ax = axes[i]
        for rent_type, color in zip(['전세', '월세'], ['blue', 'red']):
            if area_label not in time_series_results[rent_type]:
                continue
            data = time_series_results[rent_type][area_label]
            months = pd.to_datetime(data["년-월"])
            actual = data["실제값 리스트"]
            pred = data["예측값 리스트"]

            ax.plot(months, pred, linestyle='--', marker='o', color=color, label=f"{rent_type} 예측")
            ax.plot(months, actual, linestyle='-', marker='x', color=color, alpha=0.5, label=f"{rent_type} 실제")

        ax.set_title(f"{area_label} 수요량 (2025-09~10 포함)")
        ax.legend()
        ax.grid(True)

    plt.tight_layout()

    filename = FileUtils().savePngToPath("decision-tree-regressor.png",closeFlag=True)
    # plt.show()
    # return predicted_results ,performance_metrics
    return { "predicted":predicted_results
            ,"performance":performance_metrics
            ,"analysis":{}
            ,"img":filename}

#  Tree-based Models (트리 기반 모델)

from sklearn.ensemble import RandomForestRegressor
def RFRegressor():

    # 데이터 분할: 전체 데이터를 학습 데이터와 테스트 데이터로 분할하지 않고, 2024년 1월부터 2025년 8월까지의 데이터를 학습 데이터로 사용합니다. 그리고 2025년 10월은 예측 대상 시점으로 설정합니다.
    # 면적 구간별 모델 학습: 전체 데이터에 대해 하나의 모델을 학습시키는 것이 아니라, 전/월세와 각각의 면적 구간별로 별도의 RandomForestRegressor 모델을 학습시킵니다.
    # 예측 및 평가: 각 모델이 2025년 10월의 수요량을 예측하고, 학습에 사용된 데이터로 모델의 성능을 평가하기 위해 RMSE와 R² 점수를 계산하여 출력합니다.
    # RMSE: 예측값과 실제값의 차이를 제곱하여 평균낸 값의 제곱근입니다. 값이 작을수록 예측 정확도가 높습니다.
    # R² (결정 계수): 모델이 데이터의 분산을 얼마나 잘 설명하는지를 나타냅니다. 값이 1에 가까울수록 모델의 설명력이 높습니다.
    # 운영체제에 맞는 한글 폰트 설정
    # Windows
    setup_matplotlib_fonts()

    # 1. CSV 파일 불러오기
    try:
        df = FileUtils().getCsv("아파트(전월세)_실거래가_all.csv")
    except UnicodeDecodeError as un:
        df = FileUtils().getCsv("아파트(전월세)_실거래가_all.csv")
    except FileNotFoundError as e:
        print(f"파일을 찾을 수 없습니다: {e.filename}")

    # 데이터 전처리
    df['거래일'] = pd.to_datetime(
        df['계약년월'].astype(str) + df['계약일'].astype(str).str.zfill(2),
        format='%Y%m%d'
    )

    bins = [0, 40, 60, 85, 100, 135, 200]
    labels = ['~40㎡', '40~60㎡', '60~85㎡', '85~100㎡', '100~135㎡', '135㎡~']
    df['면적_구간'] = pd.cut(df['전용면적(㎡)'], bins=bins, labels=labels, right=False)

    df['월'] = df['거래일'].dt.to_period('M')
    monthly_demand = df.groupby(['월', '전월세구분', '면적_구간']).size().reset_index(name='수요량')
    monthly_demand['월'] = monthly_demand['월'].dt.to_timestamp()

    # 'month_index'를 전체 데이터 기준으로 일관되게 생성
    monthly_demand_full = monthly_demand.groupby('월').size().reset_index()
    monthly_demand_full['month_index'] = np.arange(len(monthly_demand_full))
    monthly_demand = pd.merge(monthly_demand, monthly_demand_full[['월', 'month_index']], on='월', how='left')

    # 예측할 월 (2025-09, 2025-10 두 달)
    predict_dates = pd.date_range(start='2025-09', end='2025-10', freq='MS')
    min_date = monthly_demand['월'].min()
    predict_date_indices = [(d.year - min_date.year) * 12 + (d.month - min_date.month) for d in predict_dates]

    print("--- 면적 구간별 랜덤 포레스트 예측 결과 및 성능 지표 ---")

    # 예측 결과 저장
    predicted_results = {'전세': {}, '월세': {}}
    performance_metrics = {'전세': {}, '월세': {}}
    analysis_metrics = {'전세': {}, '월세': {}}
    # 시계열 저장 추가
    time_series_results = {'전세': {}, '월세': {}}

    # 전/월세 및 면적 구간별로 반복
    for rent_type in ['전세', '월세']:
        print(f"\n--- {rent_type} ---")
        for area_label in labels:
            subset_df = monthly_demand[
                (monthly_demand['전월세구분'] == rent_type) & 
                (monthly_demand['면적_구간'] == area_label)
            ].copy()

            if len(subset_df) < 2:
                print(f"  > {area_label} : 데이터 부족으로 예측 불가")
                continue

            X = subset_df[['month_index']]
            y = subset_df['수요량']

            # RandomForestRegressor 모델 학습
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X, y)

            # 두 달 예측
            predicted_future = model.predict(np.array(predict_date_indices).reshape(-1, 1))

            # 과거 데이터 예측
            y_pred = model.predict(X)

            # === 길이 맞추기 ===
            combined_y_pred = np.concatenate([y_pred, predicted_future])
            combined_months = list(subset_df['월'].dt.strftime("%Y-%m")) + list(predict_dates.strftime("%Y-%m"))
            combined_y_actual = list(y.astype(int)) + [None] * len(predict_dates)

            # 성능 지표
            rmse = np.sqrt(mean_squared_error(y, y_pred))
            r2 = r2_score(y, y_pred)
            mae = mean_absolute_error(y, y_pred)

            # 개별 트리 예측값 분석
            individual_predictions = [tree.predict(np.array(predict_date_indices).reshape(-1, 1)) for tree in model.estimators_]
            pred_mean = np.mean(individual_predictions)
            pred_std = np.std(individual_predictions)

            # 특성 중요도
            feature_importance = model.feature_importances_[0]

            # 결과 저장
            predicted_results[rent_type][area_label] = {
                d.strftime("%Y-%m"): int(v) for d, v in zip(predict_dates, predicted_future)
            }
            performance_metrics[rent_type][area_label] = {
                'RMSE': f"{rmse:.2f}", 'R2': f"{r2:.2f}", 'MAE': f"{mae:.2f}"
            }
            analysis_metrics[rent_type][area_label] = {
                '예측값 평균': float(pred_mean), 
                '예측값 표준편차': float(pred_std), 
                '특성 중요도': feature_importance
            }
            time_series_results[rent_type][area_label] = {
                "실제값 리스트": combined_y_actual,
                "예측값 리스트": [int(val) for val in combined_y_pred],
                "년-월": combined_months
            }

            print(f"  > {area_label} 예측: {predicted_results[rent_type][area_label]}")
            print(f"  > {area_label} 성능 지표: RMSE={rmse:.2f}, R²={r2:.2f}, MAE={mae:.2f}")
            print(f"  > {area_label} 예측 신뢰도 분석: 평균={pred_mean:.2f}, 표준편차={pred_std:.2f}")

    # --- 결과 시각화 (라인 차트) ---
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for i, area_label in enumerate(labels):
        ax = axes[i]
        for rent_type, color in zip(['전세', '월세'], ['blue', 'red']):
            if area_label not in time_series_results[rent_type]:
                continue
            data = time_series_results[rent_type][area_label]
            months = pd.to_datetime(data["년-월"])
            actual = data["실제값 리스트"]
            pred = data["예측값 리스트"]

            ax.plot(months, pred, linestyle='--', marker='o', color=color, label=f"{rent_type} 예측")
            ax.plot(months, actual, linestyle='-', marker='x', color=color, alpha=0.5, label=f"{rent_type} 실제")

        ax.set_title(f"{area_label} 수요량 (2025-09~10 포함)")
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    filename = FileUtils().savePngToPath("random-forest-regressor.png",closeFlag=True)
    print(f" 파일명{filename}")
    
    # plt.show()
    # return predicted_results,performance_metrics,analysis_metrics
    return { "predicted":predicted_results
            ,"performance":performance_metrics
            ,"analysis":analysis_metrics
            ,"img":filename}


from sklearn.ensemble import GradientBoostingRegressor
def GBRegressor():

    setup_matplotlib_fonts()
    # 1. CSV 파일 불러오기
    # 데이터 불러오기
    try:
        df = FileUtils().getCsv("아파트(전월세)_실거래가_all.csv")
    except UnicodeDecodeError as un:
        df = FileUtils().getCsv("아파트(전월세)_실거래가_all.csv")
    except FileNotFoundError as e:
        print(f"파일을 찾을 수 없습니다: {e.filename}")

    # 데이터 전처리
    df['거래일'] = pd.to_datetime(
        df['계약년월'].astype(str) + df['계약일'].astype(str).str.zfill(2),
        format='%Y%m%d'
    )

    bins = [0, 40, 60, 85, 100, 135, 200]
    labels = ['~40㎡', '40~60㎡', '60~85㎡', '85~100㎡', '100~135㎡', '135㎡~']
    df['면적_구간'] = pd.cut(df['전용면적(㎡)'], bins=bins, labels=labels, right=False)

    df['월'] = df['거래일'].dt.to_period('M')
    monthly_demand = df.groupby(['월', '전월세구분', '면적_구간']).size().reset_index(name='수요량')
    monthly_demand['월'] = monthly_demand['월'].dt.to_timestamp()

    # 'month_index' 생성
    monthly_demand_full = monthly_demand.groupby('월').size().reset_index()
    monthly_demand_full['month_index'] = np.arange(len(monthly_demand_full))
    monthly_demand = pd.merge(monthly_demand, monthly_demand_full[['월', 'month_index']], on='월', how='left')

    # 예측할 월 (2025-09, 2025-10)
    predict_dates = pd.date_range(start='2025-09', end='2025-10', freq='MS')
    min_date = monthly_demand['월'].min()
    predict_date_indices = [(d.year - min_date.year) * 12 + (d.month - min_date.month) for d in predict_dates]

    print("--- 면적 구간별 Gradient Boosting Regression 예측 결과 및 성능 지표 ---")

    # 결과 저장
    predicted_results = {'전세': {}, '월세': {}}
    performance_metrics = {'전세': {}, '월세': {}}
    analysis_metrics = {'전세': {}, '월세': {}}
    time_series_results = {'전세': {}, '월세': {}}

    # 전/월세 + 면적 구간 반복
    for rent_type in ['전세', '월세']:
        print(f"\n--- {rent_type} ---")
        for area_label in labels:
            subset_df = monthly_demand[
                (monthly_demand['전월세구분'] == rent_type) & 
                (monthly_demand['면적_구간'] == area_label)
            ].copy()

            if len(subset_df) < 2:
                print(f"  > {area_label} : 데이터 부족으로 예측 불가")
                continue

            X = subset_df[['month_index']]
            y = subset_df['수요량']

            # GradientBoostingRegressor 학습
            model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
            model.fit(X, y)

            # 미래 두 달 예측
            predicted_future = model.predict(np.array(predict_date_indices).reshape(-1, 1))

            # 과거 데이터 예측
            y_pred = model.predict(X)

            # === 길이 맞추기 ===
            combined_y_pred = np.concatenate([y_pred, predicted_future])
            combined_months = list(subset_df['월'].dt.strftime("%Y-%m")) + list(predict_dates.strftime("%Y-%m"))
            combined_y_actual = list(y.astype(int)) + [None] * len(predict_dates)

            # 성능 지표
            rmse = np.sqrt(mean_squared_error(y, y_pred))
            r2 = r2_score(y, y_pred)
            mae = mean_absolute_error(y, y_pred)

            # 특성 중요도
            feature_importance = model.feature_importances_[0]

            # 결과 저장
            predicted_results[rent_type][area_label] = {
                d.strftime("%Y-%m"): int(v) for d, v in zip(predict_dates, predicted_future)
            }
            performance_metrics[rent_type][area_label] = {
                'RMSE': f"{rmse:.2f}", 'R2': f"{r2:.2f}", 'MAE': f"{mae:.2f}"
            }
            analysis_metrics[rent_type][area_label] = {
                '특성 중요도': feature_importance
            }
            time_series_results[rent_type][area_label] = {
                "실제값 리스트": combined_y_actual,
                "예측값 리스트": [int(val) for val in combined_y_pred],
                "년-월": combined_months
            }

            print(f"  > {area_label} 예측: {predicted_results[rent_type][area_label]}")
            print(f"  > {area_label} 성능 지표: RMSE={rmse:.2f}, R²={r2:.2f}, MAE={mae:.2f}")

    # --- 시각화 (라인 차트: 실제 vs 예측) ---
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for i, area_label in enumerate(labels):
        ax = axes[i]
        for rent_type, color in zip(['전세', '월세'], ['blue', 'red']):
            if area_label not in time_series_results[rent_type]:
                continue
            data = time_series_results[rent_type][area_label]
            months = pd.to_datetime(data["년-월"])
            actual = data["실제값 리스트"]
            pred = data["예측값 리스트"]

            ax.plot(months, pred, linestyle='--', marker='o', color=color, label=f"{rent_type} 예측")
            ax.plot(months, actual, linestyle='-', marker='x', color=color, alpha=0.5, label=f"{rent_type} 실제")

        ax.set_title(f"{area_label} 수요량 (2025-09~10 포함)")
        ax.legend()
        ax.grid(True)

    plt.tight_layout()

    # plt.show()
    filename = FileUtils().savePngToPath("gradient-boosting-regressor.png",closeFlag=True)
   
    GradientBoostingRegressor
    return { "predicted":predicted_results
            ,"performance":performance_metrics
            ,"analysis":analysis_metrics
            ,"img":filename}

def allRegress():

    setup_matplotlib_fonts()
    # 1. CSV 파일 불러오기
    try:
        df = FileUtils().getCsv("아파트(전월세)_실거래가_all.csv")
    except UnicodeDecodeError as un:
        df = FileUtils().getCsv("아파트(전월세)_실거래가_all.csv")
    except FileNotFoundError as e:
        print(f"파일을 찾을 수 없습니다: {e.filename}")
    # 데이터 전처리
    df['거래일'] = pd.to_datetime(df['계약년월'].astype(str) + df['계약일'].astype(str).str.zfill(2), format='%Y%m%d')

    bins = [0, 40, 60, 85, 100, 135, 200]
    labels = ['~40㎡', '40~60㎡', '60~85㎡', '85~100㎡', '100~135㎡', '135㎡~']
    df['면적_구간'] = pd.cut(df['전용면적(㎡)'], bins=bins, labels=labels, right=False)

    df['월'] = df['거래일'].dt.to_period('M')
    monthly_demand = df.groupby(['월', '전월세구분', '면적_구간']).size().reset_index(name='수요량')
    monthly_demand['월'] = monthly_demand['월'].dt.to_timestamp()

    # 'month_index'를 전체 데이터 기준으로 일관되게 생성
    monthly_demand_full = monthly_demand.groupby('월').size().reset_index()
    monthly_demand_full['month_index'] = np.arange(len(monthly_demand_full))
    monthly_demand = pd.merge(monthly_demand, monthly_demand_full[['월', 'month_index']], on='월', how='left')

    # 예측할 월 (2025년 10월)의 인덱스 계산
    predict_date_index = (pd.to_datetime('2025-10-01').year - monthly_demand['월'].min().year) * 12 + (pd.to_datetime('2025-10-01').month - monthly_demand['월'].min().month)

    # 모델 정의
    models = {
        'Linear Regression': LinearRegression(),
        'Decision Tree': DecisionTreeRegressor(random_state=42),
        'Random Forest': RandomForestRegressor(random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(random_state=42)
    }

    print("--- 면적 구간별 4가지 모델의 예측 결과 및 성능 지표 ---")

    # 결과를 저장할 딕셔너리
    all_results = {model_name: {'전세': {}, '월세': {}} for model_name in models.keys()}

    # 전/월세 및 면적 구간별로 반복
    for rent_type in ['전세', '월세']:
        print(f"\n==================== {rent_type} ====================")
        for area_label in labels:
            subset_df = monthly_demand[(monthly_demand['전월세구분'] == rent_type) & (monthly_demand['면적_구간'] == area_label)].copy()

            if len(subset_df) < 2:
                print(f"  > {area_label} : 데이터 부족으로 예측 및 평가 불가")
                continue

            X = subset_df[['month_index']]
            y = subset_df['수요량']
            
            print(f"\n--- {area_label} ---")
            
            for model_name, model in models.items():
                model.fit(X, y)
                
                # 예측
                predicted_demand = model.predict([[predict_date_index]])[0]
                
                # 성능 평가
                y_pred = model.predict(X)
                rmse = np.sqrt(mean_squared_error(y, y_pred))
                r2 = r2_score(y, y_pred)
                mae = mean_absolute_error(y, y_pred)

                # 결과 저장
                all_results[model_name][rent_type][area_label] = int(predicted_demand)

                print(f"  [{model_name}] 예측: {int(predicted_demand)}건")
            print(f"  - 성능: RMSE={rmse:.2f}, R²={r2:.2f}, MAE={mae:.2f}")

    # --- 결과 시각화 ---
    for model_name in models.keys():
        fig, axes = plt.subplots(1, 2, figsize=(18, 7))
        fig.suptitle(f'2025년 10월 {model_name} 예측 (면적 구간별)', fontsize=18)

        # 전세 예측 결과 시각화
        jeonse_demands = [all_results[model_name]['전세'].get(label, 0) for label in labels]
        axes[0].bar(labels, jeonse_demands, color='skyblue')
        axes[0].set_title('전세 수요량', fontsize=15)
        axes[0].set_xlabel('전용면적 (㎡)', fontsize=12)
        axes[0].set_ylabel('예상 거래량', fontsize=12)
        axes[0].tick_params(axis='x', rotation=45)

        # 월세 예측 결과 시각화
        wolse_demands = [all_results[model_name]['월세'].get(label, 0) for label in labels]
        axes[1].bar(labels, wolse_demands, color='salmon')
        axes[1].set_title('월세 수요량', fontsize=15)
        axes[1].set_xlabel('전용면적 (㎡)', fontsize=12)
        axes[1].set_ylabel('예상 거래량', fontsize=12)
        axes[1].tick_params(axis='x', rotation=45)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        # plt.show()
        return all_results


# 1. RandomForestRegressor
# 2. LinearRegression
# 3. DecisionTreeRegressor
# 4. GradientBoostingRegressor

# 1. RandomForestRegressor
# 2. LinearRegression
# 3. DecisionTreeRegressor
# 4. GradientBoostingRegressor
def performance_test(testType):
    """
    회귀 모델을 사용하여 아파트 전월세 수요를 예측하고,
    결과를 딕셔너리로 반환합니다.
    """
    setup_matplotlib_fonts()

    try:
        df = FileUtils().getCsv("아파트(전월세)_실거래가_all.csv")
    except (UnicodeDecodeError, FileNotFoundError):
        print("CSV 파일 읽기 실패")
        return {}

    # 날짜 생성
    df['거래일'] = pd.to_datetime(df['계약년월'].astype(str) +
                                  df['계약일'].astype(str).str.zfill(2),
                                  format='%Y%m%d')

    # 면적 구간
    bins = [0, 40, 60, 85, 100, 135, 200]
    labels = ['~40㎡', '40~60㎡', '60~85㎡', '85~100㎡', '100~135㎡', '135㎡~']
    df['면적_구간'] = pd.cut(df['전용면적(㎡)'], bins=bins, labels=labels, right=False)

    df['월'] = df['거래일'].dt.to_period('M')
    monthly_demand = df.groupby(['월', '전월세구분', '면적_구간']).size().reset_index(name='수요량')
    monthly_demand['월'] = monthly_demand['월'].dt.to_timestamp()

    # month_index
    monthly_demand_full = monthly_demand.groupby('월').size().reset_index()
    monthly_demand_full['month_index'] = np.arange(len(monthly_demand_full))
    monthly_demand = pd.merge(monthly_demand, monthly_demand_full[['월', 'month_index']],
                              on='월', how='left')

    # 미래 3개월
    predict_dates = pd.date_range(start='2025-09', end='2025-10', freq='MS')
    min_date = monthly_demand['월'].min()
    predict_date_indices = [(d.year - min_date.year) * 12 + (d.month - min_date.month)
                            for d in predict_dates]

    final_results = {'전세': [], '월세': []}

    for rent_type in ['전세', '월세']:
        for area_label in labels:
            subset_df = monthly_demand[(monthly_demand['전월세구분'] == rent_type) &
                                       (monthly_demand['면적_구간'] == area_label)].copy()
            if len(subset_df) < 6:
                continue

            X = subset_df[['month_index']]
            y = subset_df['수요량']

            test_size = 5
            X_train = X[:-test_size]
            X_test = X[-test_size:]
            y_train = y[:-test_size]
            y_test = y[-test_size:]

            # 모델 선택
            if testType == "5":
                model = RandomForestRegressor(random_state=42)
            elif testType == "6":
                model = LinearRegression()
            elif testType == "7":
                model = DecisionTreeRegressor(random_state=42)
            elif testType == "8":
                model = GradientBoostingRegressor(random_state=42)

            model.fit(X_train, y_train)

            # 훈련/테스트 예측
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            full_y_pred = np.concatenate([y_train_pred, y_test_pred])

            # 미래 예측
            predicted_demand_future = model.predict(np.array(predict_date_indices).reshape(-1, 1))

            # === 길이 맞추기 ===
            combined_y_pred = np.concatenate([full_y_pred, predicted_demand_future])
            combined_months = list(subset_df['월'].dt.strftime("%Y-%m")) + list(predict_dates.strftime("%Y-%m"))

            # 실제값은 과거값 + 미래(None)
            combined_y_actual = list(y.astype(int)) + [None] * len(predict_dates)

            # 성능 지표
            rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
            r2_train = r2_score(y_train, y_train_pred)
            mae_train = mean_absolute_error(y_train, y_train_pred)

            rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
            r2_test = r2_score(y_test, y_test_pred)
            mae_test = mean_absolute_error(y_test, y_test_pred)
            overfitting_status = '과적합 의심' if r2_train - r2_test > 0.2 else '양호'

            # 차트
            plt.figure(figsize=(10, 6))
            plt.plot(subset_df['월'], y, label='실제 수요량', color='blue', marker='o')
            plt.plot(subset_df['월'], full_y_pred, label='예측 수요량', color='red', linestyle='--')
            plt.plot(predict_dates, predicted_demand_future, label='미래 예측', color='green', marker='x')
            plt.title(f'{area_label} {rent_type} 예측')
            plt.xlabel('월')
            plt.ylabel('거래량')
            plt.legend()
            plt.grid(True)
            image_filename = f"linear_regression_{rent_type}_{area_label}.png"
            image_url = FileUtils().savePngToPath(image_filename)

            # 결과 저장
            area_result = {
                area_label: [
                    {"훈련성능": f"RMSE={rmse_train:.2f}, R²={r2_train:.2f}, MAE={mae_train:.2f}"},
                    {"테스트 성능": f"RMSE={rmse_test:.2f}, R²={r2_test:.2f}, MAE={mae_test:.2f}"},
                    {"과적합 여부": overfitting_status},
                    {"실제값 리스트": combined_y_actual},   # 미래 구간 None 추가
                    {"예측값 리스트": [int(val) for val in combined_y_pred]},  # 미래 예측 포함
                    {"image_url": image_url},
                    {"년-월": combined_months},  # 과거+미래 모두 포함
                ]
            }
            final_results[rent_type].append(area_result)

    # 병합
    jeonse = extract_full(final_results, "전세")
    wolse = extract_full(final_results, "월세")
    merged = {}
    for area in jeonse.keys():
        merged[area] = {"jeonse": jeonse[area], "wolse": wolse[area]}

    return merged


def extract_full(data,category: str):
    """
    category: "전세" 또는 "월세"
    """
    # data = { ... } 
    results = {}
    for item in data[category]:
        for size, details in item.items():
            entry = {}
            for d in details:
         
                if "훈련성능" in d:
                    entry["훈련성능"] = d["훈련성능"]
                if "테스트 성능" in d:
                    entry["테스트 성능"] = d["테스트 성능"]
                if "과적합 여부" in d:
                    entry["과적합 여부"] = d["과적합 여부"]
                if "실제값 리스트" in d:
                    entry["실제값 리스트"] = d["실제값 리스트"]
                if "예측값 리스트" in d:
                    entry["예측값 리스트"] = d["예측값 리스트"]
                if "년-월" in d:
                    entry["년-월"] = d["년-월"]
                if "image_url" in d:
                    entry["image_url"] = d["image_url"]
            results[size] = entry
    return results

# # 전세 / 월세 결과 추출
# jeonse_results = extract_full("전세")
# wolse_results = extract_full("월세")

# # 출력
# print(" 전세 결과")
# for size, vals in jeonse_results.items():
#     print(f"[{size}]")
#     print("훈련성능:", vals["훈련성능"])
#     print("테스트 성능:", vals["테스트 성능"])
#     print("과적합 여부:", vals["과적합 여부"])
#     print("실제값:", vals["실제값 리스트"])
#     print("예측값:", vals["예측값 리스트"])
#     print()

# print(" 월세 결과")
# for size, vals in wolse_results.items():
#     print(f"[{size}]")
#     print("훈련성능:", vals["훈련성능"])
#     print("테스트 성능:", vals["테스트 성능"])
#     print("과적합 여부:", vals["과적합 여부"])
#     print("실제값:", vals["실제값 리스트"])
#     print("예측값:", vals["예측값 리스트"])
#     print()

def engine(actionType):
    print(f" ACTION TYPE ====== {actionType}")
    if actionType == "1" :
        # LinearRegression(선형 회귀)
        return LRegression()
    elif actionType == "2" :  
        #DecisionTreeRegressor
        return DTreeRegressor()
    elif actionType == "3" :
        #RandomForestRegressor
        return RFRegressor()   
    elif actionType == "4" :
        return GBRegressor()
    else  :
        # 1. RandomForestRegressor
        # 2. LinearRegression
        # 3. DecisionTreeRegressor
        # 4. GradientBoostingRegressor
        return performance_test(actionType)
    

