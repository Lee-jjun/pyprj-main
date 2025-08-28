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
# def engine1():
#     ##################################################
#     # 1. 데이터 불러오기 및 전처리
#     ##################################################

#     # 1. CSV 파일 불러오기
#     try:
#         df = FileUtils().getCsv("아파트(전월세)_실거래가_all.csv")
#     except UnicodeDecodeError as un:
#         df = FileUtils().getCsv("아파트(전월세)_실거래가_all.csv")
#     except FileNotFoundError as e:
#         print(f"파일을 찾을 수 없습니다: {e.filename}")
#         # exit()

#     df.info()
#     # 2. 데이터 전처리
#     # '계약년월'과 '계약일' 컬럼을 합쳐 '거래일' 컬럼 생성
#     df['거래일'] = pd.to_datetime(df['계약년월'].astype(str) + df['계약일'].astype(str).str.zfill(2), format='%Y%m%d')

#     # '전용면적(㎡)'을 특정 구간으로 분류 (예: 10㎡ 단위)
#     bins = [0, 40, 60, 85, 100, 135, 200]
#     labels = ['~40㎡', '40~60㎡', '60~85㎡', '85~100㎡', '100~135㎡', '135㎡~']
#     df['면적_구간'] = pd.cut(df['전용면적(㎡)'], bins=bins, labels=labels, right=False)

#     # '전월세구분' 컬럼을 '전세'와 '월세'로 구분
#     df_jeonse = df[df['전월세구분'] == '전세'].copy()
#     df_wolse = df[df['전월세구분'] == '월세'].copy()

#     print("데이터 전처리 완료. 각 데이터프레임의 상위 5개 행:")
#     print("전세 데이터:")
#     print(df_jeonse.head())
#     print("\n월세 데이터:")
#     print(df_wolse.head())



#     ##################################################
#     # 2. 전용면적에 따른 수요량 분석 및 시각화
#     ##################################################
#     # 3. 전용면적별 전/월세 수요량 분석
#     # 면적 구간별 거래 건수 계산
#     demand_jeonse = df_jeonse.groupby('면적_구간').size().reset_index(name='전세_수요량')
#     demand_wolse = df_wolse.groupby('면적_구간').size().reset_index(name='월세_수요량')

#     # plot_paths = [] 
#     dict = {}
#     dict['면적_구간_jeonse'] = demand_jeonse['면적_구간']
#     dict['수요량_wolse']  = demand_wolse['월세_수요량']
#     # 운영체제에 맞는 한글 폰트 설정
#     # Windows
#     # plt.rcParams['font.family'] = 'Malgun Gothic'
#     # Mac OS
#     #plt.rcParams['font.family'] = 'AppleGothic'

#     # 마이너스 부호 깨짐 방지
#     # plt.rcParams['axes.unicode_minus'] = False
#     setup_matplotlib_fonts()
#     # 폰트 설정 확인 (선택 사항)
#     # print(plt.rcParams['font.family'])

#     # 4. 시각화
#     plt.figure(figsize=(15, 6))


#     # 전세 수요량 막대 그래프
#     plt.subplot(1, 2, 1)
#     plt.bar(demand_jeonse['면적_구간'], demand_jeonse['전세_수요량'], color='skyblue')
#     plt.title('전용면적에 따른 전세 수요량')
#     plt.xlabel('전용면적 (㎡)')
#     plt.ylabel('거래량')
#     plt.xticks(rotation=45, ha='right')
#     plt.tight_layout()
#     # filename = FileUtils().FilePathName("price_for_per_area_jeonse.png")
#     # plt.savefig(filename)
#     # plt.close()
#     # plot_paths.append(f'/static/images/{filename}')
    

#     dict['면적_구간_wolse'] = demand_wolse['면적_구간']
#     dict['월세_wolse']  = demand_wolse['월세_수요량']
#     # 월세 수요량 막대 그래프
#     plt.subplot(1, 2, 2)
#     plt.bar(demand_wolse['면적_구간'], demand_wolse['월세_수요량'], color='salmon')
#     plt.title('전용면적에 따른 월세 수요량')
#     plt.xlabel('전용면적 (㎡)')
#     plt.ylabel('거래량')
#     plt.xticks(rotation=45, ha='right')
#     plt.tight_layout()
#     # plt.show()
#     # filename = FileUtils().FilePathName("price_for_per_area_wolse.png")
#     # plt.savefig(filename)
#     # plt.close()'
    
#     #차트이미지 생성하는 메소드호출 함 
#     filename = FileUtils().savePngToPath("price_for_per_area_jeonse_wolse.png",closeFlag=True)
#     print(f"=== 생성된 이미지 파일 위치 {filename}")
#     # plot_paths.append(filename)
#     dict['img_wolse']  = filename

#     ##################################################
#     # 3. 시계열 데이터 가공 및 10월 수요량 예측
#     ##################################################

#     # 5. 시계열 데이터 생성
#     # 월별 거래량 집계
#     monthly_demand_jeonse = df_jeonse.groupby(df_jeonse['거래일'].dt.to_period('M')).size().reset_index(name='전세_수요량')
#     monthly_wolse_demand = df_wolse.groupby(df_wolse['거래일'].dt.to_period('M')).size().reset_index(name='월세_수요량')

#     monthly_demand_jeonse['거래일'] = monthly_demand_jeonse['거래일'].dt.to_timestamp()
#     monthly_wolse_demand['거래일'] = monthly_wolse_demand['거래일'].dt.to_timestamp()

#     dict['거래일_jeonse'] = monthly_demand_jeonse['거래일']
#     dict['거래일_wolse'] = monthly_wolse_demand['거래일']

#     # 6. 예측 모델 학습 (선형 회귀)
#     # 데이터를 숫자형으로 변환 (월 순서)
#     monthly_demand_jeonse['month_index'] = np.arange(len(monthly_demand_jeonse))
#     monthly_wolse_demand['month_index'] = np.arange(len(monthly_wolse_demand))

#     # 예측할 월 (2025년 10월)의 인덱스 계산
#     predict_date = pd.to_datetime('2025-10-01')
#     last_date = monthly_demand_jeonse['거래일'].max()
#     predict_month_index = len(monthly_demand_jeonse) + (predict_date.year - last_date.year) * 12 + (predict_date.month - last_date.month)
#     print(f" ################################################# ")
#     print(f" predict_month_index {predict_month_index}")
#     print(f" ################################################# ")
#     # 전세 수요량 예측
#     X_jeonse = monthly_demand_jeonse[['month_index']]
#     y_jeonse = monthly_demand_jeonse['전세_수요량']
#     model_jeonse = LinearRegression()
#     model_jeonse.fit(X_jeonse, y_jeonse)
#     predicted_jeonse_demand = model_jeonse.predict([[predict_month_index]])

#     # 월세 수요량 예측
#     X_wolse = monthly_wolse_demand[['month_index']]
#     y_wolse = monthly_wolse_demand['월세_수요량']
#     model_wolse = LinearRegression()
#     model_wolse.fit(X_wolse, y_wolse)
#     predicted_wolse_demand = model_wolse.predict([[predict_month_index]])

#     # 7. 결과 출력
#     print(f"\n2025년 10월 전세 수요량 예측: {int(predicted_jeonse_demand[0])} 건")
#     print(f"2025년 10월 월세 수요량 예측: {int(predicted_wolse_demand[0])} 건")

#     # 예측 결과를 시각화
#     plt.figure(figsize=(12, 6))
#     plt.subplot(1, 2, 1)
#     plt.plot(monthly_demand_jeonse['거래일'], monthly_demand_jeonse['전세_수요량'], marker='o', label='실제 전세 수요량')
#     plt.plot(predict_date, predicted_jeonse_demand, 'r*', markersize=10, label='예측치 (2025년 10월)')
#     plt.title('월별 전세 수요량 및 예측')
#     plt.xlabel('날짜')
#     plt.ylabel('거래량')
#     plt.legend()
#     plt.grid(True)


#     plt.subplot(1, 2, 2)
#     plt.plot(monthly_wolse_demand['거래일'], monthly_wolse_demand['월세_수요량'], marker='o', label='실제 월세 수요량')
#     plt.plot(predict_date, predicted_wolse_demand, 'r*', markersize=10, label='예측치 (2025년 10월)')
#     plt.title('월별 월세 수요량 및 예측')
#     plt.xlabel('날짜')
#     plt.ylabel('거래량')
#     plt.legend()
#     plt.grid(True)
    
#     # 이미지 파일로 저장
#     print("6. 구별, 금액대별 수요량 분석 및 시각화 성공.")
#     # plt.tight_layout()
#     # plt.show()
#     filename = FileUtils().savePngToPath("price_for_per_area_jeonse_wolse-1.png",closeFlag=True)
#     # plot_paths.append(filename)
#     dict['img_jeonse']  = filename
#     return dict


# LinearRegression(선형 회귀)
def LRegression():

    setup_matplotlib_fonts()
    ##################################################
    # 1. 데이터 불러오기 및 전처리
    ##################################################
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

    print("--- 면적 구간별 LinearRegression 예측 결과 및 성능 지표 ---")

    # 예측 결과를 담을 딕셔너리
    predicted_results = {'전세': {}, '월세': {}}
    performance_metrics = {'전세': {}, '월세': {}}


    # 전/월세 및 면적 구간별로 반복
    for rent_type in ['전세', '월세']:
        print(f"\n--- {rent_type} ---")
        for area_label in labels:
            subset_df = monthly_demand[(monthly_demand['전월세구분'] == rent_type) & (monthly_demand['면적_구간'] == area_label)].copy()

            if len(subset_df) < 2:
                print(f"  > {area_label} : 데이터 부족으로 예측 불가")
                continue

            X = subset_df[['month_index']]
            y = subset_df['수요량']
            
            # LinearRegression 모델 학습
            model = LinearRegression()
            model.fit(X, y)
            
            # 2025년 10월 예측
            predicted_demand = model.predict([[predict_date_index]])[0]
            
            # 예측 성능 평가
            y_pred = model.predict(X)
            rmse = np.sqrt(mean_squared_error(y, y_pred))
            r2 = r2_score(y, y_pred)
            mae = mean_absolute_error(y, y_pred)
            
            # 결과 저장 및 출력
            predicted_results[rent_type][area_label] = int(predicted_demand)
            performance_metrics[rent_type][area_label] = {'RMSE': f"{rmse:.2f}", 'R2': f"{r2:.2f}", 'MAE': f"{mae:.2f}"}
            
            print(f"  > {area_label} 예측: {int(predicted_demand)}건")
            print(f"  > {area_label} 성능 지표: RMSE={rmse:.2f}, R²={r2:.2f}, MAE={mae:.2f}")

    # --- 결과 시각화 ---
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # 전세 예측 결과 시각화
    jeonse_demands = [predicted_results['전세'].get(label, 0) for label in labels]
    axes[0].bar(labels, jeonse_demands, color='skyblue')
    axes[0].set_title('2025년 10월 전세 수요량 예측 (면적 구간별)', fontsize=15)
    axes[0].set_xlabel('전용면적 (㎡)', fontsize=12)
    axes[0].set_ylabel('예상 거래량', fontsize=12)
    axes[0].tick_params(axis='x', rotation=45)

    # 월세 예측 결과 시각화
    wolse_demands = [predicted_results['월세'].get(label, 0) for label in labels]
    axes[1].bar(labels, wolse_demands, color='salmon')
    axes[1].set_title('2025년 10월 월세 수요량 예측 (면적 구간별)', fontsize=15)
    axes[1].set_xlabel('전용면적 (㎡)', fontsize=12)
    axes[1].set_ylabel('예상 거래량', fontsize=12)
    axes[1].tick_params(axis='x', rotation=45)

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
        # exit()
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

    print("--- 면적 구간별 Decision Tree Regression 예측 결과 및 성능 지표 ---")

    # 예측 결과를 담을 딕셔너리
    predicted_results = {'전세': {}, '월세': {}}
    performance_metrics = {'전세': {}, '월세': {}}
    analysis_metrics = {'전세': {}, '월세': {}}

    # 전/월세 및 면적 구간별로 반복
    for rent_type in ['전세', '월세']:
        print(f"\n--- {rent_type} ---")
        for area_label in labels:
            subset_df = monthly_demand[(monthly_demand['전월세구분'] == rent_type) & (monthly_demand['면적_구간'] == area_label)].copy()

            if len(subset_df) < 2:
                print(f"  > {area_label} : 데이터 부족으로 예측 불가")
                continue

            X = subset_df[['month_index']]
            y = subset_df['수요량']
            
            # DecisionTreeRegressor 모델 학습
            model = DecisionTreeRegressor(random_state=42)
            model.fit(X, y)
            
            # 2025년 10월 예측
            predicted_demand = model.predict([[predict_date_index]])[0]
            
            # 예측 성능 평가
            y_pred = model.predict(X)
            rmse = np.sqrt(mean_squared_error(y, y_pred))
            r2 = r2_score(y, y_pred)
            mae = mean_absolute_error(y, y_pred)
            
            # 특성 중요도 (여기서는 'month_index' 하나이므로 1.0)
            feature_importance = model.feature_importances_[0]

            # 결과 저장 및 출력
            predicted_results[rent_type][area_label] = int(predicted_demand)
            performance_metrics[rent_type][area_label] = {'RMSE': f"{rmse:.2f}", 'R2': f"{r2:.2f}", 'MAE': f"{mae:.2f}"}
            analysis_metrics[rent_type][area_label] = {'특성 중요도': feature_importance}
            
            print(f"  > {area_label} 예측: {int(predicted_demand)}건")
            print(f"  > {area_label} 성능 지표: RMSE={rmse:.2f}, R²={r2:.2f}, MAE={mae:.2f}")

    # --- 결과 시각화 ---
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # 전세 예측 결과 시각화
    jeonse_demands = [predicted_results['전세'].get(label, 0) for label in labels]
    axes[0].bar(labels, jeonse_demands, color='skyblue')
    axes[0].set_title('2025년 10월 전세 수요량 예측 (면적 구간별)', fontsize=15)
    axes[0].set_xlabel('전용면적 (㎡)', fontsize=12)
    axes[0].set_ylabel('예상 거래량', fontsize=12)
    axes[0].tick_params(axis='x', rotation=45)

    # 월세 예측 결과 시각화
    wolse_demands = [predicted_results['월세'].get(label, 0) for label in labels]
    axes[1].bar(labels, wolse_demands, color='salmon')
    axes[1].set_title('2025년 10월 월세 수요량 예측 (면적 구간별)', fontsize=15)
    axes[1].set_xlabel('전용면적 (㎡)', fontsize=12)
    axes[1].set_ylabel('예상 거래량', fontsize=12)
    axes[1].tick_params(axis='x', rotation=45)

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
        # exit()
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

    print("--- 면적 구간별 랜덤 포레스트 예측 결과 및 성능 지표 ---")

    # 예측 결과를 담을 딕셔너리
    predicted_results = {'전세': {}, '월세': {}}
    performance_metrics = {'전세': {}, '월세': {}}
    analysis_metrics = {'전세': {}, '월세': {}}

    # 전/월세 및 면적 구간별로 반복
    for rent_type in ['전세', '월세']:
        print(f"\n--- {rent_type} ---")
        for area_label in labels:
            subset_df = monthly_demand[(monthly_demand['전월세구분'] == rent_type) & (monthly_demand['면적_구간'] == area_label)].copy()

            if len(subset_df) < 2:
                print(f"  > {area_label} : 데이터 부족으로 예측 불가")
                continue

            X = subset_df[['month_index']]
            y = subset_df['수요량']
            
            # RandomForestRegressor 모델 학습
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X, y)
            
            # 2025년 10월 예측
            predicted_demand = model.predict([[predict_date_index]])[0]
            
            # 예측 성능 평가
            y_pred = model.predict(X)
            rmse = np.sqrt(mean_squared_error(y, y_pred))
            r2 = r2_score(y, y_pred)
            mae = mean_absolute_error(y, y_pred)

            # 개별 트리 예측값 분석
            individual_predictions = [tree.predict([[predict_date_index]])[0] for tree in model.estimators_]
            pred_mean = np.mean(individual_predictions)
            pred_std = np.std(individual_predictions)
            
            # 특성 중요도 (여기서는 'month_index' 하나이므로 1.0)
            feature_importance = model.feature_importances_[0]
            
            # 결과 저장 및 출력
            predicted_results[rent_type][area_label] = int(predicted_demand)
            performance_metrics[rent_type][area_label] = {'RMSE': f"{rmse:.2f}", 'R2': f"{r2:.2f}", 'MAE': f"{mae:.2f}"}
            analysis_metrics[rent_type][area_label] = {'예측값 평균': pred_mean, '예측값 표준편차': pred_std, '특성 중요도': feature_importance}
            
            print(f"  > {area_label} 예측: {int(predicted_demand)}건")
            print(f"  > {area_label} 성능 지표: RMSE={rmse:.2f}, R²={r2:.2f}, MAE={mae:.2f}")
            print(f"  > {area_label} 예측 신뢰도 분석: 개별 예측 평균={pred_mean:.2f}, 표준편차={pred_std:.2f}")

    # --- 결과 시각화 ---
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # 전세 예측 결과 시각화
    jeonse_demands = [predicted_results['전세'].get(label, 0) for label in labels]
    axes[0].bar(labels, jeonse_demands, color='skyblue')
    axes[0].set_title('2025년 10월 전세 수요량 예측 (면적 구간별)', fontsize=15)
    axes[0].set_xlabel('전용면적 (㎡)', fontsize=12)
    axes[0].set_ylabel('예상 거래량', fontsize=12)
    axes[0].tick_params(axis='x', rotation=45)

    # 월세 예측 결과 시각화
    wolse_demands = [predicted_results['월세'].get(label, 0) for label in labels]
    axes[1].bar(labels, wolse_demands, color='salmon')
    axes[1].set_title('2025년 10월 월세 수요량 예측 (면적 구간별)', fontsize=15)
    axes[1].set_xlabel('전용면적 (㎡)', fontsize=12)
    axes[1].set_ylabel('예상 거래량', fontsize=12)
    axes[1].tick_params(axis='x', rotation=45)

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

    print("--- 면적 구간별 Gradient Boosting Regression 예측 결과 및 성능 지표 ---")

    # 예측 결과를 담을 딕셔너리
    predicted_results = {'전세': {}, '월세': {}}
    performance_metrics = {'전세': {}, '월세': {}}
    analysis_metrics = {'전세': {}, '월세': {}}

    # 전/월세 및 면적 구간별로 반복
    for rent_type in ['전세', '월세']:
        print(f"\n--- {rent_type} ---")
        for area_label in labels:
            subset_df = monthly_demand[(monthly_demand['전월세구분'] == rent_type) & (monthly_demand['면적_구간'] == area_label)].copy()

            if len(subset_df) < 2:
                print(f"  > {area_label} : 데이터 부족으로 예측 불가")
                continue

            X = subset_df[['month_index']]
            y = subset_df['수요량']
            
            # GradientBoostingRegressor 모델 학습
            model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
            model.fit(X, y)
            
            # 2025년 10월 예측
            predicted_demand = model.predict([[predict_date_index]])[0]
            
            # 예측 성능 평가
            y_pred = model.predict(X)
            rmse = np.sqrt(mean_squared_error(y, y_pred))
            r2 = r2_score(y, y_pred)
            mae = mean_absolute_error(y, y_pred)
            
            # 특성 중요도
            feature_importance = model.feature_importances_[0]

            # 결과 저장 및 출력
            predicted_results[rent_type][area_label] = int(predicted_demand)
            performance_metrics[rent_type][area_label] = {'RMSE': f"{rmse:.2f}", 'R2': f"{r2:.2f}", 'MAE': f"{mae:.2f}"}            
            analysis_metrics[rent_type][area_label] = {'특성 중요도': feature_importance}
            
            print(f"  > {area_label} 예측: {int(predicted_demand)}건")
            print(f"  > {area_label} 성능 지표: RMSE={rmse:.2f}, R²={r2:.2f}, MAE={mae:.2f}")

    # --- 결과 시각화 ---
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # 전세 예측 결과 시각화
    jeonse_demands = [predicted_results['전세'].get(label, 0) for label in labels]
    axes[0].bar(labels, jeonse_demands, color='skyblue')
    axes[0].set_title('2025년 10월 전세 수요량 예측 (면적 구간별)', fontsize=15)
    axes[0].set_xlabel('전용면적 (㎡)', fontsize=12)
    axes[0].set_ylabel('예상 거래량', fontsize=12)
    axes[0].tick_params(axis='x', rotation=45)

    # 월세 예측 결과 시각화
    wolse_demands = [predicted_results['월세'].get(label, 0) for label in labels]
    axes[1].bar(labels, wolse_demands, color='salmon')
    axes[1].set_title('2025년 10월 월세 수요량 예측 (면적 구간별)', fontsize=15)
    axes[1].set_xlabel('전용면적 (㎡)', fontsize=12)
    axes[1].set_ylabel('예상 거래량', fontsize=12)
    axes[1].tick_params(axis='x', rotation=45)

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
def performance_test(testType):
    """
    Linear Regression 모델을 사용하여 아파트 전월세 수요를 예측하고, 
    결과를 특정 형식의 딕셔너리로 반환하는 함수입니다.
    """
    ##################################################
    # 1. 데이터 불러오기 및 전처리
    ##################################################
    setup_matplotlib_fonts()
    # 1. CSV 파일 불러오기
    try:
        df = FileUtils().getCsv("아파트(전월세)_실거래가_all.csv")
    except UnicodeDecodeError as un:
        df = FileUtils().getCsv("아파트(전월세)_실거래가_all.csv")
    except FileNotFoundError as e:
        print(f"파일을 찾을 수 없습니다: {e.filename}")
   
  
    df['거래일'] = pd.to_datetime(df['계약년월'].astype(str) + df['계약일'].astype(str).str.zfill(2), format='%Y%m%d')
    bins = [0, 40, 60, 85, 100, 135, 200]
    labels = ['~40㎡', '40~60㎡', '60~85㎡', '85~100㎡', '100~135㎡', '135㎡~']
    df['면적_구간'] = pd.cut(df['전용면적(㎡)'], bins=bins, labels=labels, right=False)
    df['월'] = df['거래일'].dt.to_period('M')
    monthly_demand = df.groupby(['월', '전월세구분', '면적_구간']).size().reset_index(name='수요량')
    print("#####################################")
    print(monthly_demand)
    print("#####################################")
    monthly_demand['월'] = monthly_demand['월'].dt.to_timestamp()
    monthly_demand_full = monthly_demand.groupby('월').size().reset_index()
    monthly_demand_full['month_index'] = np.arange(len(monthly_demand_full))
    monthly_demand = pd.merge(monthly_demand, monthly_demand_full[['월', 'month_index']], on='월', how='left')

    # 예측할 월 (2025년 10월)의 인덱스 계산
    predict_date_index = (pd.to_datetime('2025-10').year - monthly_demand['월'].min().year) * 12 + (pd.to_datetime('2025-10').month - monthly_demand['월'].min().month)

    print("--- 면적 구간별 Linear Regression 예측 결과 및 성능 지표 ---")

    # 최종 결과를 담을 딕셔너리
    final_results = {'전세': [], '월세': []}

    for rent_type in ['전세', '월세']:
        print(f"\n==================== <<{rent_type}>> ====================")
        for area_label in labels:
            subset_df = monthly_demand[(monthly_demand['전월세구분'] == rent_type) & (monthly_demand['면적_구간'] == area_label)].copy()
            
            print("========== 데이터 정보 ==========")
            print(subset_df)
            # print(f"{list(subset_df["월"].dt.strftime("%Y-%m"))}")
            print(subset_df.info())
            print("========== 데이터 정보 ==========")
            if len(subset_df) < 6:
                print(f"  > {area_label} : 데이터 부족으로 예측 및 평가 불가")
                continue

            X = subset_df[['month_index']]
            y = subset_df['수요량']
          
            print(" === X값['month_index'] ==")
            print(X)
            print(" === y값['수요량']  ==")
            print(y)
            # 시계열 데이터 분할 (마지막 3개월을 테스트 데이터로 사용)
            test_size = 3
            X_train = X[:-test_size]
            X_test = X[-test_size:]
            y_train = y[:-test_size]
            y_test = y[-test_size:]
            
            print(" 시계열 데이터 분할 (마지막 3개월을 테스트 데이터로 사용)")
            print(f"X_train ={X_train}")
            print(f"X_test ={X_test}")
            print(f"y_train ={y_train}")
            print(f"y_test ={y_test}")
            

            # 1. RandomForestRegressor
            # 2. LinearRegression
            # 3. DecisionTreeRegressor
            # 4. GradientBoostingRegressor
            if testType == "5" :
                model = RandomForestRegressor(random_state=42)
            elif testType == "6" :
                model = LinearRegression()
            elif testType == "7" :
                model = DecisionTreeRegressor(random_state=42)
            elif testType == "8" :
                model = GradientBoostingRegressor(random_state=42)   
                 
            model.fit(X_train, y_train)
            
            predicted_demand_future = model.predict([[predict_date_index]])[0]
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            full_y_pred = np.concatenate([y_train_pred, y_test_pred])

            # 성능 평가
            rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
            r2_train = r2_score(y_train, y_train_pred)
            mae_train = mean_absolute_error(y_train, y_train_pred)
            
            rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
            r2_test = r2_score(y_test, y_test_pred)
            mae_test = mean_absolute_error(y_test, y_test_pred)
            
            overfitting_status = '과적합 의심' if r2_train > r2_test and r2_train - r2_test > 0.2 else '양호'

            print(f"\n--- {area_label} ---")
            print(f"  [Linear Regression] 2025년 10월 예측: {int(predicted_demand_future)}건")
            print(f"  - 훈련 성능: RMSE={rmse_train:.2f}, R²={r2_train:.2f}, MAE={mae_train:.2f}")
            print(f"  - 테스트 성능: RMSE={rmse_test:.2f}, R²={r2_test:.2f}, MAE={mae_test:.2f}")
            print(f'  - "년-월": { list(subset_df["월"].dt.strftime("%Y-%m"))}')
            print(f"  - 과적합 여부: {overfitting_status}")

            # 예측값과 실제값 차트 시각화 및 저장
            plt.figure(figsize=(10, 6))
            plt.plot(subset_df['월'], y, label='실제 수요량', color='blue', marker='o')
            plt.plot(subset_df['월'], full_y_pred, label='예측 수요량', color='red', linestyle='--')
            plt.title(f'{area_label} {rent_type} - Linear Regression 예측 및 성능')
            plt.xlabel('날짜')
            plt.ylabel('거래량')
            plt.legend()
            plt.grid(True)

            image_filename = f"linear_regression_{rent_type}_{area_label}.png"
            image_url = FileUtils().savePngToPath(image_filename)
           
            # 요청된 형식으로 데이터 구조화
            area_result = {
                area_label: [
                    {"훈련성능"     : f"RMSE={rmse_train:.2f}, R²={r2_train:.2f}, MAE={mae_train:.2f}"},
                    {"테스트 성능"  : f"RMSE={rmse_test:.2f}, R²={r2_test:.2f}, MAE={mae_test:.2f}"},
                    {"과적합 여부"  : overfitting_status},
                    {"실제값 리스트": list(y.astype(int))},
                    {"예측값 리스트": [int(val) for val in full_y_pred]},
                    {"image_url"  : image_url},
                    {"년-월"       : list(subset_df["월"].dt.strftime("%Y-%m"))},
                ]
            }
            
            final_results[rent_type].append(area_result)
      
    # 최종 2025년 10월 예측 결과 시각화
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle('2025년 10월 Linear Regression 최종 예측 (면적 구간별)', fontsize=18)
    
    predicted_demands_jeonse = {label: 0 for label in labels}
    predicted_demands_wolse = {label: 0 for label in labels}
    try:    
        for result in final_results['전세']:
            for area, data in result.items():
                predicted_demands_jeonse[area] = int(data[4]['예측값 리스트'][-1])
    except Exception as e:
        print("예외 발생:", str(e))  # 에러 메시지 출력
        traceback.print_exc()       # 전체 스택 추적 출력
        
    for result in final_results['월세']:
        for area, data in result.items():
            predicted_demands_wolse[area] = int(data[4]['예측값 리스트'][-1])

    axes[0].bar(labels, [predicted_demands_jeonse[label] for label in labels], color='skyblue')
    axes[0].set_title('전세 수요량', fontsize=15)
    axes[0].set_xlabel('전용면적 (㎡)', fontsize=12)
    axes[0].set_ylabel('예상 거래량', fontsize=12)
    axes[0].tick_params(axis='x', rotation=45)

    axes[1].bar(labels, [predicted_demands_wolse[label] for label in labels], color='salmon')
    axes[1].set_title('월세 수요량', fontsize=15)
    axes[1].set_xlabel('전용면적 (㎡)', fontsize=12)
    axes[1].set_ylabel('예상 거래량', fontsize=12)
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # 최종 예측 이미지 저장 및 반환
    # final_image_url = FileUtils().savePngToPath("final_linear_regression.png")
    jeonse = extract_full(final_results,"전세")
    wolse  = extract_full(final_results,"월세")

    merged = {}
    for area in jeonse.keys():
        merged[area] = {
            "jeonse": jeonse[area],
            "wolse": wolse[area]
        }
    # print(json.dumps(merged, ensure_ascii=False, indent=2))
    
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
    

