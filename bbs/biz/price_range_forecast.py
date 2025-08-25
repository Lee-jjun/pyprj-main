# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.cluster import KMeans
import matplotlib.font_manager as fm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import time
# CRITICAL FIX: Matplotlib을 비대화형 백엔드로 설정합니다.
# 이렇게 하면 "main thread is not in main loop" 오류를 방지할 수 있습니다.
# Note: This is a critical fix for environments without a main GUI thread.
mpl.use('Agg')

def resultData():
    resultData = {}
    
    # 이 함수는 이제 생성된 플롯 이미지의 경로 목록을 반환합니다.
    # 생성된 이미지의 경로를 저장할 목록
    plot_paths = []

    try:
        print("1. 데이터 파일 경로 확인 및 불러오기 시작...")
        # 플롯을 저장할 경로
        base_dir = os.path.dirname(os.path.abspath(__file__))
        img_dir = os.path.join(base_dir, '..', 'static', 'images')
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)
            
        # 데이터 불러오기 및 전처리
        # 데이터 파일 경로
        file_path = os.path.join(base_dir, '..', 'csv', 'apartment_rent_data_price.csv')
        df = pd.read_csv(file_path, sep=';', encoding='utf-8')
        print("1. 데이터 파일 불러오기 성공.")
        
        print("2. '구' 정보 추출 및 전처리 시작...")
        # '시군구'에서 '구' 정보만 추출합니다.
        df['구'] = df['시군구'].apply(lambda x: x.split()[1] if len(x.split()) > 1 else None)
        df.dropna(subset=['구'], inplace=True)
        print("2. '구' 정보 추출 및 전처리 성공.")
        
        print("3. 필요한 열 선택 및 전처리 시작...")
        # 필요한 열만 선택합니다.
        df = df[['시군구', '전월세구분', '보증금(만원)', '월세금(만원)', '계약년월', '구']]
        
        # 보증금과 월세 열에서 쉼표를 제거하고 숫자형으로 변환합니다.
        df['보증금(만원)'] = pd.to_numeric(df['보증금(만원)'].str.replace(',', '', regex=True).fillna('0'))
        df['월세금(만원)'] = pd.to_numeric(df['월세금(만원)'].str.replace(',', '', regex=True).fillna('0'))
        print("3. 필요한 열 선택 및 전처리 성공.")
        
        print("4. 환산보증금 계산 시작...")
        # '월세' 유형의 경우 보증금을 보정합니다.
        df['환산보증금'] = df.apply(
            lambda row: row['보증금(만원)'] + row['월세금(만원)'] * 100 if row['전월세구분'] == '월세' else row['보증금(만원)'], axis=1
        )
        print("4. 환산보증금 계산 성공.")
        
        print("5. 데이터 클리닝 및 금액대 분류 시작...")
        # 비정상적인 극단값(outlier) 제거
        df = df[df['환산보증금'] <= 200000]

        # 금액대별 분류 (1억원 단위)
        bins = list(range(0, int(df['환산보증금'].max()) + 10000, 10000))
        labels = [f'{i // 10000}억~{(i + 10000) // 10000}억' for i in range(0, int(df['환산보증금'].max()) + 10000, 10000)[:-1]]
        df['금액대'] = pd.cut(df['환산보증금'], bins=bins, labels=labels, right=False)
        print("5. 데이터 클리닝 및 금액대 분류 성공.")
        
        print("6. 구별, 금액대별 수요량 분석 및 시각화 시작...")
        # 전세와 월세 데이터 분리
        df_jeonse = df[df['전월세구분'] == '전세'].copy()
        df_wolse = df[df['전월세구분'] == '월세'].copy()

        # 구별, 금액대별 수요량 (거래 건수) 계산
        demand_jeonse = df_jeonse.groupby(['구', '금액대']).size().unstack(fill_value=0)
        demand_wolse = df_wolse.groupby(['구', '금액대']).size().unstack(fill_value=0)
        
        # 전/월세 히트맵 시각화
        plt.rc('font', family='Malgun Gothic')
        mpl.rcParams['axes.unicode_minus'] = False
        plt.figure(figsize=(20, 10))
        plt.subplot(1, 2, 1)
        sns.heatmap(demand_jeonse, annot=True, fmt='d', cmap='YlGnBu')
        plt.title('구별 전세 금액대별 수요량 히트맵')
        plt.xlabel('금액대')
        plt.ylabel('구')

        plt.subplot(1, 2, 2)
        sns.heatmap(demand_wolse, annot=True, fmt='d', cmap='OrRd')
        plt.title('구별 월세 금액대별 수요량 히트맵')
        plt.xlabel('금액대(보증금+월세 환산)')
        plt.ylabel('구')
        plt.tight_layout()
        
        # 이미지 파일로 저장
        file_name_heatmap = 'demand_heatmaps.png'
        file_path_heatmap = os.path.join(img_dir, file_name_heatmap)
        plt.savefig(file_path_heatmap)
        plt.close()
        plot_paths.append(f'/static/images/{file_name_heatmap}')
        print("6. 구별, 금액대별 수요량 분석 및 시각화 성공.")
        
        print("7. 구별 수요 패턴 군집화 시작...")
        # '계약년월'을 날짜 형식으로 변환
        df['계약년월'] = pd.to_datetime(df['계약년월'].astype(str), format='%Y%m')

        # 구별, 금액대별 수요량 계산
        demand_by_gu = df.groupby(['구', '금액대']).size().unstack(fill_value=0)
        
        # 스케일링
        scaler = StandardScaler()
        scaled_demand = scaler.fit_transform(demand_by_gu)

        # K-Means 모델 학습
        optimal_k = 4
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init='auto')
        demand_by_gu['cluster'] = kmeans.fit_predict(scaled_demand)

        # PCA를 이용한 군집화 결과
        pca = PCA(n_components=2)
        reduced_data = pca.fit_transform(scaled_demand)

        # 군집 시각화
        plt.figure(figsize=(12, 10))
        sns.scatterplot(x=reduced_data[:, 0], y=reduced_data[:, 1], hue=demand_by_gu['cluster'], palette='viridis', s=200)
        plt.title('구별 수요량 패턴 군집화 결과 (K-Means + PCA)')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        for i, gu in enumerate(demand_by_gu.index):
            plt.text(reduced_data[i, 0] + 0.05, reduced_data[i, 1], gu)
            
        # 이미지 파일로 저장
        file_name_cluster = 'cluster_visualization.png'
        file_path_cluster = os.path.join(img_dir, file_name_cluster)
        plt.savefig(file_path_cluster)
        plt.close()
        plot_paths.append(f'/static/images/{file_name_cluster}')
        print("7. 구별 수요 패턴 군집화 성공.")
        
        print("8. Prophet 모델을 사용한 시계열 예측 및 시각화 시작...")
        # 구별 월별 총 거래 건수 시계열 데이터 생성
        time_series_data = df.groupby(['계약년월', '구']).size().reset_index(name='거래건수')

        # 각 구별로 예측 수행
        accuracy_results = {}
        gu_list = time_series_data['구'].unique()

        forecast_results = {}
        count = 0
        for gu in gu_list:
            gu_data = time_series_data[time_series_data['구'] == gu].rename(columns={'계약년월': 'ds', '거래건수': 'y'})
            
            if len(gu_data) > 3: # 예측에 충분한 데이터가 있는지 확인
                # 데이터 분할
                train_data = gu_data[:-3]
                test_data = gu_data[-3:]

                # 첫 번째 Prophet 모델 인스턴스: 정확도 계산용
                m1 = Prophet(seasonality_mode='multiplicative', daily_seasonality=False)
                m1.fit(train_data)
                
                # 3개월 예측
                future_test = m1.make_future_dataframe(periods=3, freq='M')
                forecast_test = m1.predict(future_test)

                # 테스트 기간의 예측값만 추출
                test_forecast = forecast_test['yhat'].tail(3)

                # 정확도 계산
                y_true = test_data['y'].values
                y_pred = test_forecast.values
                mae = mean_absolute_error(y_true, y_pred)
                rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100 if np.all(y_true != 0) else float('inf')
                accuracy_results[gu] = {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}

                # CRITICAL FIX: 최종 예측을 위해 새로운 Prophet 모델 인스턴스를 생성합니다.
                m2 = Prophet(seasonality_mode='multiplicative', daily_seasonality=False)
                m2.fit(gu_data)
                
                future_final = m2.make_future_dataframe(periods=3, freq='M')
                forecast_final = m2.predict(future_final)
                forecast_results[gu] = forecast_final[['ds', 'yhat']].tail(3)
                print(f"8-1. {count} Prophet #############################################...")
            else:
                forecast_results[gu] = pd.DataFrame(columns=['ds', 'yhat']) # 데이터 부족 시 빈 DF 저장
                print(f"8-2. {count} Prophet #############################################...")
            count +=1

        # 예측 결과 시각화
        plt.figure(figsize=(30, 20))
        num_rows = int(np.ceil(len(gu_list) / 5))
        num_cols = 5

        for i, gu in enumerate(gu_list):
            ax = plt.subplot(num_rows, num_cols, i + 1)
            gu_data = time_series_data[time_series_data['구'] == gu].rename(columns={'계약년월': 'ds', '거래건수': 'y'})
            
            if not forecast_results[gu].empty:
                forecast_df = forecast_results[gu]
                bar_data = pd.concat([gu_data, forecast_df.rename(columns={'yhat': 'y'})], ignore_index=True)
                bar_data['ds_str'] = bar_data['ds'].dt.strftime('%Y-%m')
                
                bar_data['type'] = '실제'
                bar_data.loc[bar_data['ds'].isin(forecast_df['ds']), 'type'] = '예측'
                
                sns.barplot(x='ds_str', y='y', hue='type', dodge=False, ax=ax, data=bar_data)
                ax.set_title(f'{gu} 거래량 예측', fontsize=20)
                ax.set_xlabel('날짜', fontsize=12)
                ax.set_ylabel('거래 건수', fontsize=12)
                ax.legend(title='데이터 유형', loc='upper left', fontsize=10)
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
            else:
                ax.text(0.5, 0.5, '데이터 부족', ha='center', va='center', fontsize=25, color='red')
                ax.set_title(f'{gu} 거래량', fontsize=20)
                ax.set_xlabel('날짜', fontsize=12)
                ax.set_ylabel('거래 건수', fontsize=12)
                ax.set_xticks([])
                ax.set_yticks([])
                plt.tight_layout()
            print("8-3. Prophet 모델을 사용한 시계열 예측 및 시각화 성공.")
        # 모든 플롯을 하나의 이미지 파일로 저장합니다.
        file_name_forecast = 'all_districts_forecast.png'
        file_path_forecast = os.path.join(img_dir, file_name_forecast)
        plt.savefig(file_path_forecast)
        plt.close()
        plot_paths.append(f'/static/images/{file_name_forecast}')
        print("9. Prophet 모델을 사용한 시계열 예측 및 시각화 성공.")
      
        # 결과 딕셔너리에 이미지 경로를 저장합니다.
        resultData['images'] = plot_paths
        
    except Exception as e:
        resultData["ERROR"] = f"오류가 발생했습니다: {e}"
        print(f"오류 발생: {e}")


    
    return resultData
