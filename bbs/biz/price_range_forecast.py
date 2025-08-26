import os
import io
import base64
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from django.shortcuts import render
import time
import json # JSON 처리를 위한 라이브러리 추가

# Matplotlib을 비대화형 백엔드로 설정합니다.
mpl.use('Agg')

def run_analysis():
    """
    이 함수는 데이터 분석을 수행하고, 결과를 Base64 인코딩된 이미지 데이터와 함께
    템플릿에 직접 전달할 수 있는 형식으로 반환합니다.
    """
    
    result_dict = {}
    
    try:
        # 1. 데이터 파일 불러오기
        base_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(base_dir, '..', 'csv', 'apartment_rent_data_price.csv')
        df = pd.read_csv(file_path, sep=';', encoding='utf-8')
        
        # 2-5. 데이터 전처리 및 금액대 분류
        df['구'] = df['시군구'].apply(lambda x: x.split()[1] if len(x.split()) > 1 else None)
        df.dropna(subset=['구'], inplace=True)
        df = df[['시군구', '전월세구분', '보증금(만원)', '월세금(만원)', '계약년월', '구']]
        df['보증금(만원)'] = pd.to_numeric(df['보증금(만원)'].str.replace(',', '', regex=True).fillna('0'))
        df['월세금(만원)'] = pd.to_numeric(df['월세금(만원)'].str.replace(',', '', regex=True).fillna('0'))
        df['환산보증금'] = df.apply(
            lambda row: row['보증금(만원)'] + row['월세금(만원)'] * 100 if row['전월세구분'] == '월세' else row['보증금(만원)'], axis=1
        )
        df = df[df['환산보증금'] <= 200000]
        bins = list(range(0, int(df['환산보증금'].max()) + 10000, 10000))
        labels = [f'{i // 10000}억~{(i + 10000) // 10000}억' for i in range(0, int(df['환산보증금'].max()) + 10000, 10000)[:-1]]
        df['금액대'] = pd.cut(df['환산보증금'], bins=bins, labels=labels, right=False)
        df['계약년월'] = pd.to_datetime(df['계약년월'].astype(str), format='%Y%m')

        plt.rcParams['font.family'] = 'Malgun Gothic'
        mpl.rcParams['axes.unicode_minus'] = False

        
        # 6. 구별, 금액대별 수요량 분석 및 시각화
        df_jeonse = df[df['전월세구분'] == '전세'].copy()
        df_wolse = df[df['전월세구분'] == '월세'].copy()
        demand_jeonse = df_jeonse.groupby(['구', '금액대']).size().unstack(fill_value=0)
        demand_wolse = df_wolse.groupby(['구', '금액대']).size().unstack(fill_value=0)
        
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
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        plt.close()
        result_dict['images'] = [f'data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode()}']
        result_dict['demand_jeonse_html'] = demand_jeonse.head().to_html(classes=['accuracy-table'])
        result_dict['demand_wolse_html'] = demand_wolse.head().to_html(classes=['accuracy-table'])
        
        # 7. 구별 수요 패턴 군집화
        demand_by_gu = df.groupby(['구', '금액대']).size().unstack(fill_value=0)
        scaler = StandardScaler()
        scaled_demand = scaler.fit_transform(demand_by_gu)
        optimal_k = 4
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init='auto')
        demand_by_gu['cluster'] = kmeans.fit_predict(scaled_demand)
        pca = PCA(n_components=2)
        reduced_data = pca.fit_transform(scaled_demand)
        
        plt.figure(figsize=(12, 10))
        sns.scatterplot(x=reduced_data[:, 0], y=reduced_data[:, 1], hue=demand_by_gu['cluster'], palette='viridis', s=200)
        plt.title('구별 수요량 패턴 군집화 결과 (K-Means + PCA)')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
    
        for i, gu in enumerate(demand_by_gu.index):
            plt.text(reduced_data[i, 0] + 0.05, reduced_data[i, 1], gu)
            print(gu)

        
        plt.tight_layout()
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        plt.close()
        result_dict['images'].append(f'data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode()}')
        result_dict['cluster_html'] = demand_by_gu[['cluster']].to_html(classes=['accuracy-table'])
    
        print(result_dict['cluster_html'])
        # 8. Prophet 모델을 사용한 시계열 예측
        time_series_data = df.groupby(['계약년월', '구']).size().reset_index(name='거래건수')
        accuracy_results = {}
        graph_urls = {} # 각 구별 그래프 URL을 저장할 딕셔너리
        gu_list = time_series_data['구'].unique()
        
        # plt.figure(figsize=(30, 20)) # 이 부분은 삭제
        # num_rows = int(np.ceil(len(gu_list) / 5)) # 이 부분은 삭제
        # num_cols = 5 # 이 부분은 삭제

        for i, gu in enumerate(gu_list):
            plt.figure(figsize=(10, 6)) # 각 구별로 새로운 Figure 생성
            gu_data = time_series_data[time_series_data['구'] == gu].rename(columns={'계약년월': 'ds', '거래건수': 'y'})
            if len(gu_data) > 3:
                train_data = gu_data[:-3]
                test_data = gu_data[-3:]
                m1 = Prophet(seasonality_mode='multiplicative', daily_seasonality=False)
                m1.fit(train_data)
                future_test = m1.make_future_dataframe(periods=3, freq='M')
                forecast_test = m1.predict(future_test)
                test_forecast = forecast_test['yhat'].tail(3)
                y_true = test_data['y'].values
                y_pred = test_forecast.values
                mae = mean_absolute_error(y_true, y_pred)
                rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100 if np.all(y_true != 0) else float('inf')
                accuracy_results[gu] = {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}
                m2 = Prophet(seasonality_mode='multiplicative', daily_seasonality=False)
                m2.fit(gu_data)
                future_final = m2.make_future_dataframe(periods=3, freq='M')
                forecast_final = m2.predict(future_final)
                forecast_df = forecast_final[['ds', 'yhat']].tail(3)
                bar_data = pd.concat([gu_data, forecast_df.rename(columns={'yhat': 'y'})], ignore_index=True)
                bar_data['ds_str'] = bar_data['ds'].dt.strftime('%Y-%m')
                bar_data['type'] = '실제'
                bar_data.loc[bar_data['ds'].isin(forecast_df['ds']), 'type'] = '예측'
                sns.barplot(x='ds_str', y='y', hue='type', dodge=False, data=bar_data)
                plt.title(f'{gu} 거래량 예측', fontsize=15)
                plt.xlabel('날짜', fontsize=12)
                plt.ylabel('거래 건수', fontsize=12)
                plt.legend(title='데이터 유형', loc='upper left', fontsize=10)
                plt.xticks(rotation=45, ha='right')
                
            else:
                plt.text(0.5, 0.5, '데이터 부족', ha='center', va='center', fontsize=20, color='red', transform=plt.gca().transAxes)
                plt.title(f'{gu} 거래량', fontsize=15)
                plt.xlabel('날짜', fontsize=12)
                plt.ylabel('거래 건수', fontsize=12)
                plt.xticks([])
                plt.yticks([])
            
            plt.tight_layout()
            # 각 그래프를 개별적으로 버퍼에 저장
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            plt.close()
            
            # Base64 인코딩하여 딕셔너리에 추가
            graph_urls[gu] = f'data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode()}'
            
        # result_dict['images'].append(f'data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode()}') # 이 부분은 삭제
        result_dict['accuracy'] = accuracy_results
        result_dict['graph_urls'] = graph_urls # 새로운 graph_urls 딕셔너리 추가
        
    except Exception as e:
        result_dict["ERROR"] = f"오류가 발생했습니다: {e}"
    
    return result_dict

def price_prediction(request):
    """
    이 뷰 함수는 run_analysis() 함수를 호출하여 모든 분석을 수행하고
    결과를 템플릿으로 전달합니다.
    """
    result = run_analysis()
    
    # 템플릿에 전달할 JSON 데이터 추가
    accuracy_json = json.dumps(result.get('accuracy', {}))
    graph_urls_json = json.dumps(result.get('graph_urls', {}))
    
    return render(request, 'bbs/price_prediction.html', {
        'result': result, 
        'accuracy_json': accuracy_json, 
        'graph_urls_json': graph_urls_json
    })