import os
import io
import time
import json
import base64
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings("ignore")

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
        result_dict['demand_jeonse_html'] = demand_jeonse.to_html(classes=['accuracy-table'])
        result_dict['demand_wolse_html'] = demand_wolse.to_html(classes=['accuracy-table'])
        
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
         

        
        plt.tight_layout()
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        plt.close()
        result_dict['images'].append(f'data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode()}')
        result_dict['cluster_html'] = demand_by_gu[['cluster']].to_html(classes=['accuracy-table'])
    
        

        # 8. Prophet, 선형 회귀, 랜덤 포레스트 모델을 사용한 시계열 예측
        time_series_data = df.groupby(['계약년월', '구']).size().reset_index(name='거래건수')
        gu_list = time_series_data['구'].unique()

        # 모델별 결과 저장 딕셔너리
        prophet_results = {'accuracy': {}, 'graph_urls': {}}
        linear_regression_results = {'accuracy': {}, 'graph_urls': {}}
        random_forest_results = {'accuracy': {}, 'graph_urls': {}}
        
        for gu in gu_list:
            gu_data = time_series_data[time_series_data['구'] == gu].copy()
            gu_data['ds'] = gu_data['계약년월']
            gu_data['y'] = gu_data['거래건수']

            # 데이터가 충분한 경우에만 예측 수행
            if len(gu_data) > 3:
                # Prophet 모델 예측
                try:
                    m = Prophet(seasonality_mode='multiplicative', daily_seasonality=False)
                    m.fit(gu_data.iloc[:-3][['ds', 'y']])
                    # future = m.make_future_dataframe(periods=3, freq='M')
                    future_dates = pd.to_datetime(['2025-08-01', '2023-09-01', '2023-10-01'])
                    future = pd.DataFrame({'ds': future_dates})
                    forecast = m.predict(future)
                    y_true = gu_data['y'].iloc[-3:].values
                    y_pred = forecast['yhat'].iloc[-3:].values
                    
                    mae = mean_absolute_error(y_true, y_pred)
                    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100 if np.all(y_true != 0) else float('inf')
                    
                    prophet_results['accuracy'][gu] = {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}

                    plt.figure(figsize=(10, 6))
                    forecast_df = forecast[['ds', 'yhat']].tail(3)
                    bar_data = pd.concat([gu_data[['ds', 'y']], forecast_df.rename(columns={'yhat': 'y'})], ignore_index=True)
                    bar_data['ds_str'] = bar_data['ds'].dt.strftime('%Y-%m')
                    bar_data['type'] = ['실제'] * len(gu_data) + ['예측'] * 3
                    sns.barplot(x='ds_str', y='y', hue='type', dodge=False, data=bar_data)
                    plt.title(f'{gu} Prophet 모델 예측')
                    plt.xlabel('날짜')
                    plt.ylabel('거래 건수')
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    
                    buffer = io.BytesIO()
                    plt.savefig(buffer, format='png')
                    buffer.seek(0)
                    plt.close()
                    prophet_results['graph_urls'][gu] = f'data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode()}'
                except Exception as e:
                    prophet_results['accuracy'][gu] = {'MAE': 'N/A', 'RMSE': 'N/A', 'MAPE': 'N/A'}
                    plt.figure(figsize=(10, 6))
                    plt.text(0.5, 0.5, f'Prophet 모델 오류\n{e}', ha='center', va='center', fontsize=12, color='red')
                    plt.title(f'{gu} Prophet 모델 예측')
                    plt.tight_layout()
                    buffer = io.BytesIO()
                    plt.savefig(buffer, format='png')
                    buffer.seek(0)
                    plt.close()
                    prophet_results['graph_urls'][gu] = f'data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode()}'

                # 선형 회귀 모델 예측
                try:
                    gu_data['time_idx'] = (gu_data['ds'] - gu_data['ds'].min()).dt.days
                    X_train = gu_data[['time_idx']].iloc[:-3]
                    y_train = gu_data['y'].iloc[:-3]
                    X_test = gu_data[['time_idx']].iloc[-3:]
                    
                    model_lr = LinearRegression()
                    model_lr.fit(X_train, y_train)
                    y_pred = model_lr.predict(X_test)
                    
                    y_true = gu_data['y'].iloc[-3:].values
                    
                    mae = mean_absolute_error(y_true, y_pred)
                    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100 if np.all(y_true != 0) else float('inf')
                    
                    linear_regression_results['accuracy'][gu] = {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}
                    
                    plt.figure(figsize=(10, 6))
                    forecast_dates = pd.date_range(start=gu_data['ds'].iloc[-1], periods=4, freq='M')[1:]
                    bar_data = pd.DataFrame({
                        'ds_str': gu_data['ds'].dt.strftime('%Y-%m').tolist() + forecast_dates.strftime('%Y-%m').tolist(),
                        'y': gu_data['y'].tolist() + y_pred.tolist(),
                        'type': ['실제'] * len(gu_data) + ['예측'] * 3
                    })
                    sns.barplot(x='ds_str', y='y', hue='type', dodge=False, data=bar_data)
                    plt.title(f'{gu} 선형 회귀 모델 예측')
                    plt.xlabel('날짜')
                    plt.ylabel('거래 건수')
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    
                    buffer = io.BytesIO()
                    plt.savefig(buffer, format='png')
                    buffer.seek(0)
                    plt.close()
                    linear_regression_results['graph_urls'][gu] = f'data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode()}'
                except Exception as e:
                    linear_regression_results['accuracy'][gu] = {'MAE': 'N/A', 'RMSE': 'N/A', 'MAPE': 'N/A'}
                    plt.figure(figsize=(10, 6))
                    plt.text(0.5, 0.5, f'선형 회귀 모델 오류\n{e}', ha='center', va='center', fontsize=12, color='red')
                    plt.title(f'{gu} 선형 회귀 모델 예측')
                    plt.tight_layout()
                    buffer = io.BytesIO()
                    plt.savefig(buffer, format='png')
                    buffer.seek(0)
                    plt.close()
                    linear_regression_results['graph_urls'][gu] = f'data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode()}'

                # 랜덤 포레스트 회귀 모델 예측
                try:
                    gu_data['time_idx'] = (gu_data['ds'] - gu_data['ds'].min()).dt.days
                    X_train = gu_data[['time_idx']].iloc[:-3]
                    y_train = gu_data['y'].iloc[:-3]
                    X_test = gu_data[['time_idx']].iloc[-3:]
                    
                    model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
                    model_rf.fit(X_train, y_train)
                    y_pred = model_rf.predict(X_test)
                    
                    y_true = gu_data['y'].iloc[-3:].values
                    
                    mae = mean_absolute_error(y_true, y_pred)
                    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100 if np.all(y_true != 0) else float('inf')
                    
                    random_forest_results['accuracy'][gu] = {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}                    

                    plt.figure(figsize=(10, 6))
                    forecast_dates = pd.date_range(start=gu_data['ds'].iloc[-1], periods=4, freq='M')[1:]
                    bar_data = pd.DataFrame({
                        'ds_str': gu_data['ds'].dt.strftime('%Y-%m').tolist() + forecast_dates.strftime('%Y-%m').tolist(),
                        'y': gu_data['y'].tolist() + y_pred.tolist(),
                        'type': ['실제'] * len(gu_data) + ['예측'] * 3
                    })
                    sns.barplot(x='ds_str', y='y', hue='type', dodge=False, data=bar_data)
                    plt.title(f'{gu} 랜덤 포레스트 회귀 모델 예측')
                    plt.xlabel('날짜')
                    plt.ylabel('거래 건수')
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    
                    buffer = io.BytesIO()
                    plt.savefig(buffer, format='png')
                    buffer.seek(0)
                    plt.close()
                    random_forest_results['graph_urls'][gu] = f'data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode()}'
                except Exception as e:
                    random_forest_results['accuracy'][gu] = {'MAE': 'N/A', 'RMSE': 'N/A', 'MAPE': 'N/A'}
                    plt.figure(figsize=(10, 6))
                    plt.text(0.5, 0.5, f'랜덤 포레스트 회귀 모델 오류\n{e}', ha='center', va='center', fontsize=12, color='red')
                    plt.title(f'{gu} 랜덤 포레스트 회귀 모델 예측')
                    plt.tight_layout()
                    buffer = io.BytesIO()
                    plt.savefig(buffer, format='png')
                    buffer.seek(0)
                    plt.close()
                    random_forest_results['graph_urls'][gu] = f'data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode()}'
            else:
                prophet_results['accuracy'][gu] = linear_regression_results['accuracy'][gu] = random_forest_results['accuracy'][gu] = {'MAE': '데이터 부족', 'RMSE': '데이터 부족', 'MAPE': '데이터 부족'}
                prophet_results['graph_urls'][gu] = linear_regression_results['graph_urls'][gu] = random_forest_results['graph_urls'][gu] = None
        
        result_dict['prophet'] = prophet_results
        result_dict['linear_regression'] = linear_regression_results
        result_dict['random_forest'] = random_forest_results
        
    except Exception as e:
        result_dict["ERROR"] = f"오류가 발생했습니다: {e}"
    
    return result_dict