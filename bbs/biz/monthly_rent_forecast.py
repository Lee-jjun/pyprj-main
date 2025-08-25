"""
아파트 전,월세에 대한 예측한다면 어떤 방법이 가장 적합하며 그리고 예측을 위해한 방법과 데이터들은 실거래데이터로 충분한지 궁금합니다.


1. 아파트 전·월세 예측에 적합한 방법
전·월세 예측은 부동산 가격 예측의 한 유형이고, 본질적으로 시계열적 성격 + 회귀 문제를 동시에 가집니다.
따라서 접근 방식은 두 가지 축으로 나뉩니다:

(1) 전통적 머신러닝 기반 회귀 모델
랜덤 포레스트 (Random Forest)
그래디언트 부스팅 (XGBoost, LightGBM, CatBoost 등)
→ 구조적 데이터(아파트 단지 정보, 면적, 층수, 위치, 입주 연도 등)에 강합니다.
→ 특히 범주형(지역, 단지명 등) + 수치형(면적, 층수 등) 혼합 데이터에 적합합니다.

(2) 딥러닝 기반 모델
다층 퍼셉트론 (MLP) : 단순 회귀용
시계열 모델 (LSTM, GRU, Transformer 기반) : 거래 시점별 변동 반영 가능
하이브리드 모델 : 예) "시계열(가격 흐름) + 부동산 속성(feature)"을 같이 학습하는 구조

실무적으로는 XGBoost / LightGBM → 딥러닝(시계열 보강) 순서로 접근하는 게 가장 합리적입니다.

2. 예측을 위해 필요한 데이터
실거래 데이터(국토교통부 공개 전월세 자료)는 좋은 출발점이지만 그 자체로는 한계가 있습니다.

(1) 실거래 데이터만으로 가능한 부분
전·월세 보증금, 월세 수준
아파트 면적, 층수, 준공 연도, 위치
시간(계약년월) → 시세 트렌드 반영
→ 이를 통해 "비슷한 단지, 면적, 시점"의 전·월세 가격은 꽤 잘 예측됩니다.

(2) 추가되면 더 정확해지는 데이터
지역 경제 지표 : 금리, 물가상승률, 지역 소득 수준
주변 전세·매매 시세 : 같은 단지, 주변 단지의 매매가와 전세가 비율
주거환경 정보 : 지하철, 학교, 학군, 상권, 교통 접근성
공급/수요 요인 : 전세 물량, 신규 분양·입주 물량
정책 변수 : 전세자금대출 규제, 임대차보호법 등
결론: 실거래 데이터만으로도 기본적인 예측은 가능하지만,
외부 요인을 추가하면 훨씬 더 정확하고 안정적인 모델을 만들 수 있습니다.

3. 연구/실무적 접근 전략
EDA (탐색적 데이터 분석)
지역별, 단지별, 기간별 전세/월세 추이 확인
가격 분포의 이상치 제거
피처 엔지니어링
전용면적 → 로그 변환
층수 → 저층/중층/고층 구간화
계약일자 → "분기/월 단위"로 변환

모델링 단계
1차: XGBoost/LightGBM으로 베이스라인 모델
2차: LSTM/Transformer로 시간 축 반영
3차: 앙상블 → 두 결과를 합쳐서 최종 예측

요약:
가장 적합한 방법 → 구조 데이터는 XGBoost/LightGBM, 시계열 반영은 딥러닝(LSTM/Transformer)
실거래 데이터만으로도 가능하지만, 외부 요인(금리, 매매가, 입주 물량 등)을 추가하면 훨씬 더 정확해짐.

"""


"""

1. 데이터 준비 (추가 외부 데이터 수집)
전·월세 예측에 도움 되는 주요 외부 데이터는 크게 경제 변수 / 부동산 변수 / 지역 환경 변수로 나눌 수 있습니다.

(1) 경제 변수
    금리: 한국은행 기준금리, 주택담보대출 금리
    물가/소득: 소비자물가지수(CPI), 가계소득 수준
    출처: 한국은행 ECOS, 통계청

(2) 부동산 변수
    매매 시세: KB부동산, 국토부 실거래 매매가
    전세가율: 매매가 대비 전세가 비율
    입주 물량: 국토부 주택 인허가/착공/입주 통계
    출처: 국토교통부, KB시세, 부동산114 등

(3) 지역 환경 변수
    교통 인프라: 지하철역, 버스노선, 도로망
    교육 인프라: 초·중·고 위치, 학군 수준
    생활 인프라: 대형마트, 병원, 공원, 상권 밀집도
    출처: 공공데이터포털, 카카오/네이버 지도 API

2. 데이터 전처리 및 매핑 방법
    문제는 “외부 데이터를 어떻게 전·월세 실거래 데이터와 연결(매핑)하느냐” 입니다.

(1) 시간 매핑
    금리, 물가, 매매가 지수 → 계약년월 단위로 맞춤
    예: 2023년 7월 계약이면, 2023-07의 기준금리/매매가 지수 값 사용

(2) 공간(지역) 매핑
    실거래 데이터의 법정동 코드, 행정동 코드, 시군구 기준으로 외부 데이터 결합
    예: “서울시 강남구 역삼동 아파트 전세 거래”
    → 강남구 평균 매매가, 전세가율, 교통 접근성 변수 매칭

(3) 특성 변환 (Feature Engineering)
    금리, 물가 등은 전월 대비 변화율(Δ) 추가
    매매가 대비 전세가율 = 전세가 / 매매가 파생변수 생성
    인프라 변수는 거리 기반 수치화 (예: 최근접 지하철역까지 거리, 반경 500m 내 학교 수)

3. 모델 접목 방법
(1) 단순 피처 추가
외부 데이터를 새로운 열(Feature)로 추가

예:
[전용면적, 층수, 준공연도, 거래월, 금리, 매매가지수, 전세가율, 교통접근성, 학군지수 ...]
(2) 시계열 + 구조적 데이터 결합
    멀티 인풋 모델 (딥러닝에서 자주 씀)
    Input A: 아파트 특성 (면적, 층수, 준공연도 등)
    Input B: 시계열 변수 (금리 추이, 매매가 지수 변화 등)
    두 가지 입력을 Dense Layer에서 합쳐서 출력 → 전·월세 예측

(3) 지역별 클러스터링 후 반영
    아파트를 지역·단지 단위로 묶고, 지역 특성 변수(금리 민감도, 전세가율) 반영
    예: “강남권 단지 = 금리 변화에 민감 / 외곽 단지 = 공급 물량에 민감”

    4. 정리 (연구/실무 팁)
        1. 실거래 데이터 = 기본 뼈대
        (거래금액, 계약년월, 면적, 층수, 위치)
        2. 외부 데이터 수집 → 전처리 → 매핑
         시간 단위(월별)
         지역 단위(법정동, 시군구)

        3. Feature Engineering
         원본 값 + 파생 변수 (증감률, 비율, 거리 기반 변수)
        
        4.모델 학습
         머신러닝(XGBoost) : 간단한 피처 추가
         딥러닝(LSTM/멀티인풋) : 시계열과 구조 데이터를 함께 반영

         요약:
            외부 데이터를 시간·지역 기준으로 정규화 후 실거래 데이터와 매핑해야 합니다.
            접목은 단순히 Feature로 추가하거나, 시계열 + 구조 데이터 멀티인풋 모델을 사용하는 방식이 가장 효과적입니다.
"""

import pandas as pd

# -----------------------------
# 1. 실거래 데이터 로드
# -----------------------------
# 예시: rent_data.csv
# 컬럼: [contract_date, region_code, area, floor, deposit, monthly_rent]
rent_df = pd.read_csv("rent_data.csv", parse_dates=["contract_date"])

# 계약년월 생성
rent_df["year_month"] = rent_df["contract_date"].dt.to_period("M").astype(str)

# -----------------------------
# 2. 외부 데이터 로드
# -----------------------------
# (1) 금리 데이터: [year_month, interest_rate]
interest_df = pd.read_csv("interest_rate.csv")
interest_df["year_month"] = pd.to_datetime(interest_df["year_month"]).dt.to_period("M").astype(str)

# (2) 주택 매매지수 데이터: [year_month, region_code, house_price_index]
hpi_df = pd.read_csv("house_price_index.csv")
hpi_df["year_month"] = pd.to_datetime(hpi_df["year_month"]).dt.to_period("M").astype(str)

# -----------------------------
# 3. 데이터 전처리 & 파생 변수 생성
# -----------------------------
# 금리 전월 대비 변화율
interest_df["interest_rate_change"] = interest_df["interest_rate"].pct_change().fillna(0)

# 매매지수 전월 대비 변화율
hpi_df["hpi_change"] = hpi_df.groupby("region_code")["house_price_index"].pct_change().fillna(0)

# -----------------------------
# 4. 외부 데이터 병합
# -----------------------------
# (1) 실거래 + 금리
merged_df = pd.merge(rent_df, interest_df, on="year_month", how="left")

# (2) 실거래 + 매매지수 (지역 매핑)
merged_df = pd.merge(merged_df, hpi_df, on=["year_month", "region_code"], how="left")

# -----------------------------
# 5. 추가 Feature Engineering
# -----------------------------
# 전세가율: 보증금 / 매매지수
merged_df["jeonse_ratio"] = merged_df["deposit"] / (merged_df["house_price_index"] * 10000)

# 면적 로그 변환 (비선형 효과 완화)
merged_df["log_area"] = np.log1p(merged_df["area"])

# -----------------------------
# 6. 모델 학습에 사용할 최종 데이터셋
# -----------------------------
final_features = [
    "year_month", "region_code", "log_area", "floor",
    "interest_rate", "interest_rate_change",
    "house_price_index", "hpi_change", "jeonse_ratio",
]
target = ["monthly_rent"]  # 혹은 deposit 예측도 가능

X = merged_df[final_features]
y = merged_df[target]

print("최종 데이터셋 크기:", X.shape)
print(X.head())






##############################################################

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

# -----------------------------
# 1. 최종 데이터셋 준비
# -----------------------------
# 앞에서 통합한 merged_df 사용한다고 가정
final_features = [
    "log_area", "floor",
    "interest_rate", "interest_rate_change",
    "house_price_index", "hpi_change", "jeonse_ratio",
]
target = "monthly_rent"  # 월세 예측

X = merged_df[final_features]
y = merged_df[target]

# -----------------------------
# 2. Train / Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# 3. XGBoost 모델 정의
# -----------------------------
model = xgb.XGBRegressor(
    n_estimators=1000,       # 트리 개수
    learning_rate=0.05,      # 학습률
    max_depth=6,             # 트리 깊이
    subsample=0.8,           # 데이터 샘플 비율
    colsample_bytree=0.8,    # 피처 샘플 비율
    reg_lambda=1.0,          # L2 정규화
    random_state=42,
    tree_method="hist"       # 속도 최적화 (GPU 있으면 "gpu_hist")
)

# -----------------------------
# 4. 모델 학습
# -----------------------------
model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    eval_metric="rmse",
    early_stopping_rounds=50,
    verbose=50
)

# -----------------------------
# 5. 예측 및 평가
# -----------------------------
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print(f"MAE  (평균절대오차): {mae:.2f}")
print(f"RMSE (평균제곱근오차): {rmse:.2f}")
print(f"R²   (설명력): {r2:.3f}")

# -----------------------------
# 6. 중요 변수 확인
# -----------------------------
import matplotlib.pyplot as plt
xgb.plot_importance(model, importance_type="gain", height=0.5)
plt.show()




################################################################################
#한국은행 ECOS API활용예시
#(회원가입 후 API 키 발급 필요)
import requests
import pandas as pd

API_KEY = "IW04OSWSFICMJEBV1TLD"
#https://ecos.bok.or.kr/api/#/ <---- 인증키 신청
# 한국은행 기준금리 (통계코드: 722Y001, 항목: 0101000 = 기준금리)
url = f"http://ecos.bok.or.kr/api/StatisticSearch/{API_KEY}/json/kr/1/1000/722Y001/M/200001/202512/0101000"

response = requests.get(url)
data = response.json()

# 결과를 DataFrame으로 변환
rows = data['StatisticSearch']['row']
df = pd.DataFrame(rows)[['TIME', 'DATA_VALUE']]
df.columns = ["year_month", "interest_rate"]

# year_month 형식 변환
df["year_month"] = pd.to_datetime(df["year_month"], format="%Y%m")
print(df.head())