import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from prophet import Prophet

# 주식 코드 설정 (예: 테슬라의 경우 TSLA)
stock_symbol = "TSLA"

# 주식 데이터 불러오기 (최근 6개월, 1일 단위)
stock = yf.Ticker(stock_symbol)
data = stock.history(period="6mo", interval="1d")  # 1일 단위로 최근 6개월 데이터 가져오기

# 데이터가 비어 있을 경우 예외 처리
if data.empty:
    raise ValueError("주식 데이터를 불러오지 못했습니다. 데이터가 비어 있습니다.")

# Prophet을 사용하기 위해 데이터프레임 형식 변환 (타임존 제거)
prophet_data = data.reset_index()[['Date', 'Close']]
prophet_data['Date'] = prophet_data['Date'].dt.tz_localize(None)  # 타임존 제거
prophet_data.rename(columns={'Date': 'ds', 'Close': 'y'}, inplace=True)

# Prophet 모델 설정 및 하이퍼파라미터 조정
model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
    changepoint_prior_scale=0.05  # 변동점 민감도 조정
)
model.add_seasonality(name='monthly', period=30.5, fourier_order=5)  # 월별 계절성 추가
model.fit(prophet_data)

# 예측을 위한 데이터프레임 생성 (10일 예측)
future = model.make_future_dataframe(periods=10)
forecast = model.predict(future)

# 음수 예측 값을 0으로 변환
forecast['yhat'] = forecast['yhat'].apply(lambda x: max(x, 0))

# 예측 결과 시각화 (한글 폰트 설정 추가)
plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows 환경에서 한글 폰트 설정
plt.figure(figsize=(10, 6))
plt.plot(prophet_data['ds'], prophet_data['y'], label='실제 가격', color='blue')
plt.plot(forecast['ds'], forecast['yhat'], label='예측 가격', color='red')
plt.xlabel('날짜')
plt.ylabel('가격')
plt.legend()
plt.show()

# 예측 결과 출력 (예측한 10일만)
future_predictions = forecast[['ds', 'yhat']].tail(10)
for _, row in future_predictions.iterrows():
    print(f"날짜: {row['ds']}, 예측 가격: {row['yhat']}")
