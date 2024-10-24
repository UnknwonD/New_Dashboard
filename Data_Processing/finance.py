import yfinance as yf
import time

# 주식 코드 설정 (예: 애플의 경우 AAPL)
stock_symbol = "TSLA"

# 주식 데이터를 지속적으로 업데이트하며 출력
while True:
    # 주식 데이터 불러오기
    stock = yf.Ticker(stock_symbol)
    data = stock.history(period="1d", interval="1m")  # 1분 단위로 데이터 가져오기
    
    # 가장 최근 데이터 가져오기
    latest_data = data.iloc[-1]
    print(f"시간: {latest_data.name}, 가격: {latest_data['Close']}")

    # 60초 대기 후 업데이트
    time.sleep(60)
