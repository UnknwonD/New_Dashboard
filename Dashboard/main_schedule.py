import pandas as pd
from sqlalchemy import create_engine, text
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import datetime
import os
from api import db_url, sender_email, sender_password, smtp_server, smtp_port
from collections import Counter
import schedule
import time
import ast
import re

def send_email(subject, body, recipients):
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = ", ".join(recipients)
    msg['Subject'] = subject

    msg.attach(MIMEText(body, 'plain'))

    # SMTP 서버에 연결 및 메일 발송
    for attempt in range(3):  # 최대 3회 재시도
        try:
            server = smtplib.SMTP(smtp_server, smtp_port, timeout=60)
            server.starttls()
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, recipients, msg.as_string())
            server.close()
            print("Email successfully sent to:", recipients)
            break  # 성공적으로 전송된 경우 루프 종료
        except Exception as e:
            print(f"Failed to send email on attempt {attempt + 1}. Error:", e)
            time.sleep(5)  # 재시도 전에 잠깐 대기
    else:
        print("All attempts to send email have failed.")
        
# 데이터베이스 연결 및 데이터 로드
def data_load(target_date):
    # 데이터베이스 엔진 생성
    engine = create_engine(db_url)
    
    # SQL 쿼리 생성
    sql = f'''
    SELECT * 
    FROM social_data 
    WHERE url IS NOT NULL 
    AND DATE(date) = '{target_date.strftime('%Y-%m-%d')}'
    '''
    sql = text(sql)
    
    # 데이터 로드 및 날짜 형식 변환
    with engine.connect() as conn:
        df = pd.read_sql(sql, conn)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    # content와 sentences의 '-' 값을 리스트로 변환 후 처리
    df['content'] = df['content'].replace('-', "['-']")
    df['sentences'] = df['sentences'].replace('-', "['-']")

    # content와 sentences를 리스트 타입으로 변환
    df['content'] = df['content'].apply(ast.literal_eval)
    df['sentences'] = df['sentences'].apply(ast.literal_eval)
    
    return df

# 뉴스 데이터 분석
def analyze_news_data(df:pd.DataFrame):
    # 모든 키워드 데이터 모으기
    all_keywords = []
    for keywords in df['keywords']:
        all_keywords.extend(keywords.split(", "))

    # 나라와 지역명 제거
    news = list(set(tuple(df['publisher'].to_list())))
    exclude_keywords = {"한국", "미국", "중국", "서울", "대전", "부산", "경기", "대구", "인천", "광주", "울산", "강원", "충북", "충남", "전북", "전남", "경북", "경남", "제주"}
    filtered_keywords = [kw for kw in all_keywords if kw not in exclude_keywords and kw not in news]


    # 키워드 빈도 계산 및 데이터프레임 생성
    keyword_count = Counter(filtered_keywords)
    keyword_count_df = pd.DataFrame(keyword_count.items(), columns=['Keyword', 'Count']).sort_values(by='Count', ascending=False).head(10)
    return keyword_count_df

# 메일 내용 생성
def create_email_content(df, keyword_count_df):
    content = f"데일리 뉴스 리포트: {datetime.datetime.now().strftime('%Y년 %m월 %d일')}\n"
    content += "\n📰 주요 뉴스:\n"
    categories = df['category'].unique()
    for category in categories:
        content += f"\n🌐 {category} 뉴스\n"
        category_news = df[df['category'] == category].tail(3)
        for _, row in category_news.iterrows():
            content += f"- {row['title']} ({row['publisher']})\n  [링크]({row['url']})\n"

    content += "\n🔥 실시간 인기 키워드 TOP 10:\n"
    
    for i, (index, row) in enumerate(keyword_count_df.iterrows()):
        content += f"\n{i + 1}. 📌 {row['Keyword']}\n"
        
        # 해당 키워드가 포함된 긍정 및 부정 뉴스 2개씩 추가
        keyword_df = df[df['title'].str.contains(row['Keyword'])]
        positive_news = keyword_df[keyword_df['sentiment'] == '긍정'].head(2)
        negative_news = keyword_df[keyword_df['sentiment'] == '부정'].head(2)
        
        content += "\n  ➕ 긍정 뉴스:\n"
        for _, news_row in positive_news.iterrows():
            content += f"    - {news_row['title']} ({news_row['publisher']})\n      [링크]({news_row['url']})\n"
        
        content += "\n  ➖ 부정 뉴스:\n"
        for _, news_row in negative_news.iterrows():
            content += f"    - {news_row['title']} ({news_row['publisher']})\n      [링크]({news_row['url']})\n"
        
    return content

# 메일 발송 작업 함수
def send_email_now():
    target_date = datetime.datetime.now() - datetime.timedelta(days=1)  # 어제 날짜 기준
    df = data_load(target_date)

    if not df.empty:
        keyword_count_df = analyze_news_data(df)
        email_content = create_email_content(df, keyword_count_df)

        recipients = ["daeho5000@ajou.ac.kr"]  # 수신자 리스트
        send_email("데일리 뉴스 리포트", email_content, recipients)
    else:
        print("선택한 날짜에 해당하는 데이터가 없습니다.")

# 매일 오전 9시에 이메일 발송하도록 스케줄 설정
schedule.every().day.at("09:00").do(send_email_now)

# 스케줄러 실행 및 테스트 모드 선택
if __name__ == "__main__":
    # 테스트 실행 시 즉시 이메일 발송
    send_email_now()
    
    while True:
        schedule.run_pending()
        time.sleep(60)
