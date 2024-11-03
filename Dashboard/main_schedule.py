import pandas as pd
from sqlalchemy import create_engine, text
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
import datetime
import os
from api import db_url, sender_email, sender_password, smtp_server, smtp_port
from kiwipiepy import Kiwi
from collections import Counter
import schedule
import time

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
    return df

# 뉴스 데이터 분석
def analyze_news_data(df):
    # 형태소 분석기
    kiwi = Kiwi()
    all_tokens = []
    for sublist in df['sentences']:
        for sentence in sublist:
            analyzed = kiwi.analyze(sentence)
            if analyzed:
                morphs = analyzed[0][0]
                for token in morphs:
                    if token.tag[0] == 'N' and len(token.form) > 1:  # NNP 태그만 사용하여 의미있는 단어만 추출
                        all_tokens.append(token.form)

    # 단어 빈도 계산 및 데이터프레임 생성
    word_count = Counter(all_tokens)
    word_count_df = pd.DataFrame(word_count.items(), columns=['Word', 'Count']).sort_values(by='Count', ascending=False).head(10)
    return word_count_df

# 메일 내용 생성
def create_email_content(df, word_count_df):
    content = f"데일리 뉴스 리포트: {datetime.datetime.now().strftime('%Y년 %m월 %d일')}\n"
    content += "\n📰 주요 뉴스:\n"
    categories = df['category'].unique()
    for category in categories:
        content += f"\n🌐 {category} 뉴스\n"
        category_news = df[df['category'] == category].tail(3)
        for _, row in category_news.iterrows():
            content += f"- {row['title']} ({row['publisher']})\n  [링크]({row['url']})\n"

    content += "\n🔥 실시간 인기 단어 TOP 10:\n"
    for i, (index, row) in enumerate(word_count_df.iterrows()):
        content += f"{i + 1}. {row['Word']} - {row['Count']}회\n"

    return content

# 메일 발송 작업 스케줄링 함수
def schedule_email():
    target_date = datetime.datetime.now() - datetime.timedelta(days=1)  # 어제 날짜 기준
    df = data_load(target_date)

    if not df.empty:
        word_count_df = analyze_news_data(df)
        email_content = create_email_content(df, word_count_df)

        recipients = ["daeho5000@ajou.ac.kr"]  # 수신자 리스트
        send_email("데일리 뉴스 리포트", email_content, recipients)
    else:
        print("선택한 날짜에 해당하는 데이터가 없습니다.")

# 테스트 메일 발송 함수
def test_email():
    target_date = datetime.datetime.now() - datetime.timedelta(days=1)  # 어제 날짜 기준
    df = data_load(target_date)

    if not df.empty:
        word_count_df = analyze_news_data(df)
        email_content = create_email_content(df, word_count_df)

        recipients = ["daeho5000@ajou.ac.kr"]  # 테스트 수신자 리스트
        send_email("테스트: 데일리 뉴스 리포트", email_content, recipients)
    else:
        print("선택한 날짜에 해당하는 데이터가 없습니다.")

# 매일 오전 9시에 이메일 발송하도록 스케줄 설정
schedule.every().day.at("09:00").do(schedule_email)

# 스케줄러 실행 및 테스트 모드 선택
if __name__ == "__main__":
    mode = input("실행 모드를 선택하세요 (1: 테스트 메일 발송, 2: 실제 스케줄 실행): ")
    if mode == '1':
        test_email()
    elif mode == '2':
        while True:
            schedule.run_pending()
            time.sleep(60)
