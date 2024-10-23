import time
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
import pandas as pd
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from sqlalchemy import create_engine, Table, MetaData, select
from webdriver_manager.chrome import ChromeDriverManager
from summarizer import Summarizer
from kiwipiepy import Kiwi
from tqdm import tqdm
# from api import db_url
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../DB_Update')))
from DB_UPDATE_FIRST import update_db

# 크롬 옵션 설정
chrome_options = Options()
chrome_options.add_argument("--headless")
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")
chrome_options.add_argument("--disable-extensions")
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("log-level=3")

CATEGORY = {
    100: ['264', '265', '266', '267', '268', '269'],
    101: ['263', '310', '262', '260', '771', '261', '258', '259'],
    102: ['249', '250', '251', '254', '252', '59b', '255', '256', '276', '257'],
    103: ['241', '239', '240', '237', '238', '376', '242', '243', '244', '248', '245'],
    104: ['731', '226', '227', '230', '732', '283', '229', '228']
}

cat_dict = {
    100: '정치',
    101: '경제',
    102: '사회',
    103: '생활/문화',
    104: 'IT/과학'
}

sub_dict = {
    '264': '대통령실',
    '265': '국회/정당',
    '268': '북한',
    '266': '행정',
    '267': '국방/외교',
    '269': '정치일반',
    '259': '금융',
    '258': '증권',
    '261': '산업/재계',
    '771': '중기/벤처',
    '260': '부동산',
    '262': '글로벌 경제',
    '310': '생활경제',
    '263': '경제 일반',
    '249': '사건사고',
    '250': '교육',
    '251': '노동',
    '254': '언론',
    '252': '환경',
    '59b': '인권/복지',
    '255': '식품/의료',
    '256': '지역',
    '276': '인물',
    '257': '사회 일반',
    '241': '건강정보',
    '239': '자동차/시승기',
    '240': '도로/교통',
    '237': '여행/레저',
    '238': '음식/맛집',
    '376': '패션/뷰티',
    '242': '공연/전시',
    '243': '책',
    '244': '종교',
    '248': '날씨',
    '245': '생활문화 일반',
    '731': '모바일',
    '226': '인터넷/SNS',
    '227': '통신/뉴미디어',
    '230': 'IT 일반',
    '732': '보안/해킹',
    '283': '컴퓨터',
    '229': '게임/리뷰',
    '228': '과학 일반'
}

replace_dict = {
    '尹': '윤석열',
    '文': '문재인',
    '朴': '박근혜',
    '李': '이명박',
    '金': '김대중',
    '盧': '노무현',
    '全': '전두환',
    '崔': '최규하',
    '朴正熙': '박정희',
    '日': '일본',
    '美': '미국',
    '中': '중국',
    '韓': '한국',
    '北': '북한',
    '露': '러시아',
    '英': '영국',
    '獨': '독일',
    '仏': '프랑스',
    '豪': '호주',
    '加': '캐나다',
    '印': '인도',
    '伊': '이탈리아',
    '西': '스페인',
    '瑞': '스위스',
    '希': '그리스',
    '越': '베트남',
    '泰': '태국',
    '菲': '필리핀',
    '墨': '멕시코',
    '智': '칠레',
    '阿': '아르헨티나',
    '埃': '이집트',
    '土': '터키',
    '蘇': '소련',
    '芬': '핀란드',
    '葡': '포르투갈',
    '蘭': '네덜란드',
    '洪': '헝가리'
}

def replace_hanja(text):
    for hanja, korean in replace_dict.items():
        text = text.replace(hanja, korean)
    return text

def split_text(text, max_length=512):
    sentences = text.split(". ")
    current_length = 0
    current_chunk = []
    chunks = []

    for sentence in sentences:
        if current_length + len(sentence.split()) <= max_length:
            current_chunk.append(sentence)
            current_length += len(sentence.split())
        else:
            chunks.append(". ".join(current_chunk) + ".")
            current_chunk = [sentence]
            current_length = len(sentence.split())

    if current_chunk:
        chunks.append(". ".join(current_chunk) + ".")
    
    return chunks

def extractive_summarize_korean_text(text, compression_ratio=0.3):
    # 긴 텍스트 분할
    chunks = split_text(text)
    model = Summarizer()

    # 각 청크에 대해 요약 수행 후 합치기
    summarized_chunks = [model(chunk, ratio=compression_ratio) for chunk in chunks]
    summary = " ".join(summarized_chunks)
    
    return summary



def str_to_date(phrase):
    result_time = datetime.now()

    if '분전' in phrase:
        minutes = int(phrase.replace('분전', '').strip())
        result_time = result_time - timedelta(minutes=minutes)
    elif '시간전' in phrase:
        hours = int(phrase.replace('시간전', '').strip())
        result_time = result_time - timedelta(hours=hours)
    elif '일전' in phrase:
        days = int(phrase.replace('일전', '').strip())
        result_time = result_time - timedelta(days=days)
    
    return result_time.strftime('%Y-%m-%d %H:%M')

# 기존 데이터 불러오기
df = pd.read_csv('news_data.csv', encoding='utf-8-sig')
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
cnt = 0
kiwi = Kiwi(num_workers=5)

total_data = len(df)
while True:
    news_data = []
    start_time = datetime.now()

    for category in [100, 101, 102, 103, 104]:
        for sub in CATEGORY[category]:
            base_url = 'https://news.naver.com/breakingnews/section/' + str(category) + '/' + sub
            print(base_url)

            driver.get(base_url)
            time.sleep(3)

            for _ in range(100):
                try:
                    driver.find_element(By.CSS_SELECTOR, 'a.section_more_inner').click()
                except:
                    break
                finally:
                    time.sleep(1)

            html = driver.page_source
            soup = BeautifulSoup(html, 'html.parser')

            news_body = soup.select('div.section_article')

            for news in tqdm(news_body):
                contents = news.select('li.sa_item')
                for content in contents:
                    title = content.select_one('a.sa_text_title > strong').text if content.select_one('a.sa_text_title > strong') else "-"
                    title = replace_hanja(title)

                    url = content.select_one('a.sa_text_title')['href'] if content.select_one('a.sa_text_title') else "-"
                    # detail = content.select_one('div.sa_text_lede').text if content.select_one('div.sa_text_lede') else "-"
                    publisher = content.select_one('div.sa_text_press').text if content.select_one('div.sa_text_press') else "-"
                    date = content.select_one('div.sa_text_datetime > b').text if content.select_one('div.sa_text_datetime > b') else "-"
                    
                    row = {
                        'category': cat_dict[category],
                        'sub_category': sub_dict[sub],
                        'title': title,
                        'content': '-',
                        'publisher': publisher,
                        'date': str_to_date(date),
                        'sentences': '-',
                        'url' : url,
                        'summary' : '-'
                    }
                    
                    news_data.append(row)

    # 새로운 데이터를 DataFrame으로 생성
    new_data_df = pd.DataFrame(news_data)

    # 기존 데이터와 중복 제거
    df = pd.concat([df, new_data_df], ignore_index=True)
    df.drop_duplicates(subset='title', keep='first', inplace=True)
    
    new_data = len(df) - total_data
    total_data = len(df)
    # CSV 파일로 데이터 저장
    try:
        df.to_csv('news_data.csv', index=False, encoding='utf-8-sig')
    except:
        print('데이터 저장 실패')
    try:
        update_db(df)
        print('DB 업데이트 성공')
    except Exception as e:
        print('DB 업데이트 실패 : ', e)

    print(f'''
[{cnt}회 수집 결과]
시작시간 : {start_time}
종료시간 : {datetime.now()}
전체 데이터 수 : {total_data}
data 수집된 개수 : {new_data}
    ''')
    cnt += 1

    print('3시간 뒤에 다시 수집합니다.')
    time.sleep(18000)