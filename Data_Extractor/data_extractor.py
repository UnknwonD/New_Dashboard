import time
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
import pandas as pd

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

df = pd.read_csv('news_data.csv', encoding='utf-8-sig')

cnt = 0

while True:
    news_data = []
    start_time = datetime.now()

    for category in [100, 101, 102, 103, 104]:
        for sub in CATEGORY[category]:
            base_url = 'https://news.naver.com/breakingnews/section/' + str(category) + '/' + sub
            print(base_url)

            driver = webdriver.Chrome()
            driver.get(base_url)
            time.sleep(3)

            for _ in range(30):
                try:
                    driver.find_element(By.CSS_SELECTOR, 'a.section_more_inner').click()
                except:
                    break
                finally:
                    time.sleep(1)

            html = driver.page_source
            soup = BeautifulSoup(html, 'html.parser')

            news_body = soup.select('div.section_article')

            for news in news_body:
                contents = news.select('li.sa_item')
                for content in contents:
                    title = content.select_one('a.sa_text_title > strong').text if content.select_one('a.sa_text_title > strong') else "No title"
                    detail = content.select_one('div.sa_text_lede').text if content.select_one('div.sa_text_lede') else "No details"
                    publisher = content.select_one('div.sa_text_press').text if content.select_one('div.sa_text_press') else "No publisher"
                    date = content.select_one('div.sa_text_datetime > b').text if content.select_one('div.sa_text_datetime > b') else "No date"

                    row = {
                        'category': cat_dict[category],
                        'sub-category': sub_dict[sub],
                        'title': title,
                        'content': detail,
                        'publisher': publisher,
                        'date': str_to_date(date)
                    }
                    
                    news_data.append(row)

            driver.quit()
            
    ### 여기서 news_data를 df에 넣는 코드로 수정
    new_data_df = pd.DataFrame(news_data)
    df = pd.concat([df, new_data_df], ignore_index=True)
    
    df.drop_duplicates(subset='title', keep='first')
    cnt += 1

    try:
        df.to_csv('news_data.csv', index=False, encoding='utf-8-sig')
    except:
        ...

    print(f'''
[{cnt}회 수집 결과]
시작시간 : {start_time}
종료시간 : {datetime.now()}
data 수집된 개수 : {len(df)}
    ''')