# -*- coding: utf-8 -*-

import time
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
import pandas as pd
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from sqlalchemy import create_engine, Table, MetaData, update
from webdriver_manager.chrome import ChromeDriverManager
import os
import re
import sys
from tqdm import tqdm
from api import db_url
from summarizer import Summarizer
from kiwipiepy import Kiwi

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

# 크롬 옵션 설정
chrome_options = Options()
# chrome_options.add_argument("--headless")
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")
chrome_options.add_argument("--disable-extensions")
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("log-level=3")

def data_load():
    engine = create_engine(db_url)
    sql = 'SELECT * FROM social_data WHERE url is not null and summary is null'

    df = pd.read_sql(sql, engine)

    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    return df

def extract_content(url, driver):
    try:
        driver.get(url)
        time.sleep(2)
        content = driver.find_element(By.ID, 'dic_area').text
        return content
    except Exception as e:
        print('Content를 불러올 수 없습니다. : ', e)
        return None

if __name__ == '__main__':
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
    kiwi = Kiwi(num_workers=0)
    df = data_load()

    for i, row in tqdm(df.iterrows(), total=df.shape[0]):
        engine = create_engine(db_url)
        target_url = row['url']
        content = extract_content(target_url, driver)

        if content:
            # 문장을 분할하고 명사만 추출
            content = replace_hanja(content)
            sentence = kiwi.split_into_sents(content)
            total_sentence = []
            tokens = []
            for sen in sentence:
                target_sen = sen.text.replace('\n', ' ')
                total_sentence.append(target_sen)
                tmp_tokens = [
                    word.form + ('다' if word.tag.startswith('VV') else '')
                    for word in kiwi.tokenize(target_sen)
                    if word.tag.startswith('NN') or word.tag.startswith('VV')
                ]

                tokens.append(tmp_tokens)
            
            # 요약 생성
            try:
                summary = extractive_summarize_korean_text(content, compression_ratio=0.4)
            except Exception as e:
                print('Summary를 만들 수 없습니다. : ',  e)
                summary = '-'
        else:
            total_sentence = '-'
            tokens = '-'
            summary = '-'

#         print(f'''
# total_sentence = {total_sentence}
# tokens = {tokens}
# summary = {summary}
# ''')
        
        # 데이터베이스 업데이트
        with engine.connect() as conn:
            metadata = MetaData()
            social_data_table = Table('social_data', metadata, autoload_with=engine)

            update_stmt = (
                update(social_data_table)
                .where(social_data_table.c.seq == row['seq'])
                .values(
                    sentences=str(tokens),
                    content=str(total_sentence),
                    summary=str(summary)
                )
            )

            try:
                conn.execute(update_stmt)
                conn.commit()
            except Exception as e:
                print(f"Failed to update row {row['seq']}: {e}")

    driver.quit()
