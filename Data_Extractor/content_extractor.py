# -*- coding: utf-8 -*-

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
import os
import sys
from api import db_url
from summarizer import Summarizer

def extractive_summarize_korean_text(text, compression_ratio=0.5):
    # BERT 기반 추출 요약 모델 생성
    model = Summarizer()
    summary = model(text, ratio=compression_ratio)
    return summary


# 크롬 옵션 설정
chrome_options = Options()
chrome_options.add_argument("--headless")
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")
chrome_options.add_argument("--disable-extensions")
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("log-level=3")

def tmp():
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)

    engine = create_engine(db_url)
    sql = 'SELECT * FROM social_data WHERE url is not null'

    df = pd.read_sql(sql, engine)

    df['date'] = pd.to_datetime(df['date'], errors='coerce')

def extract_content(url):
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)

    driver.get(url)
    time.sleep(3)

    article = driver.find_element(By.ID, 'dic_area').text

    driver.quit()

    summary = extractive_summarize_korean_text(article, compression_ratio=0.3)

    print(summary)

if __name__ == '__main__':
    raw_text = extract_content('https://n.news.naver.com/mnews/article/029/0002909582')




    
        

