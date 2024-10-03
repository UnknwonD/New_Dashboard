import time
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By


CATEGORY = {
    100 : ['264', '265', '266', '267', '268', '269'],
    101 : ['263', '310', '262', '260', '771', '261', '258', '259'],
    102 : ['249', '250', '251', '254', '252', '59b', '255', '256', '276', '257'],
    103 : ['241', '239', '240', '237', '238', '376', '242', '243', '244', '248', '245'],
    104 : ['731', '226', '227', '230', '732', '283', '229', '228']
}

def data_extract(category: int):
    for sub in CATEGORY[category]:
        base_url = 'https://news.naver.com/breakingnews/section/' + str(category) + '/' + sub
        print(base_url)

        driver = webdriver.Chrome()
        driver.get(base_url)
        time.sleep(3)

        for _ in range(3):
            driver.find_element(By.CSS_SELECTOR, 'a.section_more_inner').click()
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
                
                print(title, detail, publisher, date)

    driver.quit()

data_extract(100)