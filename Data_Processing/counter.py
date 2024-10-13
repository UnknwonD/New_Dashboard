import pandas as pd
from collections import Counter

# title과 content를 합쳐서 입력으로 넣으면, 그 때 해당하는 단어의 counter를 반환하는 함수
def cal_counter(content:list):
    ...
    
# df를 넣으면 해당 data의 counter.csv를 만들어주는 함수
def update_db(df:pd.DataFrame):
    ...
    
    

# df를 불러와서 단위 기간별로 모든 콘텐츠를 모아서
# 그 때의 trend를 분석하는 함수를 만들 예정
# (단위기간 : month, day)
def main():
    df = pd.raed_csv('df_to_daylist.csv', encoding='utf-8-sig')
    
    
    
    

if __name__ == '__main__':
    main()