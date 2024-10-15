from tqdm import tqdm
import pandas as pd
from sqlalchemy import create_engine, Table, MetaData, select
from api import db_url

import pandas as pd
from sqlalchemy import create_engine

# PostgreSQL 연결 설정
engine = create_engine(db_url)

# PostgreSQL 테이블 데이터 불러오기
query = "SELECT * FROM social_data"
df_existing = pd.read_sql(query, engine)

# CSV 파일 데이터 불러오기
df_csv = pd.read_csv('news_data.csv', encoding='utf-8-sig')

# 중복을 제거하고 새롭게 삽입할 데이터만 필터링
# 기준이 되는 열(예: 'id' 또는 'title' 등) 설정
df_new = df_csv[~df_csv['title'].isin(df_existing['title'])]

print(len(df_new))

# 새로운 데이터만 PostgreSQL에 삽입
df_new.to_sql('social_data', engine, if_exists='append', index=False)

print(f"{len(df_new)} rows inserted into the social_data table.")
