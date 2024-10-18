import pandas as pd
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, MetaData, Table, UniqueConstraint
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from api import db_url

def update_db(df):
    # PostgreSQL 연결 설정
    engine = create_engine(db_url)

    # PostgreSQL 테이블 데이터 불러오기
    query = "SELECT * FROM social_data"
    df_existing = pd.read_sql(query, engine)

    # 중복을 제거하고 새롭게 삽입할 데이터만 필터링
    # 기준이 되는 열(예: 'id' 또는 'title' 등) 설정
    df_new = df[~df['title'].isin(df_existing['title'])]

    print(len(df_new))

    # 새로운 데이터만 PostgreSQL에 삽입
    df_new.to_sql('social_data', engine, if_exists='append', index=False)

    print(f"{len(df_new)} rows inserted into the social_data table.")


if __name__ == '__main__':
    # CSV 파일을 DataFrame으로 불러오기
    csv_file_path = 'news_data.csv'
    df = pd.read_csv(csv_file_path)

    # 데이터프레임의 날짜 형식을 datetime 타입으로 변환
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    # 데이터베이스 연결 설정
    engine = create_engine(db_url)
    metadata = MetaData()

    # 테이블 정의
    table_name = 'social_data'
    table = Table(
        table_name,
        metadata,
        Column('seq', Integer, primary_key=True, autoincrement=True),
        Column('category', String, nullable=False),
        Column('sub_category', String, nullable=True),
        Column('title', String, nullable=False, unique=True),
        Column('content', Text, nullable=True),
        Column('publisher', String, nullable=True),
        Column('date', DateTime, nullable=True),
        UniqueConstraint('title', name='uq_title')  # title 컬럼의 중복을 방지
    )

    # 테이블 생성
    metadata.create_all(engine)

    # 데이터 삽입을 위한 세션 생성
    Session = sessionmaker(bind=engine)
    session = Session()

    # DataFrame 데이터를 PostgreSQL 테이블에 삽입
    try:
        df.to_sql(table_name, engine, if_exists='append', index=False)
        session.commit()
    except Exception as e:
        print("Error occurred:", e)
        session.rollback()
    finally:
        session.close()