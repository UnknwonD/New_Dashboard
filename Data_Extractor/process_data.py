import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from kiwipiepy import Kiwi
from transformers import BertTokenizerFast, BertForSequenceClassification
import torch
from api import db_url
from tqdm import tqdm
import ast

# 토크나이저와 모델 로드
tokenizer = BertTokenizerFast.from_pretrained("sangrimlee/bert-base-multilingual-cased-nsmc")
model = BertForSequenceClassification.from_pretrained("sangrimlee/bert-base-multilingual-cased-nsmc")

# 디바이스 설정 (GPU 사용 가능 시)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)  # 모델을 GPU로 이동

def analyze_sentiment(text):
    # 입력 텍스트 토크나이징
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}  # 입력 데이터를 GPU로 이동

    # 모델 예측 수행 (그라디언트 계산 비활성화)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits

    # 감정 분류 결과 도출
    predicted_class = torch.argmax(logits, dim=1).item()
    if predicted_class == 0:
        return '부정'
    else:
        return '긍정'

# DB 연결 세션 생성
echo_engine = create_engine(db_url)
Session = sessionmaker(bind=echo_engine)
session = Session()

# KiWi 초기화
kiwi = Kiwi()

# 데이터 불러오기 (미처리 데이터)
with echo_engine.connect() as conn:
    df = pd.read_sql(text("SELECT seq, content, sentences FROM social_data WHERE sentiment IS NULL OR keywords IS NULL and content is not null"), conn)

# 데이터 포맷 정리 및 변환 (데이터를 문자열에서 리스트로 변환)
df['content'] = df['content'].replace('-', "['-']")
df['content'] = df['content'].apply(lambda x: ast.literal_eval(x))

df['sentences'] = df['sentences'].replace('-', "['-']") 
df['sentences'] = df['sentences'].apply(lambda x: ast.literal_eval(x))

# 필터링 및 처리할 데이터 담기
for index, row in tqdm(df.iterrows(), desc='DATA PREPROCESSING', total=len(df)):
    seq = row['seq']
    content = row['content']  # content는 리스트 형식
    sentences = row['sentences']  # sentences는 리스트 안의 리스트 형식

    # Sentiment 분석 (content를 하나의 문자열로 합침)
    sentiment = analyze_sentiment(' '.join(content))

    # Keywords 추출 (sentences의 각 단어에 대해 형태소 분석)
    keywords = []
    for sublist in sentences:
        for word in sublist:
            analyzed = kiwi.analyze(word)
            if analyzed:
                morphs = analyzed[0][0]
                for token in morphs:
                    if token.tag == "NNP" and len(token.form) > 1:  # 고유 명사만 추출
                        keywords.append(token.form)
    keywords_string = ', '.join(set(keywords))

    # DB 업데이트
    with echo_engine.connect() as conn:
        sql = text("UPDATE social_data SET sentiment = :sentiment, keywords = :keywords WHERE seq = :seq")
        conn.execute(sql, {"sentiment": sentiment, "keywords": keywords_string, "seq": seq})
        conn.commit()
