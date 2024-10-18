# -*- coding: utf-8 -*-

from summarizer import Summarizer
from transformers import pipeline
from kiwipiepy import Kiwi
from collections import Counter


def extractive_summarize_korean_text(text, compression_ratio=0.3):
    # BERT 기반 추출 요약 모델 생성
    model = Summarizer()
    summary = model(text, ratio=compression_ratio)
    return summary

def analyze_sentiment(summary):
    # 감정 분석 파이프라인 생성 (긍정/부정 분석)
    sentiment_pipeline = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment", device=0)
    sentiment = sentiment_pipeline(summary)[0]
    # 감정 분석 결과를 해석 가능한 형태로 변환
    sentiment_mapping = {
        "1 star": "매우 부정적",
        "2 stars": "부정적",
        "3 stars": "중립적",
        "4 stars": "긍정적",
        "5 stars": "매우 긍정적"
    }
    sentiment['label'] = sentiment_mapping.get(sentiment['label'], sentiment['label'])
    return sentiment

def extract_keywords(summary, num_keywords=5):
    # Kiwi 토크나이저를 사용하여 키워드 추출
    kiwi = Kiwi()
    tokens = kiwi.tokenize(summary)
    words = [token.form for token in tokens if token.tag.startswith('N') or token.tag.startswith('V')]
    word_freq = Counter(words)
    
    # 빈도 순으로 정렬하여 상위 키워드 추출
    keywords = [word for word, freq in word_freq.most_common(num_keywords)]
    return keywords

if __name__ == "__main__":
    text = """
    국가정보원은 “북한이 지난 8일부터 러시아 파병을 위한 특수부대 병력 이동을 시작했다”고 18일 밝혔다. 북한이 지상군을 대규모로 파병하는 것은 이번이 처음이다. 국제사회가 북·러 간 군사협력 강화를 우려하는 가운데 나온 이번 북한군 파병은 향후 한반도 안보 지형에도 막대한 영향을 미칠 것으로 보인다. 또 한·러 관계를 고려해 우크라이나에 비살상무기만 지원해왔던 정부의 방침에도 영향을 끼칠 것이란 전망이 나온다. 윤석열 대통령은 이날 긴급 국가안전보장회의(NSC)를 주재했다. 이후 대통령실은 “현 상황을 좌시하지 않고 국제사회와 공동으로 가용한 모든 수단을 동원해 나갈 것”이라며 강경 대응을 예고했다.

국정원은 이날 기자단에 배포한 보도자료에서 “북한군의 동향을 밀착 감시하던 중 북한이 지난 8일부터 13일까지 러시아 해군 수송함을 통해 특수부대를 러시아 지역으로 수송하는 것을 포착, 북한군의 참전 개시를 확인했다”고 밝혔다. 이어 “러시아 태평양함대 소속 상륙함 4척 및 호위함 3척이 해당 기간 북한의 청진·함흥·무수단 인근 지역에서 특수부대원 1500여 명을 러시아 블라디보스토크로 1차 이송 완료했고, 조만간 2차 수송 작전이 진행될 예정”이라고 덧붙였다.

이와 관련, 대북 소식통은 “북한군은 ‘폭풍군단’으로 불리는 최정예 특수작전부대인 11군단 소속 4개 여단(1만여 명 규모) 병력을 파병할 것으로 예상된다”고 말했다. 평남 덕천시에 주둔 중인 폭풍군단은 예하에 총 10개 여단(저격여단 3개, 경보병여단 4개, 항공육전여단 3개로 구성)을 두고 있으며, 수도권 및 후방 침투 임무 등을 수행하는 특수전 부대다. 국정원에 따르면 러시아 해군 함대(수송 지원)의 북한 해역 진입은 1990년 이후 처음이다. 또 러시아 공군 소속 AN-124 등 대형 수송기도 블라디보스토크와 평양을 수시로 오가고 있다고 한다.
    """
    
    # 요약을 30%로 압축
    summary = extractive_summarize_korean_text(text, compression_ratio=0.3)
    print("Original Text:\n", text)
    print("\nSummary:\n", summary)

    # 감정 분석 수행
    sentiment = analyze_sentiment(summary)
    print("\nSentiment Analysis:\n", sentiment)

    # 주요 키워드 추출
    keywords = extract_keywords(summary)
    print("\nKeywords:\n", keywords)
