# -*- coding: utf-8 -*-

from summarizer import Summarizer
from transformers import pipeline
from kiwipiepy import Kiwi
from collections import Counter


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

이 상황에서 북한과 러시아 간의 관계는 더욱 긴밀해질 것으로 예상되며, 이는 한반도 및 주변 국가들의 안보 상황에 큰 영향을 미칠 가능성이 있다. 전문가들은 이러한 상황이 지속된다면, 한국과 일본을 비롯한 주변국들이 이에 대한 대응책을 마련해야 할 것이라고 경고하고 있다. 특히 한국 정부는 미국과의 협력을 강화하고, 군사적 대비 태세를 더욱 강화해야 할 필요가 있을 것이다. 또한 국제사회는 북한의 이러한 군사적 움직임에 대해 강력히 규탄하고, 추가적인 제재를 가할 가능성이 있다. 그러나 이러한 제재가 북한의 행동을 억제할 수 있을지는 불확실하다. 북한은 과거에도 국제사회의 제재에 굴하지 않고 자신들의 군사적 목표를 추진해왔기 때문이다. 따라서 국제사회는 북한에 대한 제재와 함께 외교적 노력을 병행해야 할 것이다. 러시아 또한 이번 사태에서 중요한 역할을 하고 있다. 러시아는 북한과의 군사 협력을 통해 자신들의 이익을 극대화하려는 의도를 보이고 있으며, 이는 서방 국가들과의 갈등을 더욱 격화시킬 수 있는 요인으로 작용하고 있다. 이에 따라 미국과 유럽 연합은 러시아에 대한 추가 제재를 검토하고 있다.

한편, 이러한 군사적 긴장 상황은 경제에도 영향을 미치고 있다. 국제 유가가 급등하고 있으며, 주식 시장도 불안정한 모습을 보이고 있다. 전문가들은 이러한 경제적 불안정성이 장기화될 경우, 세계 경제에 부정적인 영향을 미칠 수 있다고 경고하고 있다. 특히 에너지 의존도가 높은 국가들은 유가 상승으로 인한 경제적 타격을 받을 가능성이 크다. 따라서 각국 정부는 이러한 상황에 대비해 에너지 자원의 다변화를 추진하고, 경제 안정을 위한 정책을 마련해야 할 것이다. 북한의 군사적 움직임과 러시아와의 협력, 그리고 이에 따른 국제사회의 대응은 앞으로도 계속해서 주목해야 할 중요한 이슈가 될 것이다. 이러한 상황에서 한국을 비롯한 주변 국가들은 안보와 경제적 안정을 동시에 추구해야 하는 어려운 과제에 직면해 있다.
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
