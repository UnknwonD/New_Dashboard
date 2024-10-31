import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
from pyvis.network import Network
import altair as alt
import os
from kiwipiepy import Kiwi
from textblob import TextBlob
import streamlit.components.v1 as components
import ast
from sqlalchemy import create_engine, text
from api import db_url
from gensim.models import Word2Vec

from transformers import BertTokenizerFast, BertForSequenceClassification
import torch

stopwords = ['대하', '때문', '경우', '그리고', '그러나', '하지만', '또한', '또는', '따라서', '그래서', '하지만', '이', '그', '저', '것', '수', '등', '및', '을', '를', '은', '는', '이', '가', '에', '와', '과', '에서', '이다', '있다', '없다', '되다', '하다', '않다', '같다', '때문에', '위해', '대한', '여러', '모든', '어떤', '하면', '그러면']

# 토크나이저와 모델 로드
tokenizer = BertTokenizerFast.from_pretrained("sangrimlee/bert-base-multilingual-cased-nsmc")
model = BertForSequenceClassification.from_pretrained("sangrimlee/bert-base-multilingual-cased-nsmc")

# 디바이스 설정 (GPU 사용 가능 시)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)  # 모델을 GPU로 이동

@st.cache_data
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

# # Sentiment analysis function
# def analyze_sentiment(text):
#     analysis = TextBlob(text)
#     if analysis.sentiment.polarity > 0:
#         return '긍정'
#     elif analysis.sentiment.polarity == 0:
#         return '중립'
#     else:
#         return '부정'

@st.cache_data
def create_wordcloud(text):
    # Use font_path to correctly display Korean text, words are horizontal only
    wordcloud = WordCloud(width=1500, height=1200, background_color='white', font_path='malgun.ttf', prefer_horizontal=1.0).generate(text)
    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    return fig


def find_similar_words(word, model, topn=3):
    try:
        similar_words = model.wv.most_similar(word, topn=topn)
        return [w[0] for w in similar_words]
    except KeyError:
        return []

@st.cache_data
def train_word2vec_model(sentences):
    # Train a Word2Vec model
    model = Word2Vec(sentences, vector_size=100, window=5, min_count=2, workers=4)
    return model

def visualize_main_word_network(category, top_keywords, w2v_model):
    net = Network(notebook=False, 
                height='60rem', 
                width='100%', 
                bgcolor='#ffffff', 
                font_color='black',
                cdn_resources="remote",
                layout=True,
                neighborhood_highlight=True)
    
    # Add main node for the category
    net.add_node(category, label=category, size=30, color='red')
    
    # Add nodes for top keywords and connect them to the category node
    for keyword in top_keywords:
        net.add_node(keyword, label=keyword, size=20, color='lightblue', shape='circle')
        net.add_edge(category, keyword, color='gray')
        
        # Find related words for each keyword
        similar_words = find_similar_words(keyword, w2v_model, topn=3)
        for similar_word in similar_words:
            net.add_node(similar_word, label=similar_word, size=15, color='lightgreen', shape='circle')
            net.add_edge(keyword, similar_word, color='lightgray')
    
    # Ensure the output directory exists
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    output_path = os.path.join(output_dir, f'{category}_word_network.html')
    net.write_html(output_path)
    with open(output_path, 'r', encoding='utf-8') as HtmlFile:
        return HtmlFile.read()


def visualize_expanded_word_network(main_word, w2v_model):
    net = Network(notebook=False, height='600px', width='100%', bgcolor='#ffffff', font_color='black', layout=True)
    
    # Add main node for the selected word
    net.add_node(main_word, label=main_word, size=30, color='red', physics=False)
    
    # Find similar words and add to the network
    try:
        similar_words = w2v_model.wv.most_similar(main_word, topn=10)
        for similar_word, similarity in similar_words:
            net.add_node(similar_word, label=similar_word, size=20, color='lightgreen', physics=False)
            net.add_edge(main_word, similar_word, value=similarity, color='gray', physics=False)
            
            # Find similar words of similar words (2nd level)
            similar_words_level_2 = w2v_model.wv.most_similar(similar_word, topn=5)
            for sub_word, sub_similarity in similar_words_level_2:
                net.add_node(sub_word, label=sub_word, size=10, color='lightyellow', physics=False)
                net.add_edge(similar_word, sub_word, value=sub_similarity, color='lightgray', physics=False)
    except KeyError:
        # Skip if the keyword is not in the vocabulary
        pass
    
    # Ensure the output directory exists
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    output_path = os.path.join(output_dir, f'{main_word}_expanded_network.html')
    net.write_html(output_path)
    with open(output_path, 'r', encoding='utf-8') as HtmlFile:
        return HtmlFile.read()

# Load or create dataframe
@st.cache_data
def data_load(target_date):
    # 데이터베이스 엔진 생성
    engine = create_engine(db_url)
    
    # SQL 쿼리 생성
    sql = f'''
    SELECT * 
    FROM social_data 
    WHERE url IS NOT NULL 
    AND DATE(date) = '{target_date.strftime('%Y-%m-%d')}'
    '''

    sql = text(sql)
    
    # 데이터 로드 및 날짜 형식 변환
    with engine.connect() as conn:
        df = pd.read_sql(sql, conn)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    # content와 sentences의 '-' 값을 리스트로 변환 후 처리
    df['content'] = df['content'].replace('-', "['-']")
    df['sentences'] = df['sentences'].replace('-', "['-']")

    # content와 sentences를 리스트 타입으로 변환
    df['content'] = df['content'].apply(ast.literal_eval)
    df['sentences'] = df['sentences'].apply(ast.literal_eval)
    
    return df

import streamlit as st
import pandas as pd
import altair as alt
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from kiwipiepy import Kiwi
import streamlit.components.v1 as components
import os

import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from prophet import Prophet
import streamlit as st

def stock_prediction_dashboard():
    st.title('주가 예측 대시보드 📈')
    st.write("이 페이지에서는 주식 코드와 예측 기간을 입력하여 해당 주식의 향후 가격을 예측합니다.")

    # 주식 코드 입력 폼
    with st.form(key='stock_form'):
        st.markdown("### 주식 코드와 예측 기간을 입력하세요:")
        stock_symbol = st.text_input('주식 코드를 입력하세요 (예: TSLA, AAPL 등)', value='TSLA')
        prediction_period = st.number_input('예측할 기간을 입력하세요 (일 단위, 최대 30일)', min_value=1, max_value=30, value=10)
        submit_button = st.form_submit_button(label='예측하기')

    if submit_button:
        try:
            # 주식 데이터 불러오기 (최근 3년, 1일 단위)
            stock = yf.Ticker(stock_symbol)
            data = stock.history(period="5y", interval="1d")  # 1일 단위로 최근 3년 데이터 가져오기

            # 데이터가 비어 있을 경우 예외 처리
            if data.empty:
                st.error("주식 데이터를 불러오지 못했습니다. 주식 코드가 올바른지 확인해주세요.")
                return

            # Prophet을 사용하기 위해 데이터프레임 형식 변환 (타임존 제거)
            prophet_data = data.reset_index()[['Date', 'Close']]
            prophet_data['Date'] = prophet_data['Date'].dt.tz_localize(None)  # 타임존 제거
            prophet_data.rename(columns={'Date': 'ds', 'Close': 'y'}, inplace=True)

            # Prophet 모델 설정 및 하이퍼파라미터 조정
            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
                changepoint_prior_scale=0.05  # 변동점 민감도 조정
            )
            model.add_seasonality(name='monthly', period=30.5, fourier_order=5)  # 월별 계절성 추가
            model.fit(prophet_data)

            # 예측을 위한 데이터프레임 생성
            future = model.make_future_dataframe(periods=prediction_period)
            forecast = model.predict(future)

            # 음수 예측 값을 0으로 변환
            forecast['yhat'] = forecast['yhat'].apply(lambda x: max(x, 0))

            # 예측 결과 시각화
            st.subheader('예측 결과 그래프')
            plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows 환경에서 한글 폰트 설정
            plt.figure(figsize=(12, 6))
            plt.plot(prophet_data['ds'], prophet_data['y'], label='실제 가격', color='blue')
            plt.plot(forecast['ds'], forecast['yhat'], label='예측 가격', color='red')
            plt.xlabel('날짜')
            plt.ylabel('가격')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.6)
            st.pyplot(plt)

            # 예측 결과 출력 (예측한 기간만)
            st.subheader('예측 결과 데이터')
            future_predictions = forecast[['ds', 'yhat']].tail(prediction_period)
            future_predictions.columns = ['날짜', '예측 가격']
            st.dataframe(future_predictions)

            # 상세 결과 개별 표시
            st.markdown("### 예측된 가격 상세 보기:")
            for _, row in future_predictions.iterrows():
                st.write(f"- 날짜: {row['날짜']}, 예측 가격: {row['예측 가격']:.2f}")

        except Exception as e:
            st.error(f"오류가 발생했습니다: {str(e)}")

def main():
    # 페이지 기본 설정
    # st.set_page_config(layout='wide', page_title='데일리 뉴스 리포트 대시보드', page_icon='📊')
    st.title('데일리 뉴스 리포트 대시보드 📊')

    # 사이드바 및 페이지 제목
    st.sidebar.title('데일리 뉴스 리포트')
    st.sidebar.subheader("데이터 선택")
    selected_date = st.sidebar.date_input('날짜 선택', pd.Timestamp('today'))

    # 사이드바 - 특정 단어 필터링 기능
    st.sidebar.subheader("🔍 특정 단어로 기사 필터링")
    filter_keywords = st.sidebar.text_area("검색할 단어들을 입력하세요 (쉼표로 구분):")
    filter_keywords = [word.strip() for word in filter_keywords.split(',') if word.strip()]  # 쉼표로 구분된 단어 리스트로 변환
    reset_filter = st.sidebar.button("🔄 필터 초기화")

    # 데이터 불러오기
    if selected_date:
        df = data_load(selected_date)
        if df.empty:
            st.warning("선택한 날짜에 해당하는 데이터가 없습니다.")
        else:
            # 필터 적용
            if filter_keywords:
                df = df[df['content'].apply(lambda x: any(keyword in ' '.join(x) for keyword in filter_keywords))]

            if reset_filter:
                filter_keywords = []  # 필터 초기화

            st.title(f"📍 {selected_date.strftime('%Y년 %m월 %d일')} 데일리 뉴스 리포트")
            # 탭 구조로 뉴스 세부 정보 표시 (탭을 상단에 배치)
            tab_labels = ['메인', '정치', '경제', '사회', '생활/문화', 'IT/과학']
            tabs = st.tabs(tab_labels)

            # 사이드바 - 실시간 WORDCOUNT TOP 10 단어
            st.sidebar.subheader("🔥 실시간 인기 단어 TOP 10")
            all_tokens = []
            kiwi = Kiwi()
            for sublist in df['sentences']:
                for sentence in sublist:
                    for word in sentence:
                        analyzed = kiwi.analyze(word)
                        if analyzed:
                            morphs = analyzed[0][0]
                            for token in morphs:
                                if token.tag.startswith('N') and len(token.form) > 1:
                                    all_tokens.append(token.form)
            word_count = Counter(all_tokens)
            word_count_df = pd.DataFrame(word_count.items(), columns=['Word', 'Count']).sort_values(by='Count', ascending=False).head(10)
            for i, (index, row) in enumerate(word_count_df.iterrows()):
                st.sidebar.markdown(f"**{i + 1}. {row['Word']}**")

            # 메인 화면 레이아웃 - 컬럼 사용으로 가독성 개선
            with tabs[0]:
                main_container = st.container()
                with main_container:
                    st.markdown("---")
                    summary_col, chart_col = st.columns([2, 1], gap="medium")

                    # 주요 뉴스 요약 정보 (분야별로 5개씩만 보이도록 수정)
                    with summary_col:
                        st.subheader('📰 주요 뉴스')
                        categories = df['category'].unique()
                        for category in categories:
                            st.markdown(f"### 🌐 {category} 뉴스")
                            category_news = df[df['category'] == category].tail(5)
                            for i, (index, row) in enumerate(category_news.iterrows()):
                                st.markdown(f"<div style='margin-bottom: 10px;'><strong>{i + 1}. <a href='{row['url']}' target='_blank'>{row['title']}</a></strong> 🌐 {row['publisher']}</div>", unsafe_allow_html=True)

                    # 분야별 뉴스 개수 및 긍/부정 비율 시각화
                    with chart_col:
                        st.subheader('📊 분야별 뉴스 개수')
                        news_count_by_category = df['category'].value_counts()
                        news_count_df = pd.DataFrame({'Category': news_count_by_category.index, 'Count': news_count_by_category.values})
                        category_chart = alt.Chart(news_count_df).mark_bar(color='steelblue').encode(
                            x=alt.X('Count', sort='-y'),
                            y=alt.Y('Category', sort='-x', axis=alt.Axis(labelFontSize=12)),
                            tooltip=['Category', 'Count']
                        ).properties(height=300)
                        st.altair_chart(category_chart, use_container_width=True)

                        # 분야별 긍/부정 비율 시각화
                        st.subheader('📊 분야별 긍정/부정 비율')
                        df['sentiment'] = df['content'].apply(lambda x: analyze_sentiment(' '.join(x)))
                        sentiment_category_df = df.groupby(['category', 'sentiment']).size().reset_index(name='count')
                        sentiment_chart = alt.Chart(sentiment_category_df).mark_bar().encode(
                            x=alt.X('count', title='Count'),
                            y=alt.Y('category', title='Category', sort='-x'),
                            color='sentiment',
                            tooltip=['category', 'sentiment', 'count']
                        ).properties(height=300)
                        st.altair_chart(sentiment_chart, use_container_width=True)

            # 카테고리별 탭 구성
            for idx, selected_category in enumerate(tab_labels[1:]):
                with tabs[idx + 1]:
                    filtered_data = df[df['category'] == selected_category]
                    if filtered_data.empty:
                        st.warning('해당 카테고리에 대한 데이터가 없습니다.')
                        continue

                    st.subheader(f"📝 {selected_category} 뉴스 분석")
                    sentences = [sentence for sublist in filtered_data['sentences'] for sentence in sublist]
                    w2v_model = train_word2vec_model(sentences)

                    # 카테고리 상세 뉴스 시각화 레이아웃
                    st.markdown("---")
                    st.subheader('💭 가장 많이 발생한 단어 및 네트워크 분석')
                    cloud_network_col1, cloud_network_col2 = st.columns([1, 1], gap="large")

                    # 워드 클라우드 및 주요 단어 분석
                    with cloud_network_col1:
                        st.subheader('🔍 워드 클라우드')
                        tokens = []
                        for sublist in filtered_data['sentences']:
                            for sentence in sublist:
                                for word in sentence:
                                    analyzed = kiwi.analyze(word)

                                    if analyzed:
                                        morphs = analyzed[0][0]

                                    for token in morphs:
                                        if token.tag.startswith('N') and len(token.form) > 1 and token.form not in stopwords:
                                            tokens.append(token.form)
                        all_text = ' '.join(tokens)
                        wordcloud_fig = create_wordcloud(all_text)
                        st.pyplot(wordcloud_fig)

                        # 주요 단어 빈도 테이블
                        word_count = Counter(tokens)
                        word_count_df = pd.DataFrame(word_count.items(), columns=['Word', 'Count']).sort_values(by='Count', ascending=False).head(10)
                        st.table(word_count_df)

                    # 워드 네트워크 시각화
                    with cloud_network_col2:
                        st.subheader('🌐 단어 네트워크')
                        for word in word_count_df['Word']:
                            with st.expander(f"🛠️ {word} 유사 단어 네트워크 보기"):
                                expanded_network_html = visualize_expanded_word_network(word.replace('/', '_'), w2v_model)
                                components.html(expanded_network_html, height=500)

                    # 긍정, 부정 평가 시각화 및 뉴스 예시
                    st.markdown("---")
                    st.subheader('🗳️ 긍정, 부정 평가 비율 및 뉴스')
                    pos_neg_col1, pos_neg_col2 = st.columns([1, 1], gap="large")

                    # 긍정, 부정 평가 시각화
                    with pos_neg_col1:
                        st.subheader('📊 긍/부정 비율')
                        sentiments = filtered_data['content'].apply(lambda x: analyze_sentiment(' '.join(x)))
                        filtered_data['sentiment'] = sentiments
                        sentiment_counts = sentiments.value_counts().to_dict()
                        sentiment_df = pd.DataFrame(list(sentiment_counts.items()), columns=['Sentiment', 'Count'])
                        pie_chart = alt.Chart(sentiment_df).mark_arc(innerRadius=50).encode(
                            theta=alt.Theta('Count', stack=True),
                            color=alt.Color('Sentiment', scale=alt.Scale(scheme='category10')),
                            tooltip=['Sentiment', 'Count']
                        ).properties(height=300)
                        st.altair_chart(pie_chart, use_container_width=True)

                    # 긍정, 부정 뉴스
                    with pos_neg_col2:
                        st.subheader('✅ 긍정 뉴스 TOP 5')
                        positive_data = filtered_data[filtered_data['sentiment'] == '긍정'].tail(5)
                        for i, (index, row) in enumerate(positive_data.iterrows()):
                            st.markdown(f"<div style='margin-bottom: 10px;'><strong>{i + 1}. <a href='{row['url']}' target='_blank'>{row['title']}</a></strong> 🌐 {row['publisher']}</div>", unsafe_allow_html=True)

                        st.subheader('❌ 부정 뉴스 TOP 5')
                        negative_data = filtered_data[filtered_data['sentiment'] == '부정'].tail(5)
                        for i, (index, row) in enumerate(negative_data.iterrows()):
                            st.markdown(f"<div style='margin-bottom: 10px;'><strong>{i + 1}. <a href='{row['url']}' target='_blank'>{row['title']}</a></strong> 🌐 {row['publisher']}</div>", unsafe_allow_html=True)

                    # 긍정, 부정 뉴스의 주요 단어 분석
                    st.markdown("---")
                    st.subheader('💬 긍정 및 부정 뉴스에서 가장 많이 발생한 단어')
                    pos_neg_word_col1, pos_neg_word_col2 = st.columns([1, 1], gap="large")

                    with pos_neg_word_col1:
                        st.subheader('💬 긍정 뉴스에서 가장 많이 발생한 단어')
                        positive_tokens = []
                        for sublist in positive_data['sentences']:
                            for sentence in sublist:
                                for word in sentence:
                                    analyzed = kiwi.analyze(word)
                                    if analyzed:
                                        morphs = analyzed[0][0]
                                        for token in morphs:
                                            if token.tag.startswith('N') and len(token.form) > 1 and token.form not in stopwords:
                                                positive_tokens.append(token.form)
                        positive_word_count = Counter(positive_tokens)
                        positive_word_count_df = pd.DataFrame(positive_word_count.items(), columns=['Word', 'Count']).sort_values(by='Count', ascending=False).head(10)
                        st.table(positive_word_count_df)

                    with pos_neg_word_col2:
                        st.subheader('💬 부정 뉴스에서 가장 많이 발생한 단어')
                        negative_tokens = []
                        for sublist in negative_data['sentences']:
                            for sentence in sublist:
                                for word in sentence:
                                    analyzed = kiwi.analyze(word)
                                    if analyzed:
                                        morphs = analyzed[0][0]
                                        for token in morphs:
                                            if token.tag.startswith('N') and len(token.form) > 1 and token.form not in stopwords:
                                                negative_tokens.append(token.form)
                        negative_word_count = Counter(negative_tokens)
                        negative_word_count_df = pd.DataFrame(negative_word_count.items(), columns=['Word', 'Count']).sort_values(by='Count', ascending=False).head(10)
                        st.table(negative_word_count_df)

                    st.markdown("---")
                    st.subheader(f'🛈 {selected_category} 중심 워드 네트워크')
                    if len(tokens) > 1:
                        top_keywords = word_count_df['Word'].tolist()
                        word_network_html = visualize_main_word_network(selected_category.replace('/', '_'), top_keywords, w2v_model)
                        components.html(word_network_html, height=500)
                    else:
                        st.warning('워드 네트워크를 생성하기에 충분한 데이터가 없습니다.')

if __name__ == "__main__":
    # main()

    st.set_page_config(layout='wide', page_title='종합 대시보드', page_icon='📊')
    st.sidebar.title('📊 대시보드 메뉴')
    page = st.sidebar.radio("이동할 페이지를 선택하세요:", ('데일리 뉴스 리포트', '주가 예측 대시보드'))

    if page == '데일리 뉴스 리포트':
        main()
    elif page == '주가 예측 대시보드':
        stock_prediction_dashboard()
