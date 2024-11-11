import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
from pyvis.network import Network
import altair as alt
import plotly.graph_objects as go
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

stopwords = ['대하', '때문', '경우', '그리고', '그러나', '하지만', '또한', '또는', '따라서', 
             '그래서', '하지만', '이', '그', '저', '것', '수', '등', '및', '을', '를', '은', '는', '이', 
             '가', '에', '와', '과', '에서', '이다', '있다', '없다', '되다', '하다', '않다', '같다', '때문에',
            '위해', '대한', '여러', '모든', '어떤', '하면', '그러면', '연합뉴스']

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
def data_load(target_date, word_like = None):
    # 데이터베이스 엔진 생성
    engine = create_engine(db_url)
    
    # SQL 쿼리 생성
    if word_like:
        sql = f'''
        SELECT * 
        FROM social_data 
        WHERE url IS NOT NULL 
        AND title like '%{word_like}%'
        ORDER BY seq DESC
        LIMIT 10
        ''' 
    else:
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


# 주식 데이터 가져오기 함수
def get_stock_data(stock_symbol, period="5y", interval="1d"):
    stock = yf.Ticker(stock_symbol)
    return stock.history(period=period, interval=interval)

# Streamlit 주가 예측 대시보드
def stock_prediction_dashboard():
    st.title('주가 예측 대시보드 📈')
    st.write("이 페이지에서는 주식 코드와 예측 기간을 입력하여 해당 주식의 향후 가격을 예측합니다.")

    # 주식 코드 입력 폼
    with st.form(key='stock_form'):
        st.markdown("### 주식 코드와 예측 기간을 입력하세요:")
        stock_symbol = st.text_input('주식 코드를 입력하세요 (예: TSLA, AAPL 등)', value='TSLA')
        prediction_period = st.number_input('예측할 기간을 입력하세요 (일 단위, 최대 30일)', min_value=1, max_value=30, value=10)
        related_word = st.text_input('해당 주식과 연관이 있는 키워드를 입력하세요 (예: 트럼프, 테슬라 등, 선택)')
        submit_button = st.form_submit_button(label='예측하기')

    if submit_button:
        try:
            # 주식 데이터 불러오기 (최근 5년, 1일 단위)
            data = get_stock_data(stock_symbol)

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

            # 캔들 차트 및 예측 결과 시각화
            st.subheader('캔들 차트 및 예측 결과')
            fig = go.Figure()

            # 캔들차트 추가
            fig.add_trace(go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name='캔들 차트'
            ))

            # 예측 라인 추가
            fig.add_trace(go.Scatter(
                x=forecast['ds'],
                y=forecast['yhat'],
                mode='lines',
                name='예측 가격',
                line=dict(color='red', width=2)
            ))

            # 주식 실시간 가격 업데이트 및 차이 표시
            current_price = data['Close'][-1]
            predicted_price = forecast['yhat'].iloc[-1]
            price_difference = predicted_price - current_price
            price_color = 'red' if price_difference > 0 else 'blue'

            st.markdown(f"### {stock_symbol} 실시간 가격: ${current_price:.2f}")
            st.markdown(f"### 예측 가격과의 차이: <span style='color:{price_color};'>${price_difference:.2f}</span>", unsafe_allow_html=True)

            # 차트 레이아웃 설정 및 출력
            fig.update_layout(
                title=f'{stock_symbol} 주가 및 예측 결과',
                xaxis_title='날짜',
                yaxis_title='가격',
                xaxis_rangeslider_visible=False
            )

            st.plotly_chart(fig)

            # 관련 뉴스 출력
            col1, col2 = st.columns([0.3, 0.7])
            with col1:
                # 예측 결과 출력 (예측한 기간만)
                st.subheader('예측 결과 데이터')
                future_predictions = forecast[['ds', 'yhat']].tail(prediction_period)
                future_predictions.columns = ['날짜', '예측 가격']
                st.dataframe(future_predictions)


            with col2:
                if len(related_word) > 1:
                    df_related = data_load(None, related_word)

                    st.markdown(f"### 🌐 {related_word} 관련 뉴스")
                    category_news = df_related.tail(10)
                    for i, (index, row) in enumerate(category_news.iterrows()):
                        st.markdown(f"<div style='margin-bottom: 10px;'><strong>{i + 1}. <a href='{row['url']}' target='_blank'>{row['title']}</a></strong> 🌐 {row['publisher']}</div>", unsafe_allow_html=True)
                else:
                    st.subheader('🗅 주요 뉴스')

        except Exception as e:
            st.error(f"오류가 발생했습니다: {str(e)}")
            st.write('연관어가 입력되지 않았습니다.')

def daily_news_dashboard():
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
            ##################################################################################################################################
            for sublist in df['sentences']:
                for sentence in sublist:
                    for word in sentence:
                        analyzed = kiwi.analyze(word)
                        if analyzed:
                            morphs = analyzed[0][0]
                            for token in morphs:
                                if token.tag.startswith('N') and len(token.form) > 1 and token.tag == "NNP":
                                    all_tokens.append(token.form)
            ##################################################################################################################################
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
                        
                        ##################################################################################################################################
                        with engine.connect() as conn:
                            for i, row in df.iterrows():
                                # Assuming there's an identifier or column you can match on (e.g., 'id')
                                sql = text("UPDATE social_data SET sentiment = :sentiment WHERE seq = :seq")
                                conn.execute(sql, {"sentiment": row['sentiment'], "seq": row['seq']})
                            conn.commit()
                        ##################################################################################################################################

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
                        ##################################################################################################################################
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
                        ##################################################################################################################################
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
                        ##################################################################################################################################
                        sentiments = filtered_data['content'].apply(lambda x: analyze_sentiment(' '.join(x)))
                        filtered_data['sentiment'] = sentiments
                        ##################################################################################################################################
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
                        ##################################################################################################################################
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
                        ##################################################################################################################################
                        positive_word_count = Counter(positive_tokens)
                        positive_word_count_df = pd.DataFrame(positive_word_count.items(), columns=['Word', 'Count']).sort_values(by='Count', ascending=False).head(10)
                        st.table(positive_word_count_df)

                    with pos_neg_word_col2:
                        st.subheader('💬 부정 뉴스에서 가장 많이 발생한 단어')
                        ##################################################################################################################################
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
                        ##################################################################################################################################
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

# 메인 함수
def main_dashboard():
    query_params = st.query_params
    page = query_params.get('page', 'main')

    if page == 'main':
        st.markdown(
            """
            <style>
            @import url('https://fonts.googleapis.com/css?family=Lato:100,300,400');
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            .button-container-2 {
                position: relative;
                width: 48%;
                height: 100px;
                margin: 2%;
                overflow: hidden;
                border: 1px solid #000;
                font-family: 'Lato', sans-serif;
                font-weight: 300;
                transition: 0.5s;
                letter-spacing: 1px;
                border-radius: 8px;
            }

            .button-container-2 button {
                width: 100%;
                height: 100%;
                font-family: 'Lato', sans-serif;
                font-weight: 300;
                font-size: 20px;
                letter-spacing: 1px;
                font-weight: bold;
                background: #000;
                -webkit-mask: url('https://raw.githubusercontent.com/robin-dela/css-mask-animation/master/img/urban-sprite.png');
                mask: url('https://raw.githubusercontent.com/robin-dela/css-mask-animation/master/img/urban-sprite.png');
                -webkit-mask-size: 3000% 100%;
                mask-size: 3000% 100%;
                border: none;
                color: #fff;
                cursor: pointer;
                -webkit-animation: ani2 0.7s steps(29) forwards;
                animation: ani2 0.7s steps(29) forwards;
            }

            .button-container-2 button:hover {
                -webkit-animation: ani 0.7s steps(29) forwards;
                animation: ani 0.7s steps(29) forwards;
            }

            .mas {
                position: absolute;
                color: #000;
                text-align: center;
                width: 100%;
                font-family: 'Lato', sans-serif;
                font-weight: 300;
                font-size: 20px;
                margin-top: 30px;
                overflow: hidden;
                font-weight: bold;
            }

            @-webkit-keyframes ani {
                from {
                    -webkit-mask-position: 0 0;
                    mask-position: 0 0;
                }
                to {
                    -webkit-mask-position: 100% 0;
                    mask-position: 100% 0;
                }
            }

            @keyframes ani {
                from {
                    -webkit-mask-position: 0 0;
                    mask-position: 0 0;
                }
                to {
                    -webkit-mask-position: 100% 0;
                    mask-position: 100% 0;
                }
            }

            @-webkit-keyframes ani2 {
                from {
                    -webkit-mask-position: 100% 0;
                    mask-position: 100% 0;
                }
                to {
                    -webkit-mask-position: 0 0;
                    mask-position: 0 0;
                }
            }

            @keyframes ani2 {
                from {
                    -webkit-mask-position: 100% 0;
                    mask-position: 100% 0;
                }
                to {
                    -webkit-mask-position: 0 0;
                    mask-position: 0 0;
                }
            }
            </style>
            <div style="display: flex; justify-content: space-between; align-items: center; width: 100%;">
                <div class="button-container-2">
                    <a href="?page=daily_news">
                    <span class="mas">데일리 뉴스 리포트</span>
                    <button type="button" onclick="location.href='?page=daily_news'">데일리 뉴스 리포트</button>
                    </a>
                </div>
                <div class="button-container-2">
                    <a href="?page=stock_prediction">
                    <span class="mas">주가 예측 대시보드</span>
                    <button type="button" onclick="location.href='?page=stock_prediction'">주가 예측 대시보드</button>
                    </a>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    if page == 'daily_news':
        st.markdown("<style>.button-container-2 { display: none; }</style>", unsafe_allow_html=True)
        st.session_state.logged_in = True
        daily_news_dashboard()
        st.sidebar.button("🏠 홈으로 돌아가기", on_click=lambda: st.query_params.update(page='main'))
        st.session_state.logged_in = True
    elif page == 'stock_prediction':
        st.markdown("<style>.button-container-2 { display: none; }</style>", unsafe_allow_html=True)
        stock_prediction_dashboard()
        st.sidebar.button("🏠 홈으로 돌아가기", on_click=lambda: st.query_params.update(page='main'))

import re

def login():
    st.set_page_config(layout='wide', page_title='로그인', page_icon='\U0001F512')

    # 로그인 입력 필드 및 스타일
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css?family=Lato:100,300,400');
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            background: #050801;
            font-family: 'Lato', sans-serif;
            font-weight: bold;
            color: #ffffff;
        }
        .login-container {
            text-align: center;
            width: 300px;
            padding: 40px;
            background: #333;
            border-radius: 10px;
            box-shadow: 0 0 20px #000;
        }
        input[type="text"], input[type="password"] {
            width: 100%;
            padding: 10px;
            margin: 10px 0;
            border-radius: 5px;
            border: none;
            font-size: 16px;
        }
        button {
            padding: 10px 20px;
            margin-top: 20px;
            background: #03e9f4;
            color: #050801;
            font-size: 16px;
            font-weight: bold;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            transition: 0.3s;
        }
        button:hover {
            background: #0298b9;
        }
        </style>
        """, unsafe_allow_html=True
    )

    # 로그인 폼 생성
    with st.form("login"):
        input_username = st.text_input("아이디를 입력하세요", key="input_username")
        input_password = st.text_input("비밀번호를 입력하세요", type="password", key="input_password")
        login_btn = st.form_submit_button("로그인")

    if login_btn:
        if input_username and input_password:
            if check_user(input_username, input_password):
                # 로그인 성공 시 세션 상태 업데이트 및 페이지 전환
                st.session_state.username = input_username
                st.session_state.logged_in = True
                st.rerun()
            else:
                # 로그인 실패
                st.error("아이디 또는 비밀번호가 틀렸습니다.")
        else:
            st.error("아이디와 비밀번호를 입력해주세요.")

    # 회원가입 버튼 추가
    if st.button("회원가입"):
        st.session_state.page = "register"
        st.rerun()


def register_user():
    st.subheader("회원가입")
    with st.form('register'):
        register_username = st.text_input("아이디를 입력하세요", key="register_username")
        register_password = st.text_input("비밀번호를 입력하세요", type="password", key="register_password")
        register_email = st.text_input("이메일을 입력하세요", key="register_email")
        email_req = st.checkbox("메일 수신 여부", key="email_req")

        submit = st.form_submit_button("회원가입하기")

    if submit:
        if not register_username or not register_password or not register_email:
            st.error("모든 필드를 입력해주세요.")
            return

        # 아이디 중복 체크
        if user_exists(register_username):
            st.error("이미 사용 중인 아이디입니다.")
            return

        # 비밀번호 유효성 검사 (5자 이상, 영문과 숫자가 모두 포함)
        if len(register_password) < 5 or not re.search("[a-zA-Z]", register_password) or not re.search("[0-9]", register_password):
            st.error("비밀번호는 5자 이상이며, 영문과 숫자가 모두 포함되어야 합니다.")
            return

        # 이메일 형식 유효성 검사
        if not re.match(r"[^@\s]+@[^@\s]+\.[^@\s]+", register_email):
            st.error("올바른 이메일 형식을 입력해주세요.")
            return

        # 회원 정보 저장 (예: 데이터베이스에 추가)
        if save_user(register_username, register_password, register_email, email_req):
            st.success("회원가입이 완료되었습니다. 이제 로그인해 주세요.")
            st.session_state.page = "login"
            st.rerun()


def check_user(input_username, input_password):
    query = text("SELECT COUNT(*) FROM user_table WHERE id = :username AND password = :password")

    with engine.connect() as conn:
        result = conn.execute(query, {"username": input_username, "password": input_password}).scalar()
        return result > 0  # Returns True if the user exists, otherwise False


def user_exists(username):
    query = text("SELECT COUNT(*) FROM user_table WHERE id = :username")

    with engine.connect() as conn:
        result = conn.execute(query, {"username": username}).scalar()
        return result > 0


def save_user(username, password, email, email_req):
    email_req = 1 if email_req else 0
    try:
        with engine.connect() as conn:
            query = text("INSERT INTO user_table(id, password, email, email_req) VALUES(:username, :password, :email, :email_req)")
            conn.execute(query, {"username": username,
                                 "password": password,
                                 "email": email,
                                 "email_req": email_req})
            conn.commit()
        return True
    except Exception as e:
        st.error("회원가입에 실패하였습니다. 입력정보를 다시 확인해주세요.")
        print(e)
        return False


# Streamlit 실행
if __name__ == '__main__':
    engine = create_engine(db_url)

    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
        st.session_state.username = None
        st.session_state.page = "login"

    if st.session_state.page == "login":
        if not st.session_state.logged_in:
            login()
        else:
            st.success(f"안녕하세요, {st.session_state.username}님!")
            # main_dashboard() 함수 호출 (메인 대시보드 화면)
            main_dashboard()

    elif st.session_state.page == "register":
        register_user()