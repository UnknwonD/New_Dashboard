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
from sqlalchemy import create_engine
from api import db_url
from gensim.models import Word2Vec

@st.cache_data
def create_wordcloud(text):
    # Use font_path to correctly display Korean text, words are horizontal only
    wordcloud = WordCloud(width=600, height=600, background_color='white', font_path='malgun.ttf', prefer_horizontal=1.0).generate(text)
    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    return fig

@st.cache_data
def train_word2vec_model(sentences):
    # Train a Word2Vec model
    model = Word2Vec(sentences, vector_size=100, window=5, min_count=2, workers=4)
    return model

def visualize_main_word_network(category, top_keywords, w2v_model):
    net = Network(notebook=False, height='600px', width='100%', bgcolor='#ffffff', font_color='black', layout=True)
    
    # Add main node for the category
    net.add_node(category, label=category, size=30, color='red', physics=False)
    
    # Add nodes for top keywords and connect them to the category node
    for keyword in top_keywords:
        net.add_node(keyword, label=keyword, size=20, color='lightblue', physics=False)
        net.add_edge(category, keyword, color='gray', physics=False)
    
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
    
    # 데이터 로드 및 날짜 형식 변환
    df = pd.read_sql(sql, engine)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    # content와 sentences의 '-' 값을 리스트로 변환 후 처리
    df['content'] = df['content'].replace('-', "['-']")
    df['sentences'] = df['sentences'].replace('-', "['-']")

    # content와 sentences를 리스트 타입으로 변환
    df['content'] = df['content'].apply(ast.literal_eval)
    df['sentences'] = df['sentences'].apply(ast.literal_eval)
    
    return df

# Sentiment analysis function
def analyze_sentiment(text):
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0:
        return '긍정'
    elif analysis.sentiment.polarity == 0:
        return '중립'
    else:
        return '부정'

def main():
    st.set_page_config(layout='wide', page_title='데일리 뉴스 리포트 대시보드', page_icon='📊')
    st.title('데일리 뉴스 리포트 대시보드')

    # 날짜 선택 창
    st.sidebar.subheader("데이터 선택")
    selected_date = st.sidebar.date_input('날짜 선택', pd.Timestamp('today'))

    if selected_date:
        df = data_load(selected_date)
        
        if df.empty:
            st.warning("선택한 날짜에 해당하는 데이터가 없습니다.")
        else:
            # 메인 화면에 데일리 리포트 표시
            st.header(f"{selected_date.strftime('%Y년 %m월 %d일')} 데일리 뉴스 리포트")

            # 탭을 이용한 메인 및 카테고리 선택
            tab_labels = ['메인', '정치', '경제', '사회', '생활/문화', 'IT/과학']
            tabs = st.tabs(tab_labels)

            # 메인 탭
            with tabs[0]:
                st.header('전체 뉴스 요약 정보')
                # 주요 뉴스
                st.subheader('주요 뉴스')
                for i, row in df.iterrows():
                    st.markdown(f"{i+1}. [{row['title']}]({row['url']}) / {row['category']} 뉴스")
                
                # 분야별 금일 뉴스 개수 정리
                st.subheader('분야별 뉴스 개수')
                news_count_by_category = df['category'].value_counts()
                news_count_df = pd.DataFrame({'Category': news_count_by_category.index, 'Count': news_count_by_category.values})
                category_chart = alt.Chart(news_count_df).mark_bar(color='steelblue').encode(
                    x=alt.X('Category', sort='-y', axis=alt.Axis(labelAngle=-45)),
                    y='Count'
                )
                try:
                    st.altair_chart(category_chart, use_container_width=True)
                except ValueError as e:
                    st.error(f"차트 생성 중 오류가 발생했습니다: {e}")
                
                # 분야별 성향 비율 시각화
                st.subheader('분야별 성향 분석')
                df['sentiment'] = df['content'].apply(lambda x: analyze_sentiment(' '.join(x)))
                sentiment_by_category = df.groupby('category')['sentiment'].value_counts().unstack().fillna(0)
                
                # 각 카테고리에 대해 파이차트 그리기
                for category in sentiment_by_category.index:
                    st.markdown(f"**{category}** 뉴스 성향 분석")
                    category_sentiments = sentiment_by_category.loc[category]
                    sentiment_df = pd.DataFrame({'Sentiment': category_sentiments.index, 'Count': category_sentiments.values})
                    pie_chart = alt.Chart(sentiment_df).mark_arc(innerRadius=50).encode(
                        theta=alt.Theta('Count', stack=True),
                        color=alt.Color('Sentiment', scale=alt.Scale(scheme='category10')),
                        tooltip=['Sentiment', 'Count']
                    ).properties(width=300, height=300)
                    try:
                        st.altair_chart(pie_chart, use_container_width=True)
                    except ValueError as e:
                        st.error(f"파이차트 생성 중 오류가 발생했습니다: {e}")

            # 카테고리별 탭
            for idx, selected_category in enumerate(tab_labels[1:]):
                with tabs[idx + 1]:
                    # 선택한 날짜 및 카테고리 필터링
                    filtered_data = df[df['category'] == selected_category]
                    if filtered_data.empty:
                        st.warning('해당 카테고리에 대한 데이터가 없습니다.')
                        continue

                    # Train Word2Vec model
                    sentences = [sentence for sublist in filtered_data['sentences'] for sentence in sublist]
                    w2v_model = train_word2vec_model(sentences)

                    # 컨테이너 사용하여 시각화
                    with st.container():
                        st.markdown("---")
                        col1, col2 = st.columns([1, 1], gap="large")
                        # 가장 많이 발생한 단어 시각화 및 워드 클라우드 생성
                        with col1:
                            st.subheader('가장 많이 발생한 단어 | 워드 클라우드')
                            tokens = []
                            kiwi = Kiwi()
                            for sublist in filtered_data['sentences']:
                                for sentence in sublist:
                                    for word in sentence:
                                        analyzed = kiwi.analyze(word)

                                        if analyzed:
                                            morphs = analyzed[0][0]
                                            for token in morphs:
                                                if token.tag.startswith('N') and len(token.form) > 1:
                                                    tokens.append(token.form)
                            all_text = ' '.join(tokens)
                            wordcloud_fig = create_wordcloud(all_text)
                            st.pyplot(wordcloud_fig)
                            
                            # 워드 카운트 시각화
                            st.subheader('가장 많이 발생한 단어')
                            word_count = Counter(tokens)
                            word_count_df = pd.DataFrame(word_count.items(), columns=['Word', 'Count']).sort_values(by='Count', ascending=False).head(10)
                            # 이쁘게 꾸민 표로 단어 출력
                            st.markdown("<style>table {width: 100%; text-align: left;} th, td {padding: 8px; text-align: left; border-bottom: 1px solid #ddd;} tr:hover {background-color: #f5f5f5;}</style>", unsafe_allow_html=True)
                            st.markdown(word_count_df.to_html(index=False), unsafe_allow_html=True)

                        # 관련 버튼 및 워드 네트워크 시각화
                        with col2:
                            st.subheader('관련 버튼 | 워드 네트워크')
                            # 단어 클릭 시 네트워크 시각화
                            for word in word_count_df['Word']:
                                with st.expander(f"{word} 유사 단어 네트워크 보기"):
                                    expanded_network_html = visualize_expanded_word_network(word.replace('/', '_'), w2v_model)
                                    components.html(expanded_network_html, height=750)

                            # 워드 네트워크 시각화
                            st.subheader('카테고리 중심 워드 네트워크')
                            if len(tokens) > 1:
                                top_keywords = word_count_df['Word'].tolist()
                                word_network_html = visualize_main_word_network(selected_category.replace('/', '_'), top_keywords, w2v_model)
                                components.html(word_network_html, height=750)
                            else:
                                st.warning('워드 네트워크를 생성하기에 충분한 데이터가 없습니다.')

                    # 주요 키워드 시각화
                    st.markdown("---")
                    st.subheader('주요 키워드')
                    top_keywords = word_count_df['Word'].tolist()
                    st.write(f"주요 키워드: {', '.join(top_keywords)}")

                    # 뉴스 요약 출력
                    st.subheader('뉴스 요약 (주요 키워드 관련 기사)')
                    for keyword in top_keywords:
                        st.write(f"### 키워드: {keyword}")
                        keyword_articles = filtered_data[filtered_data['content'].apply(lambda x: any(keyword in sentence for sentence in x))].head(5)
                        for i, row in keyword_articles.iterrows():
                            st.markdown(f"<div style='padding: 10px; border: 1px solid #ddd; border-radius: 5px; margin-bottom: 5px;'> <a href='{row['url']}' target='_blank' style='text-decoration: none; color: #2a9d8f;'> <strong>{row['title']}</strong></a></div>", unsafe_allow_html=True)

                    # 긍정, 부정 평가 시각화
                    st.subheader('긍정, 부정 평가')
                    sentiments = filtered_data['content'].apply(lambda x: analyze_sentiment(' '.join(x)))
                    sentiment_counts = sentiments.value_counts().to_dict()
                    sentiment_df = pd.DataFrame(list(sentiment_counts.items()), columns=['Sentiment', 'Count'])
                    pie_chart = alt.Chart(sentiment_df).mark_arc(innerRadius=50).encode(
                        theta=alt.Theta('Count', stack=True),
                        color=alt.Color('Sentiment', scale=alt.Scale(scheme='category10'))
                    )
                    try:
                        st.altair_chart(pie_chart, use_container_width=True)
                    except ValueError as e:
                        st.error(f"파이차트 생성 중 오류가 발생했습니다: {e}")


if __name__ == "__main__":
    main()
