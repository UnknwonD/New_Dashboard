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
from jinja2 import Template

@st.cache_data
def create_wordcloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color='white', font_path='malgun.ttf').generate(text)
    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    return fig

def visualize_word_network(words):
    net = Network(notebook=False)
    word_counts = Counter(words)
    for word, count in word_counts.items():
        net.add_node(word, label=word, size=count*10)
    for i in range(len(words)-1):
        net.add_edge(words[i], words[i+1])
    
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    output_path = os.path.join(output_dir, 'word_network.html')
    net.write_html(output_path)
    with open(output_path, 'r', encoding='utf-8') as HtmlFile:
        return HtmlFile.read()

data = {'date': ['2024-10-18', '2024-10-18', '2024-10-17'],
        'category': ['경제', '정치', '사회'],
        'content': ["경제 상황이 매우 어렵습니다", "정치적 논란이 계속되고 있습니다", "사회적 문제와 환경 문제"]}
df = pd.DataFrame(data)

def analyze_sentiment(text):
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0:
        return '긍정'
    elif analysis.sentiment.polarity == 0:
        return '중립'
    else:
        return '부정'

def main():
    st.title('데일리 뉴스 리포트 대시보드')
    st.sidebar.header('옵션')

    selected_date = st.sidebar.date_input('날짜 선택', pd.Timestamp('today'))
    selected_category = st.sidebar.selectbox('카테고리 선택', options=['정치', '경제', '사회', '생활/문화', 'IT/과학'])

    filtered_data = df[(df['date'] == str(selected_date)) & (df['category'] == selected_category)]
    if filtered_data.empty:
        st.warning('해당 날짜와 카테고리에 대한 데이터가 없습니다.')
        return

    st.header(f"{selected_date} {selected_category} 뉴스 리포트")
    st.write(filtered_data)

    # 워드 클라우드 생성
    all_text = ' '.join(filtered_data['content'])
    st.subheader('워드 클라우드')
    wordcloud_fig = create_wordcloud(all_text)
    st.pyplot(wordcloud_fig)

    # 워드 카운트 시각화
    st.subheader('가장 많이 발생한 단어')
    kiwi = Kiwi()
    tokens = [word.form for word in kiwi.tokenize(all_text)]
    word_count = Counter(tokens)
    word_count_df = pd.DataFrame(word_count.items(), columns=['Word', 'Count']).sort_values(by='Count', ascending=False)
    bar_chart = alt.Chart(word_count_df).mark_bar().encode(
        x=alt.X('Word', sort='-y'),
        y='Count'
    )
    st.altair_chart(bar_chart, use_container_width=True)

    # 워드 네트워크 시각화
    st.subheader('워드 네트워크')
    word_network_html = visualize_word_network(tokens)
    st.components.v1.html(word_network_html, height=500)

    # 주요 키워드 시각화
    st.subheader('주요 키워드')
    keywords = word_count_df.head(5)['Word'].tolist()
    st.write(f"주요 키워드: {', '.join(keywords)}")

    # 긍정, 부정 평가 시각화
    st.subheader('긍정, 부정 평가')
    sentiments = filtered_data['content'].apply(analyze_sentiment)
    sentiment_counts = sentiments.value_counts().to_dict()
    sentiment_df = pd.DataFrame(list(sentiment_counts.items()), columns=['Sentiment', 'Count'])
    pie_chart = alt.Chart(sentiment_df).mark_arc().encode(
        theta=alt.Theta('Count', stack=True),
        color='Sentiment'
    )
    st.altair_chart(pie_chart, use_container_width=True)

    # 두 번째 Tab: 관련 주식 정보
    tab1, tab2 = st.tabs(["뉴스 분석", "관련 주식 정보"])
    with tab2:
        st.subheader("관련 주식 정보")
        st.write("관련 주식 데이터 및 시각화를 여기에 추가하세요.")

if __name__ == "__main__":
    main()
