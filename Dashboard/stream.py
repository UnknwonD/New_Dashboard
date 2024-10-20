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
from summarizer import Summarizer
import streamlit.components.v1 as components

@st.cache_data
def create_wordcloud(text):
    # Use font_path to correctly display Korean text
    wordcloud = WordCloud(width=600, height=600, background_color='white', font_path='malgun.ttf').generate(text)
    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    return fig

def visualize_word_network(words):
    net = Network(notebook=False, height='600px', width='100%', bgcolor='#ffffff', font_color='black', layout=True)
    word_counts = Counter(words)
    for word, count in word_counts.items():
        net.add_node(word, label=word, size=count*10, color='lightblue', physics=False)
    for i in range(len(words)-1):
        net.add_edge(words[i], words[i+1], color='gray', physics=False)
    
    # Ensure the output directory exists
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    output_path = os.path.join(output_dir, 'word_network.html')
    net.write_html(output_path)
    with open(output_path, 'r', encoding='utf-8') as HtmlFile:
        return HtmlFile.read()

@st.cache_data
def extractive_summarize_korean_text(text, compression_ratio=0.5):
    # BERT 기반 추출 요약 모델 생성
    model = Summarizer()
    summary = model(text, ratio=compression_ratio)
    return summary

# Load or create dataframe (example)
data = {'date': ['2024-10-18', '2024-10-18', '2024-10-17', '2024-10-18', '2024-10-18'],
        'category': ['경제', '정치', '사회', '생활/문화', 'IT/과학'],
        'content': ["경제 상황이 매우 어렵습니다", "정치적 논란이 계속되고 있습니다", "사회적 문제와 환경 문제", "문화 행사와 관련된 새로운 소식", "기술 발전과 혁신적인 뉴스"],
        'title': ["경제 위기", "정치 논란", "사회 문제", "문화 소식", "기술 혁신"]}
df = pd.DataFrame(data)

# Sentiment analysis function
def analyze_sentiment(text):
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0:
        return '긍정'
    elif analysis.sentiment.polarity == 0:
        return '중립'
    else:
        return '부정'

# Main function
def main():
    st.set_page_config(layout='wide', page_title='데일리 뉴스 리포트 대시보드', page_icon='📊')
    st.title('데일리 뉴스 리포트 대시보드')

    # 날짜 선택
    selected_date = st.date_input('날짜 선택', pd.Timestamp('today'))

    # 탭을 이용한 메인 및 카테고리 선택
    tab_labels = ['메인', '정치', '경제', '사회', '생활/문화', 'IT/과학']
    tabs = st.tabs(tab_labels)

    # 메인 탭
    with tabs[0]:
        st.header('전체 뉴스 요약 정보')
        col1, col2, col3 = st.columns([1, 1, 1], gap="large")
        # 주요 뉴스
        with col1:
            st.subheader('주요 뉴스')
            for i, row in df.iterrows():
                if row['date'] == str(selected_date):
                    st.markdown(f"{i+1}. [{row['title']}]({row['content']}) / {row['category']} 뉴스")
        # 분야별 금일 뉴스 개수 정리
        with col2:
            st.subheader('분야별 뉴스 개수')
            filtered_df = df[df['date'] == str(selected_date)].copy()
            news_count_by_category = filtered_df['category'].value_counts()
            news_count_df = pd.DataFrame({'Category': news_count_by_category.index, 'Count': news_count_by_category.values})
            category_chart = alt.Chart(news_count_df).mark_bar(color='steelblue').encode(
                x=alt.X('Category', sort='-y', axis=alt.Axis(labelAngle=-45)),
                y='Count'
            )
            st.altair_chart(category_chart, use_container_width=True)
        # 분야별 성향 비율 시각화
        with col3:
            st.subheader('분야별 성향 비율')
            filtered_df['sentiment'] = filtered_df['content'].apply(analyze_sentiment)
            sentiment_by_category = filtered_df.groupby('category')['sentiment'].value_counts(normalize=True).unstack().fillna(0)
            sentiment_by_category = sentiment_by_category.reset_index().melt(id_vars='category', var_name='Sentiment', value_name='Ratio')
            sentiment_chart = alt.Chart(sentiment_by_category).mark_bar().encode(
                x=alt.X('category:N', title='Category'),
                y=alt.Y('Ratio:Q', title='Sentiment Ratio', axis=alt.Axis(format='%')), 
                color=alt.Color('Sentiment:N', scale=alt.Scale(scheme='category10')),
                column='Sentiment:N'
            )
            st.altair_chart(sentiment_chart, use_container_width=True)

    # 카테고리별 탭
    for idx, selected_category in enumerate(tab_labels[1:]):
        with tabs[idx + 1]:
            # 선택한 날짜 및 카테고리 필터링
            filtered_data = df[(df['category'] == selected_category) & (df['date'] == str(selected_date))]
            if filtered_data.empty:
                st.warning('해당 카테고리에 대한 데이터가 없습니다.')
                continue

            # 뉴스 요약 생성
            all_text = ' '.join(filtered_data['content'])
            summary = extractive_summarize_korean_text(all_text)
            st.header(f"{selected_category} 뉴스 리포트")
            st.subheader('뉴스 요약')
            st.write(summary)

            # 컨테이너 사용하여 시각화
            with st.container():
                st.markdown("---")
                col1, col2, col3 = st.columns([1, 1, 2], gap="large")
                # 워드 클라우드 생성
                with col1:
                    st.subheader('워드 클라우드')
                    wordcloud_fig = create_wordcloud(all_text)
                    st.pyplot(wordcloud_fig)
                # 워드 카운트 시각화
                with col2:
                    st.subheader('가장 많이 발생한 단어')
                    kiwi = Kiwi()
                    tokens = [word.form for word in kiwi.tokenize(all_text) if word.tag.startswith('NN')]
                    word_count = Counter(tokens)
                    word_count_df = pd.DataFrame(word_count.items(), columns=['Word', 'Count']).sort_values(by='Count', ascending=False)
                    bar_chart = alt.Chart(word_count_df).mark_bar(color='lightgreen').encode(
                        x=alt.X('Word', sort='-y', axis=alt.Axis(labelAngle=-45)),
                        y='Count'
                    )
                    st.altair_chart(bar_chart, use_container_width=True)
                # 워드 네트워크 시각화
                with col3:
                    st.subheader('워드 네트워크')
                    if len(tokens) > 1:
                        word_network_html = visualize_word_network(tokens)
                        components.html(word_network_html, height=750)
                    else:
                        st.warning('워드 네트워크를 생성하기에 충분한 데이터가 없습니다.')

            # 주요 키워드 시각화
            st.markdown("---")
            st.subheader('주요 키워드')
            keywords = word_count_df.head(5)['Word'].tolist()
            st.write(f"주요 키워드: {', '.join(keywords)}")

            # 긍정, 부정 평가 시각화
            st.subheader('긍정, 부정 평가')
            sentiments = filtered_data['content'].apply(analyze_sentiment)
            sentiment_counts = sentiments.value_counts().to_dict()
            sentiment_df = pd.DataFrame(list(sentiment_counts.items()), columns=['Sentiment', 'Count'])
            pie_chart = alt.Chart(sentiment_df).mark_arc(innerRadius=50).encode(
                theta=alt.Theta('Count', stack=True),
                color=alt.Color('Sentiment', scale=alt.Scale(scheme='category10'))
            )
            st.altair_chart(pie_chart, use_container_width=True)

if __name__ == "__main__":
    main()
