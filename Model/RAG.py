import psycopg2
import openai
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

# 데이터베이스 설정 및 pgvector 연결
def connect_db():
    return psycopg2.connect(
        host="192.168.219.104",
        port=5433,
        database="postgres",
        user="postgres",
        password="lee03260"
    )

# 데이터베이스 초기 설정 - pgvector 확장 설치 및 테이블 생성
def setup_database():
    conn = connect_db()
    cursor = conn.cursor()
    
    # pgvector 확장 설치
    cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    
    # 기존 테이블에 embedding 컬럼 추가
    cursor.execute("""
        ALTER TABLE social_table
        ADD COLUMN IF NOT EXISTS embedding VECTOR(768);
    """)
    
    conn.commit()
    cursor.close()
    conn.close()

# 텍스트 임베딩 함수
def embed_text(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors='pt')
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1).squeeze()
    return embeddings.numpy()

# 기존 데이터베이스에 임베딩 추가 함수
def add_embeddings_to_database(table_name, tokenizer, model):
    conn = connect_db()
    cursor = conn.cursor()
    
    cursor.execute(f"SELECT id, content FROM {table_name} WHERE embedding IS NULL;")
    rows = cursor.fetchall()
    
    for row in rows:
        record_id, content = row
        embedding = embed_text(content, tokenizer, model)
        embedding_str = np.array2string(embedding, separator=',')[1:-1]  # numpy 배열을 문자열로 변환하고 불필요한 괄호 제거
        
        cursor.execute(
            f"UPDATE {table_name} SET embedding = ARRAY[{embedding_str}]::vector WHERE id = %s;",
            (record_id,)
        )
    
    conn.commit()
    cursor.close()
    conn.close()

# 질문의 임베딩을 생성하여 pgvector에서 유사한 데이터 검색
def retrieve_data(query, conn, tokenizer, model):
    query_embedding = embed_text(query, tokenizer, model)
    query_vector = np.array2string(query_embedding, separator=',')[1:-1]  # numpy 배열을 문자열로 변환하고 불필요한 괄호 제거

    cursor = conn.cursor()
    sql = f"""
        SELECT content, embedding <=> ARRAY[{query_vector}]::vector AS distance
        FROM social_table
        ORDER BY distance ASC
        LIMIT 5;
    """
    cursor.execute(sql)
    results = cursor.fetchall()
    cursor.close()
    return [result[0] for result in results]

# GPT 모델을 사용한 답변 생성
def generate_answer(query, retrieved_docs):
    context = "\n".join(retrieved_docs)
    prompt = f"Context: {context}\n\nQuestion: {query}\nAnswer:"
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=200
    )
    return response.choices[0].text.strip()

# 전체 프로세스
def main():
    # 데이터베이스 초기 설정
    setup_database()
    
    # 임베딩 모델 설정
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

    # 기존 데이터 추가 (임베딩이 없는 경우에만 추가)
    add_embeddings_to_database('social_table', tokenizer, model)

    # 데이터베이스 연결
    conn = connect_db()
    
    # 사용자 쿼리 처리
    query = "최근 AI 트렌드에 대해 알려줘"
    retrieved_docs = retrieve_data(query, conn, tokenizer, model)
    answer = generate_answer(query, retrieved_docs)
    print(answer)
    
    # 데이터베이스 연결 종료
    conn.close()

if __name__ == "__main__":
    main()
