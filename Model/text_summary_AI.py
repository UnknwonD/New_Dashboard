from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

def summarize_text(text, max_length=150, min_length=30):
    # T5 모델과 토크나이저 불러오기
    model_name = "t5-small"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    # 입력 텍스트를 요약 작업에 맞게 인코딩
    input_text = "summarize: " + text
    input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)

    # 모델을 사용하여 요약 생성
    summary_ids = model.generate(input_ids, max_length=max_length, min_length=min_length, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return summary

if __name__ == "__main__":
    # 긴 텍스트를 입력받기
    text = """
    인공지능(AI)은 컴퓨터 시스템이 인간처럼 사고하고 학습하며 문제를 해결할 수 있도록 설계된 기술입니다. 이는 다양한 기계 학습 기법과 딥러닝 알고리즘을 통해 발전하고 있으며, 오늘날 여러 산업 분야에서 중요한 역할을 하고 있습니다. 예를 들어, 의료 산업에서는 AI가 질병을 진단하고 새로운 치료법을 개발하는 데 도움을 주며, 자율주행차에서는 안전하고 효율적인 운전을 지원합니다. 또한, 고객 서비스, 금융 분석, 교육 등 여러 분야에서도 AI 기술이 사용되고 있습니다.
    """

    # 요약 수행
    summary = summarize_text(text)
    print("Original Text:\n", text)
    print("\nSummary:\n", summary)