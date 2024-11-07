from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

# 토크나이저와 모델 로드
tokenizer = AutoTokenizer.from_pretrained("WhitePeak/bert-base-cased-Korean-sentiment")
model = AutoModelForSequenceClassification.from_pretrained("WhitePeak/bert-base-cased-Korean-sentiment")

# 파이프라인 생성
sentiment_model = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

def analyze_sentiment_detailed(text):
    # 문장 단위로 분리
    sentences = text.split('.')
    sentences = [s.strip() for s in sentences if s.strip()]
    
    scores = []
    sentiments = []
    for sentence in sentences:
        result = sentiment_model(sentence)[0]
        score = result['score']
        label = result['label']
        
        # 중립 범위 정의 (예: 0.4 ~ 0.6)
        if 0.4 <= score <= 0.6:
            sentiment = "중립"
            adjusted_score = 50
        elif label == 'LABEL_1':
            sentiment = "긍정"
            adjusted_score = int(score * 50 + 50)  # 50-100 범위로 조정
        else:  # LABEL_0
            sentiment = "부정"
            adjusted_score = int((1 - score) * 50)  # 0-50 범위로 조정
        
        scores.append(adjusted_score)
        sentiments.append(sentiment)
    
    # 전체 텍스트의 평균 점수 계산
    average_score = int(np.mean(scores))
    
    # 결과 출력
    print(f"전체 텍스트 감정 점수: {average_score}/100")
    print("\n개별 문장 분석:")
    for i, (sentence, score, sentiment) in enumerate(zip(sentences, scores, sentiments), 1):
        print(f"{i}. '{sentence}': {score}/100 ({sentiment})")
    
    return average_score

# 테스트
long_text = "오늘 발표를 잘했다고 교수님께 칭찬을 받았다. 하지만 교수님께서 PPT 발표자료는 잘 만들었다고 하였으나 사족이 너무 길다고 피드백해 주셨다. 나도 동감한다. 사족이 긴 부분을 더 논리적이고 단순하고 확실하게 발표할 수 있도록 발표 연습을 할 예정이다. 교수님께서 내가 어디를 가나 있다고 정말 열심히 산다고 말씀해 주셔서 더욱 용기가났다. 이런 열심히 라는 수식어를 붙여주는 게 왜 이리 기분이 좋은지 모르겠다. 어제 밤새서 공모전 보고서도 제출하고 오늘 발표도 잘 마무리하여서 정말 괜찮은 하루를 보낸 것 같다."

analyze_sentiment_detailed(long_text)