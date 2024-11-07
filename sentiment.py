from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

# 토크나이저와 모델 로드
tokenizer = AutoTokenizer.from_pretrained("WhitePeak/bert-base-cased-Korean-sentiment")
model = AutoModelForSequenceClassification.from_pretrained("WhitePeak/bert-base-cased-Korean-sentiment")

# 파이프라인 생성
sentiment_model = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# 감정 분석 실행
result = sentiment_model("짜증나 죽겠어")
print(result)