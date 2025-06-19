# app.py
from flask import Flask, render_template, request
import finnhub
import pandas as pd
import re
from transformers import pipeline
from datetime import datetime, timedelta

app = Flask(__name__)

# ——— 템플릿 필터 정의 ———
@app.template_filter('timestamp_to_date')
def timestamp_to_date(timestamp):
    """타임스탬프를 날짜 문자열로 변환"""
    return datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d')

# ——— 1) Finnhub 클라이언트 & FinBERT 초기화 ———
API_KEY = "d1997v9r01qkcat6rsu0d1997v9r01qkcat6rsug"
finnhub_client = finnhub.Client(api_key=API_KEY)
finbert = pipeline(
    "sentiment-analysis",
    model="ProsusAI/finbert",
    tokenizer="ProsusAI/finbert"
)

# ——— 2) 유틸 함수 정의 ———
def get_ticker(company_name: str) -> str | None:
    res = finnhub_client.symbol_lookup(company_name)
    results = res.get('result', [])
    return results[0]['symbol'] if results else None

def clean_text(text: str) -> str:
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip().lower()

def analyze_sentiment_finbert(text: str) -> dict:
    out = finbert(text[:512])[0]
    return {'label': out['label'].lower(), 'score': out['score']}

def generate_advice(df, pos_thresh=0.6, neg_thresh=0.6, neu_thresh=0.6) -> str:
    total = len(df)
    pos = (df['label']=='positive').sum()/total
    neg = (df['label']=='negative').sum()/total
    neu = (df['label']=='neutral').sum()/total

    if pos >= pos_thresh:
        return f"긍정 {pos:.1%} ≥ {pos_thresh:.1%} → BUY"
    elif neg >= neg_thresh:
        return f"부정 {neg:.1%} ≥ {neg_thresh:.1%} → SELL"
    elif neu >= neu_thresh:
        return f"중립 {neu:.1%} ≥ {neu_thresh:.1%} → HOLD"
    return f"긍정 {pos:.1%}, 부정 {neg:.1%}, 중립 {neu:.1%} → HOLD"

# ——— 3) 라우트 ———
@app.route("/", methods=["GET", "POST"])
def index():
    advice = None
    chart_data = None
    news_items = None

    # GET 혹은 POST에서 company 파라미터 받기
    if request.method == "POST":
        company = request.form.get("company")
    else:
        company = request.args.get("company")

    if company:
        ticker = get_ticker(company)
        if not ticker:
            advice = f"{company}에 대한 티커를 찾을 수 없습니다."
        else:
            # ─ 뉴스 수집
            today = datetime.today().date()
            news = finnhub_client.company_news(
                ticker,
                _from=(today - timedelta(days=30)).isoformat(),
                to=today.isoformat()
            )
            df = pd.DataFrame(news).sort_values("datetime", ascending=False).head(20)

            # ─ 전처리 + FinBERT 분석
            df['clean'] = df['headline'].astype(str).apply(clean_text)
            sent = df['clean'].apply(analyze_sentiment_finbert).apply(pd.Series)
            df = pd.concat([df, sent], axis=1)

            # ─ 차트 데이터 & 뉴스 리스트
            chart_data = df['label'].value_counts().to_dict()
            news_items = df.to_dict(orient="records")

            # ─ 투자 조언
            advice = generate_advice(df)

    return render_template(
        "index.html",
        advice=advice,
        chart_data=chart_data,
        news_items=news_items
    )

if __name__ == "__main__":
    app.run(debug=True, port=5002)