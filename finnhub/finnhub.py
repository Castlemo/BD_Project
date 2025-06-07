#!/usr/bin/env python3
import requests
import pandas as pd
import datetime
import os
from dotenv import load_dotenv
import yfinance as yf

# .env 설정 로드
load_dotenv()
FINHUB_API_KEY = os.getenv("finhub")  # 변수명은 'finnhub' 아닌 'finhub'


# Finnhub에서 AAPL 보다 개정된 뉴스 수집

def fetch_finnhub_news(symbol: str = "AAPL") -> pd.DataFrame:
    url = f"https://finnhub.io/api/v1/company-news"
    today = datetime.date.today()
    from_date = today - datetime.timedelta(days=365)

    params = {
        "symbol": symbol,
        "from": from_date.isoformat(),
        "to": today.isoformat(),
        "token": FINHUB_API_KEY
    }

    try:
        res = requests.get(url, params=params)
        if res.status_code != 200:
            print(f"[ERROR] Finnhub API 요청 실패: {res.status_code}")
            print(res.text)
            return pd.DataFrame()

        data = res.json()
        articles = []
        for item in data:
            articles.append({
                "title": item.get("headline"),
                "summary": item.get("summary"),
                "link": item.get("url"),
                "publisher": item.get("source"),
                "pubDate": pd.to_datetime(item.get("datetime"), unit='s'),
                "source": "finnhub"
            })

        return pd.DataFrame(articles)

    except Exception as e:
        print(f"[ERROR] 요청 예외 발생: {e}")
        return pd.DataFrame()


# yfinance 주가 데이터

def fetch_stock_data(ticker_symbol: str, period: str = '1y', interval: str = '1d') -> pd.DataFrame:
    try:
        df = yf.download(
            tickers=ticker_symbol,
            period=period,
            interval=interval,
            auto_adjust=False,
            progress=False
        )
        if df.empty:
            return pd.DataFrame()
        df = df.reset_index()
        df = df.rename(columns={'Adj Close': 'Adj_Close'})
        return df[['Date', 'Open', 'High', 'Low', 'Close', 'Adj_Close', 'Volume']]
    except Exception as e:
        print(f"[ERROR] 주식 데이터 오류: {e}")
        return pd.DataFrame()


if __name__ == "__main__":
    ticker = "AAPL"
    print(f"[INFO] Finnhub 뉴스 수집 중... ({ticker})")
    df_news = fetch_finnhub_news(ticker)

    if not df_news.empty:
        print(f"[SUCCESS] 뉴스 {len(df_news)}개 수집 완료")
        for i, row in df_news.head(3).iterrows():
            print(f"- {row['title']}")
    else:
        print("[ERROR] 뉴스 수집 실패")

    print(f"\n[INFO] {ticker} 주가 데이터 수집 중...")
    df_stock = fetch_stock_data(ticker)
    if not df_stock.empty:
        print(f"[SUCCESS] {len(df_stock)}일치 주가 데이터 수집 완료")
    else:
        print("[ERROR] 주가 데이터 수집 실패")

    today = datetime.date.today().isoformat()
    if not df_news.empty:
        df_news.to_csv(f"{ticker}_news_{today}.csv", index=False, encoding='utf-8-sig')
        print(f"[FILE] 뉴스 저장 완료: {ticker}_news_{today}.csv")
    if not df_stock.empty:
        df_stock.to_csv(f"{ticker}_stock_{today}.csv", index=False, encoding='utf-8-sig')
        print(f"[FILE] 주가 저장 완료: {ticker}_stock_{today}.csv")
