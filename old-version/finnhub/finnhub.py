#!/usr/bin/env python3
import requests
import pandas as pd
import datetime
import os
from dotenv import load_dotenv
import yfinance as yf
import time

# .env 설정 로드
load_dotenv()
FINHUB_API_KEY = os.getenv("finhub")  # 변수명은 'finnhub' 아닌 'finhub'

# API 호출 제한 (60 calls/minute)
API_CALLS_PER_MINUTE = 60
DELAY_BETWEEN_CALLS = 60.0 / API_CALLS_PER_MINUTE  # 1초 간격


def safe_datetime_conversion(timestamp):
    """
    타임스탬프를 안전하게 datetime으로 변환합니다.
    Out of bounds nanosecond timestamp 오류를 방지합니다.
    """
    if not timestamp or timestamp == 0:
        return None
    
    try:
        # 유닉스 타임스탬프 범위 확인 (1970-01-01 이후)
        if timestamp < 0:
            return None
            
        # 너무 큰 값 확인 (2262년 이후는 pandas에서 처리 불가)
        if timestamp > 9223372036:  # 2262-04-11 정도
            return None
            
        # 정상적인 변환 시도
        return pd.to_datetime(timestamp, unit='s')
        
    except (ValueError, OutOfBoundsDatetime, OverflowError):
        # 변환 실패 시 None 반환
        return None
    except Exception:
        # 기타 예외 시 None 반환
        return None


# Finnhub에서 회사 뉴스 수집 (최적화된 버전 - 대량 수집)
def fetch_finnhub_news_extended(symbol: str = "GOOGL", start_date: str = "2025-06-14", days_per_request: int = 30) -> pd.DataFrame:
    """
    하나의 심볼에 대해 최대한 많은 뉴스를 수집합니다.
    날짜 구간을 나누어서 여러 번 API 호출하여 더 많은 기사를 수집합니다.
    
    Args:
        symbol: 주식 심볼 (예: "AAPL")
        start_date: 시작 날짜 (YYYY-MM-DD)
        days_per_request: 한 번의 API 호출당 수집할 일수 (기본: 30일)
    """
    if not FINHUB_API_KEY:
        print("[ERROR] Finnhub API 키가 설정되지 않았습니다. .env 파일을 확인하세요.")
        return pd.DataFrame()

    url = "https://finnhub.io/api/v1/company-news"
    
    try:
        start_date_obj = datetime.datetime.strptime(start_date, "%Y-%m-%d").date()
        today = datetime.date.today()
        
        print(f"[INFO] 요청 시작 날짜: {start_date_obj}")
        print(f"[INFO] 현재 날짜: {today}")
        
        # Free Tier 제한을 넘어서 최대한 많이 수집 시도
        two_years_ago = today - datetime.timedelta(days=730)  # 2년 전
        three_years_ago = today - datetime.timedelta(days=1095)  # 3년 전
        
        print(f"[INFO] 🚀 Free Tier 제한 돌파 시도: 2-3년간 뉴스 수집을 시도합니다!")
        print(f"[INFO] 2년 전 날짜: {two_years_ago}")
        print(f"[INFO] 3년 전 날짜: {three_years_ago}")
        
        # 3년 전부터 현재까지로 시도 (API가 어디까지 허용하는지 테스트)
        actual_start = three_years_ago
        actual_end = today
        
        print(f"[INFO] {symbol} 뉴스 대량 수집 시작 (제한 돌파 시도)")
        print(f"[INFO] 실제 수집 기간: {actual_start.isoformat()} ~ {actual_end.isoformat()}")
        print(f"[INFO] 수집 기간: {(actual_end - actual_start).days}일 (약 3년)")
        print(f"[INFO] 예상 API 호출 횟수: {(actual_end - actual_start).days // days_per_request + 1}회")
        print(f"[WARNING] API가 제한할 수 있습니다. 테스트 중...")
        
    except ValueError:
        print(f"[ERROR] 잘못된 날짜 형식: {start_date}. YYYY-MM-DD 형식을 사용하세요.")
        return pd.DataFrame()

    all_articles = []
    current_date = actual_start
    request_count = 0
    
    while current_date < actual_end:
        # 각 요청의 종료 날짜 계산
        period_end = min(current_date + datetime.timedelta(days=days_per_request), actual_end)
        
        params = {
            "symbol": symbol,
            "from": current_date.isoformat(),
            "to": period_end.isoformat(),
            "token": FINHUB_API_KEY
        }
        
        try:
            request_count += 1
            print(f"[INFO] API 호출 {request_count}: {current_date.isoformat()} ~ {period_end.isoformat()}")
            
            # API 호출 제한을 위한 딜레이
            time.sleep(DELAY_BETWEEN_CALLS)
            
            res = requests.get(url, params=params, timeout=30)
            
            if res.status_code == 429:
                print(f"[WARNING] API 호출 제한 도달. 더 긴 대기 후 재시도...")
                time.sleep(10)
                res = requests.get(url, params=params, timeout=30)
            
            if res.status_code == 403:
                print(f"[ERROR] API 접근 거부 (기간: {current_date} ~ {period_end}): Free Tier 제한 도달 가능성")
                print(f"[INFO] 현재까지 수집된 기사: {len(all_articles)}개")
                break
            
            if res.status_code != 200:
                print(f"[WARNING] API 요청 실패 (기간: {current_date} ~ {period_end}): HTTP {res.status_code}")
                print(f"[INFO] 응답 내용: {res.text[:200]}...")
                
                # 429 (Too Many Requests)가 아니라면 계속 진행
                if res.status_code != 429:
                    current_date = period_end + datetime.timedelta(days=1)
                    continue
                else:
                    print(f"[INFO] 호출 제한으로 인한 대기...")
                    time.sleep(30)
                    continue

            data = res.json()
            
            if not isinstance(data, list):
                print(f"[WARNING] 예상과 다른 응답 형식 (기간: {current_date} ~ {period_end})")
                if isinstance(data, dict) and 'error' in data:
                    print(f"[ERROR] API 오류: {data['error']}")
                    if 'limit' in data['error'].lower():
                        print(f"[INFO] Free Tier 제한 도달. 현재까지 수집: {len(all_articles)}개")
                        break
                current_date = period_end + datetime.timedelta(days=1)
                continue

            # 해당 기간의 뉴스 수집
            period_articles = []
            for item in data:
                # 안전한 datetime 변환 사용
                pub_date = safe_datetime_conversion(item.get("datetime"))
                
                article = {
                    "id": item.get("id"),
                    "title": item.get("headline", ""),
                    "summary": item.get("summary", ""),
                    "link": item.get("url", ""),
                    "publisher": item.get("publisher", ""),
                    "category": item.get("category", ""),
                    "pubDate": pub_date,
                    "image": item.get("image", ""),
                    "related": item.get("related", ""),
                    "source": item.get("source", ""),
                    "collection_period": f"{current_date.isoformat()}_{period_end.isoformat()}"
                }
                period_articles.append(article)
            
            all_articles.extend(period_articles)
            print(f"[SUCCESS] 기간별 수집: {len(period_articles)}개 기사 (총 {len(all_articles)}개)")
            
        except Exception as e:
            print(f"[ERROR] API 호출 오류 (기간: {current_date} ~ {period_end}): {e}")
            if "limit" in str(e).lower() or "403" in str(e):
                print(f"[INFO] Free Tier 제한 도달 가능성. 현재까지 수집: {len(all_articles)}개")
                break
        
        # 다음 기간으로 이동
        current_date = period_end + datetime.timedelta(days=1)
        
        # API 호출 제한 방지를 위한 추가 대기
        if request_count % 10 == 0:  # 10번 호출마다 추가 대기
            print(f"[INFO] API 제한 방지를 위한 대기... (현재까지 {len(all_articles)}개 수집)")
            time.sleep(2)

    # 전체 결과 처리
    if all_articles:
        df = pd.DataFrame(all_articles)
        
        # 중복 제거 (ID, 제목, 링크 기준)
        before_dedup = len(df)
        df = df.drop_duplicates(subset=['id', 'title', 'link'])
        after_dedup = len(df)
        
        if before_dedup != after_dedup:
            print(f"[INFO] 중복 기사 제거: {before_dedup - after_dedup}개")
        
        # 날짜순 정렬 (최신순) - None 값 처리
        df = df.sort_values('pubDate', ascending=False, na_position='last')
        
        print(f"[SUCCESS] {symbol} 총 {len(df)}개의 뉴스 기사 수집 완료!")
        print(f"[INFO] 총 API 호출 횟수: {request_count}회")
        
        return df
    else:
        print(f"[WARNING] {symbol}에 대한 뉴스를 찾을 수 없습니다.")
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
    # 대량 뉴스 수집할 단일 심볼 설정
    target_symbol = "GOOGL"  # 원하는 심볼로 변경 가능 (예: MSFT, GOOGL, AMZN, TSLA, META, NVDA 등)
    start_date = "2025-06-13"  # 참고용 (실제로는 1년 전부터 자동 수집)
    days_per_request = 7  # 한 번의 API 호출당 7일씩 수집 (더 많은 API 호출로 최대 수집)
    
    print(f"🚀 {target_symbol} 뉴스 대량 수집 시작!")
    print(f"📅 참고 날짜: {start_date} (실제로는 3년 전부터 수집 시도)")
    print(f"📊 수집 방식: {days_per_request}일씩 구간별 수집")
    print(f"⏱️  API 제한: {API_CALLS_PER_MINUTE}회/분")
    print(f"🎯 목표: 🚀 Free Tier 제한 돌파 시도 - 최대 3년간 뉴스 수집!")
    print(f"⚠️  주의: API가 제한을 걸 수 있으니 실험적 수집입니다.")
    print("="*60)
    
    # 확장된 뉴스 수집 함수 사용
    df_extended_news = fetch_finnhub_news_extended(
        symbol=target_symbol, 
        start_date=start_date,
        days_per_request=days_per_request
    )
    
    if not df_extended_news.empty:
        print(f"\n🎉 {target_symbol} 뉴스 수집 완료!")
        print(f"📰 총 수집 기사 수: {len(df_extended_news)}개")
        
        # 날짜별 기사 분포 분석 (유효한 날짜만)
        if 'pubDate' in df_extended_news.columns:
            valid_dates = df_extended_news[df_extended_news['pubDate'].notna()].copy()
            if not valid_dates.empty:
                valid_dates['date_only'] = valid_dates['pubDate'].dt.date
                date_counts = valid_dates['date_only'].value_counts().sort_index()
                print(f"📈 수집 기간: {date_counts.index.min()} ~ {date_counts.index.max()}")
                print(f"📊 평균 일일 기사 수: {date_counts.mean():.1f}개")
                print(f"📅 유효한 날짜 기사: {len(valid_dates)}개 / 전체 {len(df_extended_news)}개")
        
        # 최신 기사 10개 미리보기
        print(f"\n📰 최신 뉴스 미리보기 (상위 10개)")
        print("-" * 80)
        for i, row in df_extended_news.head(10).iterrows():
            pub_date = row['pubDate'].strftime('%Y-%m-%d %H:%M') if pd.notna(row['pubDate']) else 'N/A'
            title = row['title'][:60] + "..." if len(row['title']) > 60 else row['title']
            publisher = row['publisher'] if row['publisher'] else row['source']
            print(f"{i+1:2d}. [{publisher}] {title}")
            print(f"    📅 {pub_date} | 🔗 {row['link'][:50]}...")
            print()
        
        # CSV 파일로 저장
        today = datetime.date.today().isoformat()
        filename = f"{target_symbol}_extended_news_{today}.csv"
        df_extended_news.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"💾 파일 저장 완료: {filename}")
        
        # 발행처별 통계
        print(f"\n📊 발행처별 기사 수 통계 (상위 10개)")
        print("-" * 40)
        publisher_col = 'publisher' if df_extended_news['publisher'].notna().any() else 'source'
        publisher_counts = df_extended_news[publisher_col].value_counts()
        for publisher, count in publisher_counts.head(10).items():
            if publisher:  # 빈 값이 아닌 경우만
                print(f"{publisher:25s}: {count:3d}개")
        
        # 카테고리별 통계 (있는 경우)
        if 'category' in df_extended_news.columns and df_extended_news['category'].notna().any():
            print(f"\n🏷️  카테고리별 기사 수")
            print("-" * 30)
            category_counts = df_extended_news['category'].value_counts()
            for category, count in category_counts.items():
                if category:
                    print(f"{category:20s}: {count:3d}개")
                    
    else:
        print(f"❌ {target_symbol} 뉴스 수집에 실패했습니다.")
        print("🔧 가능한 해결 방법:")
        print("   1. API 키 확인 (.env 파일의 'finhub' 변수)")
        print("   2. 인터넷 연결 확인")  
        print("   3. 심볼명 확인 (미국 상장 기업만 지원)")
        print("   4. 날짜 범위 조정")
    
    print("\n" + "="*60)
    print("📋 수집 완료 요약")
    print("="*60)
    print(f"🏢 대상 기업: {target_symbol}")
    print(f"📅 수집 날짜: {start_date}")
    print(f"📊 총 기사 수: {len(df_extended_news) if not df_extended_news.empty else 0}개")
    print(f"💾 저장 파일: {filename if not df_extended_news.empty else 'N/A'}")
