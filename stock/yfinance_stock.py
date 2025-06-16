import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import warnings
import numpy as np
warnings.filterwarnings('ignore')

def get_hourly_stock_data(ticker, days=365, save_to_csv=True):
    """
    티커를 입력받아 최근 N일간의 1시간 간격 주식 데이터를 가져오는 함수
    
    Parameters:
    ticker (str): 주식 티커 심볼 (예: 'AAPL', 'TSLA', 'AMZN')
    days (int): 수집할 일수 (최대 730일, yfinance 제약)
    save_to_csv (bool): CSV 파일로 저장할지 여부
    
    Returns:
    pandas.DataFrame: 1시간 간격 주식 데이터
    """
    
    try:
        # yfinance 1시간 간격 제약사항 확인
        if days > 730:
            print(f"⚠️ yfinance 1시간 간격 데이터는 최대 730일까지만 지원됩니다.")
            print(f"요청한 {days}일 → 730일로 조정합니다.")
            days = 730
        
        # 날짜 설정 (현재 날짜 기준)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        print(f"📊 {ticker} 주식 데이터 수집 중...")
        print(f"기간: {start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')} ({days}일)")
        print(f"간격: 1시간")
        
        # yfinance로 데이터 수집 (24시간 데이터 포함)
        stock_data = yf.download(
            ticker,
            start=start_date.strftime('%Y-%m-%d'),
            end=end_date.strftime('%Y-%m-%d'),
            interval='1h',
            prepost=True,  # 시장 외 시간 데이터 포함
            progress=False
        )
        
        if stock_data.empty:
            print(f"❌ {ticker}에 대한 데이터를 찾을 수 없습니다.")
            return None
        
        # 인덱스를 컬럼으로 변환
        stock_data = stock_data.reset_index()
        
        # 컬럼명 확인 및 정리
        print(f"🔍 원본 컬럼명: {list(stock_data.columns)}")
        
        # 인덱스 컬럼명 통일 (Datetime으로)
        if 'Date' in stock_data.columns:
            stock_data = stock_data.rename(columns={'Date': 'Datetime'})
        elif stock_data.columns[0] not in ['Datetime', 'Date']:
            # 첫 번째 컬럼이 시간 데이터인 경우
            stock_data = stock_data.rename(columns={stock_data.columns[0]: 'Datetime'})
        
        # 멀티레벨 컬럼인 경우 처리
        if isinstance(stock_data.columns, pd.MultiIndex):
            new_columns = []
            for col in stock_data.columns:
                if isinstance(col, tuple):
                    if col[0] == 'Datetime' or 'Date' in str(col[0]):
                        new_columns.append('Datetime')
                    else:
                        new_columns.append(col[0])
                else:
                    new_columns.append(col)
            stock_data.columns = new_columns
        
        # 기본 정보 출력
        print(f"✅ 데이터 수집 완료!")
        print(f"총 데이터 포인트: {len(stock_data):,}개")
        print(f"정리된 컬럼명: {list(stock_data.columns)}")
        
        # Datetime 컬럼 확인
        if 'Datetime' in stock_data.columns:
            print(f"데이터 기간: {stock_data['Datetime'].min()} ~ {stock_data['Datetime'].max()}")
        else:
            print(f"⚠️ Datetime 컬럼을 찾을 수 없습니다. 첫 번째 컬럼 사용: {stock_data.columns[0]}")
            stock_data = stock_data.rename(columns={stock_data.columns[0]: 'Datetime'})
        
        # 기본 통계
        print(f"\n📈 기본 통계:")
        print(f"시작 가격: ${stock_data['Open'].iloc[0]:.2f}")
        print(f"종료 가격: ${stock_data['Close'].iloc[-1]:.2f}")
        print(f"최고가: ${stock_data['High'].max():.2f}")
        print(f"최저가: ${stock_data['Low'].min():.2f}")
        print(f"평균 거래량: {stock_data['Volume'].mean():,.0f}")
        
        # 시간을 정시로 조정 (예: 13:30 -> 13:00)
        stock_data = adjust_time_to_hour(stock_data)
        
        # 추가 특성 계산
        stock_data = add_technical_features(stock_data)
        
        # CSV 저장
        if save_to_csv:
            filename = f"{ticker}_1hour_data_{days}days.csv"
            stock_data.to_csv(filename, index=False)
            print(f"💾 데이터가 '{filename}'에 저장되었습니다.")
        
        return stock_data
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        print(f"오류 상세: {type(e).__name__}")
        return None

def get_30min_stock_data(ticker, days=60, save_to_csv=True):
    """
    티커를 입력받아 최근 N일간의 30분 간격 주식 데이터를 가져오는 함수
    
    Parameters:
    ticker (str): 주식 티커 심볼 (예: 'AAPL', 'TSLA', 'AMZN')
    days (int): 수집할 일수 (최대 60일, yfinance 제약)
    save_to_csv (bool): CSV 파일로 저장할지 여부
    
    Returns:
    pandas.DataFrame: 30분 간격 주식 데이터
    """
    
    try:
        # yfinance 30분 간격 제약사항 확인
        if days > 60:
            print(f"⚠️ yfinance 30분 간격 데이터는 최대 60일까지만 지원됩니다.")
            print(f"요청한 {days}일 → 60일로 조정합니다.")
            days = 60
        
        # 날짜 설정 (현재 날짜 기준)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        print(f"📊 {ticker} 주식 데이터 수집 중...")
        print(f"기간: {start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')} ({days}일)")
        print(f"간격: 30분")
        
        # yfinance로 데이터 수집 (24시간 데이터 포함)
        stock_data = yf.download(
            ticker,
            start=start_date.strftime('%Y-%m-%d'),
            end=end_date.strftime('%Y-%m-%d'),
            interval='30m',
            prepost=True,  # 시장 외 시간 데이터 포함
            progress=False
        )
        
        if stock_data.empty:
            print(f"❌ {ticker}에 대한 데이터를 찾을 수 없습니다.")
            return None
        
        # 인덱스를 컬럼으로 변환
        stock_data = stock_data.reset_index()
        
        # 컬럼명 확인 및 정리
        print(f"🔍 원본 컬럼명: {list(stock_data.columns)}")
        
        # 인덱스 컬럼명 통일 (Datetime으로)
        if 'Date' in stock_data.columns:
            stock_data = stock_data.rename(columns={'Date': 'Datetime'})
        elif stock_data.columns[0] not in ['Datetime', 'Date']:
            # 첫 번째 컬럼이 시간 데이터인 경우
            stock_data = stock_data.rename(columns={stock_data.columns[0]: 'Datetime'})
        
        # 멀티레벨 컬럼인 경우 처리
        if isinstance(stock_data.columns, pd.MultiIndex):
            new_columns = []
            for col in stock_data.columns:
                if isinstance(col, tuple):
                    if col[0] == 'Datetime' or 'Date' in str(col[0]):
                        new_columns.append('Datetime')
                    else:
                        new_columns.append(col[0])
                else:
                    new_columns.append(col)
            stock_data.columns = new_columns
        
        # 기본 정보 출력
        print(f"✅ 데이터 수집 완료!")
        print(f"총 데이터 포인트: {len(stock_data):,}개")
        print(f"정리된 컬럼명: {list(stock_data.columns)}")
        
        # Datetime 컬럼 확인
        if 'Datetime' in stock_data.columns:
            print(f"데이터 기간: {stock_data['Datetime'].min()} ~ {stock_data['Datetime'].max()}")
        else:
            print(f"⚠️ Datetime 컬럼을 찾을 수 없습니다. 첫 번째 컬럼 사용: {stock_data.columns[0]}")
            stock_data = stock_data.rename(columns={stock_data.columns[0]: 'Datetime'})
        
        # 기본 통계
        print(f"\n📈 기본 통계:")
        print(f"시작 가격: ${stock_data['Open'].iloc[0]:.2f}")
        print(f"종료 가격: ${stock_data['Close'].iloc[-1]:.2f}")
        print(f"최고가: ${stock_data['High'].max():.2f}")
        print(f"최저가: ${stock_data['Low'].min():.2f}")
        print(f"평균 거래량: {stock_data['Volume'].mean():,.0f}")
        
        # 시간을 정시로 조정 (예: 13:30 -> 13:00)
        stock_data = adjust_time_to_hour(stock_data)
        
        # 추가 특성 계산
        stock_data = add_technical_features(stock_data)
        
        # CSV 저장
        if save_to_csv:
            filename = f"{ticker}_30min_data_{days}days.csv"
            stock_data.to_csv(filename, index=False)
            print(f"💾 데이터가 '{filename}'에 저장되었습니다.")
        
        return stock_data
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        print(f"오류 상세: {type(e).__name__}")
        return None

def get_max_period_data(ticker, save_to_csv=True):
    """
    yfinance 제약사항에 맞춰 가능한 최대 기간의 데이터를 수집
    - 1시간: 730일 (약 2년)
    - 30분: 60일
    - 15분: 60일
    - 5분: 60일
    - 1분: 7일
    """
    
    print(f"🚀 {ticker} 최대 기간 데이터 수집...")
    
    intervals_and_periods = [
        ('1h', 730, '2년'),
        ('30m', 60, '60일'),
        ('15m', 60, '60일'),
        ('5m', 60, '60일'),
        ('1m', 7, '7일')
    ]
    
    all_data = {}
    
    for interval, max_days, description in intervals_and_periods:
        try:
            print(f"\n📊 {interval} 간격 데이터 수집 중... (최대 {description})")
            
            end_date = datetime.now()
            start_date = end_date - timedelta(days=max_days)
            
            data = yf.download(
                ticker,
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                interval=interval,
                prepost=True,  # 시장 외 시간 데이터 포함
                progress=False
            )
            
            if not data.empty:
                data = data.reset_index()
                all_data[interval] = data
                print(f"✅ {interval} 데이터: {len(data):,}개 포인트")
                
                if save_to_csv:
                    filename = f"{ticker}_{interval}_data_{max_days}days.csv"
                    data.to_csv(filename, index=False)
                    print(f"💾 저장: {filename}")
            else:
                print(f"❌ {interval} 데이터 없음")
                
        except Exception as e:
            print(f"❌ {interval} 오류: {e}")
    
    return all_data

def get_longer_period_with_daily(ticker, days=365, save_to_csv=True):
    """
    1년 데이터가 필요한 경우 일별 데이터로 수집
    """
    
    try:
        print(f"📊 {ticker} 일별 데이터 수집 중... ({days}일)")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # 일별 데이터는 제약이 거의 없음
        daily_data = yf.download(
            ticker,
            start=start_date.strftime('%Y-%m-%d'),
            end=end_date.strftime('%Y-%m-%d'),
            interval='1d',
            prepost=True,  # 시장 외 시간 데이터 포함
            progress=False
        )
        
        if daily_data.empty:
            print(f"❌ {ticker} 일별 데이터를 찾을 수 없습니다.")
            return None
        
        daily_data = daily_data.reset_index()
        
        print(f"✅ 일별 데이터 수집 완료!")
        print(f"총 데이터 포인트: {len(daily_data):,}개")
        print(f"데이터 기간: {daily_data['Date'].min()} ~ {daily_data['Date'].max()}")
        
        # 기술적 지표 추가
        daily_data = add_technical_features_daily(daily_data)
        
        if save_to_csv:
            filename = f"{ticker}_daily_data_{days}days.csv"
            daily_data.to_csv(filename, index=False)
            print(f"💾 데이터가 '{filename}'에 저장되었습니다.")
        
        return daily_data
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        return None

def add_technical_features_daily(df):
    """일별 데이터용 기술적 지표 추가"""
    
    print("🔧 기술적 지표 계산 중...")
    
    # 수익률 계산
    df['Returns'] = df['Close'].pct_change()
    df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # 이동평균
    df['SMA_5'] = df['Close'].rolling(window=5).mean()
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    
    # 지수이동평균
    df['EMA_12'] = df['Close'].ewm(span=12).mean()
    df['EMA_26'] = df['Close'].ewm(span=26).mean()
    
    # MACD
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
    df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
    
    # RSI
    df['RSI'] = calculate_rsi(df['Close'])
    
    # 볼린저 밴드
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
    df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
    df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
    df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
    
    # 변동성
    df['Volatility_5'] = df['Returns'].rolling(window=5).std()
    df['Volatility_10'] = df['Returns'].rolling(window=10).std()
    df['Volatility_20'] = df['Returns'].rolling(window=20).std()
    
    # 가격 변화
    df['Price_Change'] = df['Close'] - df['Open']
    df['Price_Change_Pct'] = (df['Close'] - df['Open']) / df['Open'] * 100
    
    # High-Low 스프레드
    df['HL_Spread'] = df['High'] - df['Low']
    df['HL_Spread_Pct'] = (df['High'] - df['Low']) / df['Close'] * 100
    
    # 시간 특성 (일별 데이터용)
    df['DayOfWeek'] = pd.to_datetime(df['Date']).dt.dayofweek
    df['Month'] = pd.to_datetime(df['Date']).dt.month
    df['Quarter'] = pd.to_datetime(df['Date']).dt.quarter
    df['DayOfMonth'] = pd.to_datetime(df['Date']).dt.day
    df['WeekOfYear'] = pd.to_datetime(df['Date']).dt.isocalendar().week
    
    # 거래일 특성
    df['Is_Monday'] = (df['DayOfWeek'] == 0).astype(int)
    df['Is_Friday'] = (df['DayOfWeek'] == 4).astype(int)
    df['Is_MonthEnd'] = pd.to_datetime(df['Date']).dt.is_month_end.astype(int)
    df['Is_MonthStart'] = pd.to_datetime(df['Date']).dt.is_month_start.astype(int)
    
    print(f"✅ 기술적 지표 추가 완료! 총 컬럼 수: {len(df.columns)}개")
    
    return df

def add_technical_features(df):
    """기술적 지표 추가"""
    
    print("🔧 기술적 지표 계산 중...")
    
    # 수익률 계산
    df['Returns'] = df['Close'].pct_change()
    df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # 이동평균
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    
    # 지수이동평균
    df['EMA_12'] = df['Close'].ewm(span=12).mean()
    df['EMA_26'] = df['Close'].ewm(span=26).mean()
    
    # MACD
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
    df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
    
    # RSI
    df['RSI'] = calculate_rsi(df['Close'])
    
    # 볼린저 밴드
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
    df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
    df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
    df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
    
    # 변동성
    df['Volatility_10'] = df['Returns'].rolling(window=10).std()
    df['Volatility_20'] = df['Returns'].rolling(window=20).std()
    
    # 가격 변화
    df['Price_Change'] = df['Close'] - df['Open']
    df['Price_Change_Pct'] = (df['Close'] - df['Open']) / df['Open'] * 100
    
    # High-Low 스프레드
    df['HL_Spread'] = df['High'] - df['Low']
    df['HL_Spread_Pct'] = (df['High'] - df['Low']) / df['Close'] * 100
    
    # 시간 특성
    df['Hour'] = df['Datetime'].dt.hour
    df['DayOfWeek'] = df['Datetime'].dt.dayofweek
    df['Month'] = df['Datetime'].dt.month
    df['Quarter'] = df['Datetime'].dt.quarter
    
    # 거래시간 여부 (미국 주식시장: 9:30-16:00 EST, 24시간 포함으로 확대)
    df['Is_Trading_Hours'] = ((df['Hour'] >= 9) & (df['Hour'] <= 16)).astype(int)  # 정규 거래시간
    df['Is_Market_Open'] = ((df['Hour'] >= 9) & (df['Hour'] < 16)).astype(int)     # 시장 개장시간
    df['Is_Premarket'] = ((df['Hour'] >= 4) & (df['Hour'] < 9)).astype(int)       # 프리마켓 (4:00-9:30)
    df['Is_Aftermarket'] = ((df['Hour'] >= 16) & (df['Hour'] <= 20)).astype(int)  # 애프터마켓 (16:00-20:00)
    df['Is_Extended_Hours'] = (df['Is_Premarket'] | df['Is_Aftermarket']).astype(int)  # 연장거래시간
    
    print(f"✅ 기술적 지표 추가 완료! 총 컬럼 수: {len(df.columns)}개")
    
    return df

def calculate_rsi(prices, window=14):
    """RSI (Relative Strength Index) 계산"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def adjust_time_to_hour(df):
    """시간을 정시로 조정하는 함수 (예: 13:30 -> 13:00)"""
    
    print("🕐 시간을 정시로 조정 중...")
    
    # Datetime 컬럼이 있는지 확인
    if 'Datetime' in df.columns:
        # 시간을 정시로 조정 (분, 초를 0으로 설정)
        df['Datetime'] = pd.to_datetime(df['Datetime'])
        df['Datetime'] = df['Datetime'].dt.floor('H')  # 시간 단위로 내림
        
        print(f"✅ 시간 조정 완료: {df['Datetime'].min()} ~ {df['Datetime'].max()}")
        
        # 중복된 시간이 있는 경우 마지막 값 유지
        df = df.drop_duplicates(subset=['Datetime'], keep='last')
        print(f"중복 제거 후 데이터 포인트: {len(df):,}개")
        
    return df

def get_multiple_tickers_hourly(tickers, days=365, save_individual=True, save_combined=True):
    """여러 티커의 1시간 간격 데이터를 한번에 수집"""
    
    print(f"🚀 {len(tickers)}개 티커 1시간 간격 데이터 수집 시작...")
    print(f"티커 목록: {', '.join(tickers)}")
    print(f"수집 기간: 최근 {days}일")
    print("=" * 60)
    
    all_data = {}
    
    for i, ticker in enumerate(tickers, 1):
        print(f"\n[{i}/{len(tickers)}] {ticker} 처리 중...")
        
        data = get_hourly_stock_data(ticker, days=days, save_to_csv=save_individual)
        
        if data is not None:
            all_data[ticker] = data
            print(f"✅ {ticker} 완료")
        else:
            print(f"❌ {ticker} 실패")
        
        print("-" * 40)
    
    # 통합 데이터 저장
    if save_combined and all_data:
        print(f"\n💾 통합 데이터 저장 중...")
        
        # 각 티커별로 컬럼에 티커명 추가
        combined_data = pd.DataFrame()
        
        for ticker, data in all_data.items():
            ticker_data = data.copy()
            ticker_data['Ticker'] = ticker
            combined_data = pd.concat([combined_data, ticker_data], ignore_index=True)
        
        combined_filename = f"multiple_stocks_1hour_data_{days}days.csv"
        combined_data.to_csv(combined_filename, index=False)
        print(f"✅ 통합 데이터가 '{combined_filename}'에 저장되었습니다.")
        print(f"총 데이터 포인트: {len(combined_data):,}개")
    
    return all_data

def get_multiple_tickers(tickers, days=60, save_individual=True, save_combined=True):
    """여러 티커의 30분 간격 데이터를 한번에 수집"""
    
    print(f"🚀 {len(tickers)}개 티커 30분 간격 데이터 수집 시작...")
    print(f"티커 목록: {', '.join(tickers)}")
    print(f"수집 기간: 최근 {days}일")
    print("=" * 60)
    
    all_data = {}
    
    for i, ticker in enumerate(tickers, 1):
        print(f"\n[{i}/{len(tickers)}] {ticker} 처리 중...")
        
        data = get_30min_stock_data(ticker, days=days, save_to_csv=save_individual)
        
        if data is not None:
            all_data[ticker] = data
            print(f"✅ {ticker} 완료")
        else:
            print(f"❌ {ticker} 실패")
        
        print("-" * 40)
    
    # 통합 데이터 저장
    if save_combined and all_data:
        print(f"\n💾 통합 데이터 저장 중...")
        
        # 각 티커별로 컬럼에 티커명 추가
        combined_data = pd.DataFrame()
        
        for ticker, data in all_data.items():
            ticker_data = data.copy()
            ticker_data['Ticker'] = ticker
            combined_data = pd.concat([combined_data, ticker_data], ignore_index=True)
        
        combined_filename = f"multiple_stocks_30min_data_{days}days.csv"
        combined_data.to_csv(combined_filename, index=False)
        print(f"✅ 통합 데이터가 '{combined_filename}'에 저장되었습니다.")
        print(f"총 데이터 포인트: {len(combined_data):,}개")
    
    return all_data

def analyze_data_summary(data_dict):
    """수집된 데이터 요약 분석"""
    
    print("\n" + "=" * 60)
    print("📊 데이터 수집 요약")
    print("=" * 60)
    
    for ticker, data in data_dict.items():
        if data is not None:
            print(f"\n{ticker}:")
            print(f"  데이터 포인트: {len(data):,}개")
            print(f"  기간: {data['Datetime'].min().strftime('%Y-%m-%d %H:%M')} ~ {data['Datetime'].max().strftime('%Y-%m-%d %H:%M')}")
            print(f"  가격 범위: ${data['Low'].min():.2f} ~ ${data['High'].max():.2f}")
            print(f"  평균 거래량: {data['Volume'].mean():,.0f}")
            
            # 결측치 확인
            missing_count = data.isnull().sum().sum()
            if missing_count > 0:
                print(f"  ⚠️ 결측치: {missing_count}개")
            else:
                print(f"  ✅ 결측치 없음")

# 사용 예시
if __name__ == "__main__":
    
    print("🎯 yfinance 제약사항 안내:")
    print("- 1시간 간격: 최대 730일 (약 2년) ⭐ 추천!")
    print("- 30분 간격: 최대 60일")
    print("- 일별 간격: 제한 없음")
    print("=" * 60)
    
    # 1. 1시간 간격 데이터 (1년) - 메인 추천!
    print("\n🎯 1시간 간격 데이터 수집 (1년) - 추천!")
    aapl_1h = get_hourly_stock_data('AAPL', days=365)
    
    if aapl_1h is not None:
        print(f"\n📋 AAPL 1시간 데이터 미리보기:")
        print(aapl_1h[['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']].head())
        
        # 데이터 양 분석
        trading_hours = aapl_1h[aapl_1h['Is_Trading_Hours'] == 1]
        print(f"\n📊 LSTM 학습용 데이터 분석:")
        print(f"전체 시간: {len(aapl_1h):,}개")
        print(f"거래시간만: {len(trading_hours):,}개")
        print(f"LSTM 시퀀스 길이 30 가정 시 학습 샘플: {len(trading_hours) - 30:,}개")
    
    print("\n" + "="*80)
    
    # 2. 여러 티커 1시간 데이터 (1년)
    print("\n🎯 여러 티커 1시간 데이터 수집 (1년)")
    tickers = ['AAPL', 'AMZN', 'TSLA', 'GOOGL', 'MSFT']
    
    all_stock_data = get_multiple_tickers_hourly(tickers, days=365)
    
    # 3. 요약 분석
    analyze_data_summary(all_stock_data)
    
    print("\n" + "="*80)
    
    # 4. 30분 간격 비교용 (60일)
    print("\n🎯 30분 간격 데이터 비교 (60일)")
    print("⚠️ 30분 간격은 최대 60일 제한이 있습니다.")
    
    # 현재 날짜 확인
    current_date = datetime.now()
    print(f"현재 날짜: {current_date.strftime('%Y-%m-%d')}")
    
    aapl_30m = get_30min_stock_data('AAPL', days=30)  # 30일로 줄여서 안전하게 테스트
    
    if aapl_30m is not None:
        print(f"\n📋 AAPL 30분 데이터 미리보기:")
        print(aapl_30m[['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']].head())
    
    print("\n🎉 모든 데이터 수집 완료!")
    print("\n💡 권장사항:")
    print("✅ 1시간 간격 1년 데이터 - LSTM 학습에 최적!")
    print(f"   → 약 {365 * 6.5:.0f}개 거래시간 데이터 포인트")
    print("   → 충분한 데이터 양 + 적절한 시간 해상도")
    print("⚠️ 30분 간격은 60일 제한으로 데이터 부족")
    print("⚠️ 일별 데이터는 시간 해상도 부족")
