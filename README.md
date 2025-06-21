# 간접 투자 조언 시스템 (News & SNS-Driven)

## 프로젝트 개요
'뉴스 드리븐 마켓' 개념을 바탕으로, 뉴스 감정분석 결과를 토대로 투자자에게 **간접 투자 인사이트**를 제공하는 웹 애플리케이션입니다.

---

## 📁 프로젝트 구조

### 1. 데이터 수집 및 전처리 (`/data`)
모든 원시 데이터와 전처리된 데이터가 저장되는 폴더입니다.

#### 📊 주요 데이터 파일
- **`AAPL_1hour_data_365days.csv`** : 애플(AAPL) 1시간 단위 주가 데이터 (365일간)
  - OHLCV (Open/High/Low/Close/Volume) 기본 데이터
  - RSI, MACD, 볼린저밴드 등 기술적 지표 포함
  - 거래시간, 시간대별 특성 변수 포함

- **`AAPL_extended_news_2025-06-14.csv`** : Finnhub API로 수집한 애플 관련 뉴스 원시 데이터
  - 뉴스 제목, 본문, 발행일, 발행처 정보
  - 3년간의 뉴스 데이터 (API 제한 내에서 최대 수집)

- **`apple_finbert_finnhub.csv`** : FinBERT 감정분석이 적용된 애플 뉴스 데이터
  - 원본 뉴스 + FinBERT 감정분석 점수 (positive/neutral/negative)
  - 금융 도메인 특화 감정분석 결과

- **`news_stock_classification.csv`** : 뉴스-주가 연동 분류 학습 데이터
  - 뉴스 발생 시점과 이후 주가 변동을 연결한 라벨링 데이터
  - 상승/보합/하락 3클래스 분류용

- **`tweet_stock_classification.csv`** : 트윗-주가 연동 분류 학습 데이터
  - 트윗 발생 시점과 이후 주가 변동을 연결한 라벨링 데이터
  - VADER 감정분석 + 주가 시계열 특성 결합

- **`merged_tweets_with_sentiment.csv`** : 통합 트윗 감정분석 데이터
  - 영향력 있는 인물들의 트윗 통합
  - VADER 감정분석 결과 포함

#### 👥 개별 사용자 트윗 데이터 (13명의 영향력 있는 인물)

**🏛️ 정치/정부 관계자**
- `user_@WhiteHouse_tweets.csv` : 백악관 (도널드 트럼프 대통령)
- `user_@SecScottBessent_tweets.csv` : 스콧베센트 재무장관
- `user_@JDVance_tweets.csv` : 밴스 부통령
- `user_@marcorubio_tweets.csv` : 마르코 루비오 국무장관

**🏢 기업 CEO (테크 대기업)**
- `user_@elonmusk_tweets.csv` : 일론 머스크 (테슬라 CEO)
- `user_@sundarpichai_tweets.csv` : 순다르 피차이 (구글 CEO)
- `user_@tim_cook_tweets.csv` : 팀 쿡 (애플 CEO)

**💰 투자 전문가 및 펀드매니저**
- `user_@CathieDWood_tweets.csv` : 캐시 우드 (ARK Invest CEO, 혁신 성장주 투자, 시장 트렌드 주도)
- `user_@BillAckman_tweets.csv` : 빌 액먼 (펀드매니저, 행동주의 투자 성향)
- `user_@RayDalio_tweets.csv` : 레이 달리오 (브리지워터 창립자, 거시경제 분석, 투자 전략가)
- `user_@michaelbatnick_tweets.csv` : 마이클 배트닉 (투자 분석, 금융 인사이트 제공)
- `user_@LizAnnSonders_tweets.csv` : 리즈앤 손더스 (찰스슈왑 수석 투자전략가, 시장 전망, 투자 전략)
- `user_@Ajay_Bagga_tweets.csv` : 아제이 바가 (글로벌 매크로 전문가, 시장 전망, 투자 전략)

#### 🔧 데이터 수집 및 전처리 스크립트
- **`data_crawling.ipynb`** : 전체 데이터 수집 및 전처리리 파이프라인

---

### 2. 데이터 수집 상세 (`data_crawling.ipynb`)

이 노트북은 **6개의 주요 섹션**으로 구성된 완전한 데이터 수집 파이프라인입니다:

#### 📰 Section 1-2: Finnhub 뉴스 데이터 수집
```python
# 주요 기능
- Finnhub API를 통한 대량 뉴스 수집
- 3년간의 뉴스 데이터 수집 (API 제한 내 최대)
- 중복 제거, 날짜 정렬, 안전한 에러 처리
- FinBERT 금융 감정분석 적용
```

**특징:**
- API 호출 제한 준수 (분당 60회)
- 날짜 구간별 분할 수집으로 최대 데이터 확보
- 안전한 timestamp 변환 함수로 데이터 손실 방지
- 금융 도메인 특화 FinBERT 모델 사용

#### 🐦 Section 3-4: Twitter(X) 데이터 수집
```python
# 주요 기능  
- RapidAPI Twitter API를 통한 트윗 수집
- 13명의 영향력 있는 인물 트윗 대량 수집
- VADER 감정분석으로 소셜미디어 감정 측정
- 다중 사용자 데이터 통합 및 전처리
```

**특징:**
- 정치인, CEO, 투자전문가 등 시장 영향력 있는 인물 선정
- URL 제거 등 최소한의 전처리로 원본 감정 보존
- VADER 소셜미디어 특화 감정분석
- 실시간 API 제한 처리 및 재시도 로직

#### 📈 Section 5: 주가 데이터 수집
```python
# 주요 기능
- yfinance API로 1시간 단위 주가 데이터 수집
- OHLCV + 30개 이상의 기술적 지표 자동 계산
- 프리마켓/애프터마켓 데이터 포함
- 거래시간대별 특성 변수 생성
```

**기술적 지표:**
- 이동평균: SMA_10/20/50, EMA_12/26
- 모멘텀: RSI, MACD, MACD_Signal
- 변동성: 볼린저밴드, Volatility 지표
- 시간 특성: Hour, DayOfWeek, 거래시간 분류

#### 🔗 Section 6: 시계열 데이터 병합 및 라벨링
```python
# 주요 기능
- 트윗/뉴스 발생 시점과 이후 주가 변동 연결
- 과거 3시점 주가 데이터를 시계열 특성으로 변환
- 0.4% 임계값 기반 3클래스 분류 라벨 생성
- 머신러닝용 최종 데이터셋 구성
```

---

### 3. 모델 학습 (`/final_code`)

#### 🧠 `learning.ipynb` - 하이브리드 딥러닝 모델
이 노트북은 **뉴스 감정 + 주가 시계열**을 결합한 혁신적인 하이브리드 모델을 구현합니다.

##### 🏗️ 모델 아키텍처
```python
# 이중 경로 아키텍처
1. LSTM 경로: 주가 시계열 패턴 학습
   - Input: 과거 3시점의 OHLCV + 기술적 지표
   - LSTM Layer → Dense Layer → 시계열 특성 추출

2. MLP 경로: 감정분석 특성 학습  
   - Input: FinBERT/VADER 감정 점수
   - Dense Layer → 감정 특성 추출

3. 융합 레이어: 두 경로 결합
   - Concatenate → Dense → 3클래스 분류 (상승/보합/하락)
```

##### 📊 주요 실험 결과
**발표에 사용된 최종 모델:**
- **하이브리드 LSTM+MLP**: 64.8% 정확도
- **베이스라인 대비**: Random Forest 55.6%, SVM 52.3% 
- **SMOTE 오버샘플링**: 데이터 불균형 해결 (0.4% 기준)
- **피처 중요도**: 감정 점수 + RSI, MACD 등 기술적 지표 조합

##### 🔬 실험 구성
1. **데이터 전처리**
   - 결측치 처리, 정규화, 시계열 윈도우 생성
   - Train/Validation/Test 분할 (6:2:2)

2. **모델 비교**
   - 전통적 ML: Random Forest, SVM, XGBoost
   - 딥러닝: LSTM, MLP, 하이브리드 모델
   - 앙상블: Voting Classifier

3. **성능 평가**
   - Accuracy, Precision, Recall, F1-Score
   - Confusion Matrix, ROC Curve
   - 클래스별 성능 분석

---

### 4. 웹 애플리케이션 (`/WebPage`)

#### 🌐 `app.py` - Flask 기반 실시간 투자 조언 시스템
실시간으로 뉴스를 수집하고 감정분석하여 투자 인사이트를 제공하는 웹 인터페이스입니다.

##### 🔧 핵심 기능

**1. 실시간 뉴스 수집**
```python
@app.route("/", methods=["GET", "POST"])
def index():
    # 사용자가 입력한 기업명을 티커로 변환
    # Finnhub API로 최신 뉴스 20건 실시간 수집
    # 즉시 FinBERT 감정분석 수행
```

**2. 지능형 티커 변환**
```python
def get_ticker(company_name: str) -> str | None:
    # "애플" → "AAPL", "테슬라" → "TSLA" 등
    # 한글/영문 기업명을 자동으로 주식 티커로 변환
```

**3. 실시간 감정분석**
```python
def analyze_sentiment_finbert(text: str) -> dict:
    # FinBERT 모델로 뉴스 텍스트 즉시 분석
    # positive/neutral/negative 점수 반환
```

**4. 규칙 기반 투자 조언**
```python
def generate_advice(df, pos_thresh=0.6, neg_thresh=0.6, neu_thresh=0.6) -> str:
    # 감정분석 결과 기반 투자 조언 생성
    # 임계값 조절로 민감도 제어
```

##### 🎨 사용자 인터페이스
- **입력**: 기업명 (한글/영문 모두 지원)
- **출력**: 
  - 최신 뉴스 20건 목록
  - 감정분석 결과 파이차트 시각화
  - 데이터 기반 투자 조언 메시지
  - 뉴스별 상세 감정 점수

##### ⚡ 실시간 처리 플로우
```
사용자 입력 → 티커 변환 → 뉴스 수집 → 감정분석 → 조언 생성 → 시각화
     ↓
웹페이지에 즉시 결과 표시 (약 3-5초 소요)
```

---

## 🏆 주요 성과

### 기술적 혁신
1. **하이브리드 모델**: LSTM(시계열) + MLP(감정) 융합으로 64.8% 정확도 달성
2. **실시간 파이프라인**: 뉴스 수집부터 투자조언까지 자동화
3. **도메인 특화**: 금융 특화 FinBERT + 소셜미디어 특화 VADER 결합

### 데이터 규모
- **뉴스**: 3년간 Apple 관련 뉴스 대량 수집
- **트윗**: 13명 인플루언서 8,000+ 트윗 분석  
- **주가**: 1시간 단위 8,760 데이터포인트 + 30개 기술적 지표

### 실용성
- **웹 인터페이스**: 비전문가도 쉽게 사용 가능
- **실시간 분석**: 최신 뉴스 즉시 감정분석
- **투자 인사이트**: 데이터 기반 객관적 조언 제공

---

## 🚀 실행 방법

1. **데이터 수집**: `data/data_crawling.ipynb` 실행
2. **모델 학습**: `final_code/learning.ipynb` 실행  
3. **웹 서비스**: `WebPage/app.py` 실행 (사전에 pip install finnhub-python 설치 필요)

```bash
cd WebPage
pip install finnhub-python
python app.py
```

---

## 💡 프로젝트 의의

이 프로젝트는 **뉴스 드리븐 마켓**의 실제 구현체로서, 텍스트 감정분석과 시계열 예측을 결합한 새로운 접근법을 제시합니다. 추가적으로 실시간 웹 인터페이스를 통해 일반 투자자들도 데이터 기반의 객관적인 투자 인사이트를 얻을 수 있도록 했습니다.

