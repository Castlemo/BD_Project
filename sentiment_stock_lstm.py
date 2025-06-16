import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ML/DL 라이브러리
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# 시각화
import matplotlib.pyplot as plt
import seaborn as sns

class SentimentStockPredictor:
    """
    뉴스 감정분석(FinBERT) + 트위터 감정분석(VADER) + 주가 데이터를 
    통합하여 LSTM으로 주가 방향성을 예측하는 클래스
    """
    
    def __init__(self, stock_file, news_file, twitter_file):
        """
        Parameters:
        stock_file (str): 주가 데이터 CSV 파일 경로
        news_file (str): 뉴스 감정분석 CSV 파일 경로  
        twitter_file (str): 트위터 감정분석 CSV 파일 경로
        """
        self.stock_file = stock_file
        self.news_file = news_file
        self.twitter_file = twitter_file
        
        self.combined_data = None
        self.model = None
        self.scaler = None
        self.sequence_length = 24  # 24시간 시퀀스
        
        print("🚀 감정분석 기반 주가 예측 시스템 초기화 완료!")
    
    def load_and_sync_data(self):
        """3개 데이터셋을 로드하고 시간 기준으로 동기화"""
        
        print("\n📊 데이터 로딩 및 동기화 시작...")
        
        # 1. 주가 데이터 로드
        print("📈 주가 데이터 로딩...")
        try:
            stock_df = pd.read_csv(self.stock_file)
            stock_df['Datetime'] = pd.to_datetime(stock_df['Datetime'])
            stock_df = stock_df.sort_values('Datetime').reset_index(drop=True)
            print(f"✅ 주가 데이터: {len(stock_df):,}개 행, 기간: {stock_df['Datetime'].min()} ~ {stock_df['Datetime'].max()}")
        except Exception as e:
            print(f"❌ 주가 데이터 로딩 실패: {e}")
            return None
        
        # 2. 뉴스 감정분석 데이터 로드
        print("📰 뉴스 감정분석 데이터 로딩...")
        try:
            news_df = pd.read_csv(self.news_file)
            news_df['Date'] = pd.to_datetime(news_df['Date'])
            news_df = news_df.sort_values('Date').reset_index(drop=True)
            print(f"✅ 뉴스 데이터: {len(news_df):,}개 행, 기간: {news_df['Date'].min()} ~ {news_df['Date'].max()}")
        except Exception as e:
            print(f"❌ 뉴스 데이터 로딩 실패: {e}")
            return None
        
        # 3. 트위터 감정분석 데이터 로드
        print("🐦 트위터 감정분석 데이터 로딩...")
        try:
            twitter_df = pd.read_csv(self.twitter_file)
            twitter_df['created_at'] = pd.to_datetime(twitter_df['created_at'])
            twitter_df = twitter_df.sort_values('created_at').reset_index(drop=True)
            print(f"✅ 트위터 데이터: {len(twitter_df):,}개 행, 기간: {twitter_df['created_at'].min()} ~ {twitter_df['created_at'].max()}")
        except Exception as e:
            print(f"❌ 트위터 데이터 로딩 실패: {e}")
            return None
        
        # 4. 시간 기준으로 데이터 동기화
        print("\n🔄 시간 기준 데이터 동기화...")
        
        # 주가 데이터를 기준으로 1시간 단위로 집계
        stock_df['hour'] = stock_df['Datetime'].dt.floor('H')
        
        # 뉴스 데이터를 1시간 단위로 집계
        news_df['hour'] = news_df['Date'].dt.floor('H')
        news_hourly = news_df.groupby('hour').agg({
            'pos': 'mean',
            'neu': 'mean', 
            'neg': 'mean',
            'sentiment': lambda x: x.mode().iloc[0] if not x.empty else 'neutral'
        }).reset_index()
        news_hourly.columns = ['hour', 'news_pos', 'news_neu', 'news_neg', 'news_sentiment']
        
        # 트위터 데이터를 1시간 단위로 집계
        twitter_df['hour'] = twitter_df['created_at'].dt.floor('H')
        twitter_hourly = twitter_df.groupby('hour').agg({
            'pos': 'mean',
            'neu': 'mean',
            'neg': 'mean', 
            'sentiment': lambda x: x.mode().iloc[0] if not x.empty else 'neutral'
        }).reset_index()
        twitter_hourly.columns = ['hour', 'twitter_pos', 'twitter_neu', 'twitter_neg', 'twitter_sentiment']
        
        # 주가 데이터도 시간별로 정리 (중복 제거)
        stock_hourly = stock_df.drop_duplicates(subset=['hour'], keep='last').reset_index(drop=True)
        
        # 3개 데이터 병합
        print("🔗 데이터 병합 중...")
        combined = stock_hourly.merge(news_hourly, on='hour', how='left')
        combined = combined.merge(twitter_hourly, on='hour', how='left')
        
        # 결측치 처리 (forward fill)
        sentiment_cols = ['news_pos', 'news_neu', 'news_neg', 'twitter_pos', 'twitter_neu', 'twitter_neg']
        combined[sentiment_cols] = combined[sentiment_cols].fillna(method='ffill')
        combined[sentiment_cols] = combined[sentiment_cols].fillna(0.5)  # 초기값
        
        # 감정 라벨 결측치 처리
        combined['news_sentiment'] = combined['news_sentiment'].fillna('neutral')
        combined['twitter_sentiment'] = combined['twitter_sentiment'].fillna('neutral')
        
        print(f"✅ 데이터 병합 완료: {len(combined):,}개 행")
        print(f"📅 통합 데이터 기간: {combined['hour'].min()} ~ {combined['hour'].max()}")
        
        self.combined_data = combined
        return combined
    
    def create_combined_features(self):
        """통합 특성 생성"""
        
        if self.combined_data is None:
            print("❌ 먼저 데이터를 로드해주세요.")
            return
        
        print("\n🔧 통합 특성 생성 중...")
        
        df = self.combined_data.copy()
        
        # 1. 감정 점수 통합 (가중 평균: FinBERT 60%, VADER 40%)
        df['combined_pos'] = 0.6 * df['news_pos'] + 0.4 * df['twitter_pos']
        df['combined_neu'] = 0.6 * df['news_neu'] + 0.4 * df['twitter_neu']
        df['combined_neg'] = 0.6 * df['news_neg'] + 0.4 * df['twitter_neg']
        
        # 2. 감정 강도 및 극성
        df['news_sentiment_intensity'] = df[['news_pos', 'news_neu', 'news_neg']].max(axis=1) - df[['news_pos', 'news_neu', 'news_neg']].min(axis=1)
        df['twitter_sentiment_intensity'] = df[['twitter_pos', 'twitter_neu', 'twitter_neg']].max(axis=1) - df[['twitter_pos', 'twitter_neu', 'twitter_neg']].min(axis=1)
        df['combined_sentiment_intensity'] = df[['combined_pos', 'combined_neu', 'combined_neg']].max(axis=1) - df[['combined_pos', 'combined_neu', 'combined_neg']].min(axis=1)
        
        # 3. 감정 점수 변화율 (시간별)
        for col in ['news_pos', 'news_neu', 'news_neg', 'twitter_pos', 'twitter_neu', 'twitter_neg']:
            df[f'{col}_change'] = df[col].pct_change()
        
        # 4. 감정 점수 이동평균 (3시간, 6시간)
        for window in [3, 6]:
            for col in ['combined_pos', 'combined_neu', 'combined_neg']:
                df[f'{col}_ma{window}'] = df[col].rolling(window).mean()
        
        # 5. 주가 특성
        df['price_change'] = df['Close'].pct_change()  # 수익률
        df['price_volatility'] = df['price_change'].rolling(6).std()  # 변동성
        df['volume_ma'] = df['Volume'].rolling(6).mean()  # 거래량 이동평균
        
        # 6. 기술적 지표
        df['sma_5'] = df['Close'].rolling(5).mean()
        df['sma_10'] = df['Close'].rolling(10).mean()
        df['rsi'] = self._calculate_rsi(df['Close'])
        
        # 7. 시간 특성
        df['hour_of_day'] = df['hour'].dt.hour
        df['day_of_week'] = df['hour'].dt.dayofweek
        df['is_trading_hours'] = ((df['hour_of_day'] >= 9) & (df['hour_of_day'] <= 16)).astype(int)
        
        # 8. 타겟 라벨 생성 (다음 시간의 주가 방향성)
        df['next_price_change'] = df['price_change'].shift(-1)  # 다음 시간 수익률
        
        # 임계값 설정: ±0.5%
        def classify_direction(change):
            if pd.isna(change):
                return 'neutral'
            elif change > 0.005:  # +0.5%
                return 'positive'
            elif change < -0.005:  # -0.5%
                return 'negative'
            else:
                return 'neutral'
        
        df['target'] = df['next_price_change'].apply(classify_direction)
        
        # 결측치 제거
        df = df.dropna().reset_index(drop=True)
        
        print(f"✅ 특성 생성 완료: {len(df):,}개 행, {len(df.columns)}개 컬럼")
        print(f"📊 타겟 분포:")
        print(df['target'].value_counts())
        
        self.combined_data = df
        return df
    
    def _calculate_rsi(self, prices, window=14):
        """RSI 계산"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def prepare_lstm_data(self):
        """LSTM 입력용 시퀀스 데이터 준비"""
        
        if self.combined_data is None:
            print("❌ 먼저 특성을 생성해주세요.")
            return
        
        print(f"\n🔄 LSTM 시퀀스 데이터 준비 중... (시퀀스 길이: {self.sequence_length})")
        
        df = self.combined_data.copy()
        
        # 특성 선택 (감정 + 주가 + 기술적 지표)
        feature_cols = [
            # 감정 점수 (원본)
            'news_pos', 'news_neu', 'news_neg',
            'twitter_pos', 'twitter_neu', 'twitter_neg',
            # 통합 감정 점수
            'combined_pos', 'combined_neu', 'combined_neg',
            # 감정 강도
            'news_sentiment_intensity', 'twitter_sentiment_intensity', 'combined_sentiment_intensity',
            # 주가 특성
            'Close', 'Volume', 'price_change', 'price_volatility',
            # 기술적 지표
            'rsi', 'sma_5', 'sma_10',
            # 시간 특성
            'hour_of_day', 'day_of_week', 'is_trading_hours'
        ]
        
        # 사용 가능한 컬럼만 선택
        available_cols = [col for col in feature_cols if col in df.columns]
        print(f"📋 사용할 특성: {len(available_cols)}개")
        print(f"  {available_cols}")
        
        # 데이터 준비
        X = df[available_cols].values
        y = df['target'].values
        
        # 레이블 인코딩
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        self.label_encoder = le
        
        print(f"📊 레이블 매핑: {dict(zip(le.classes_, le.transform(le.classes_)))}")
        
        # 데이터 정규화
        self.scaler = MinMaxScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # 시퀀스 데이터 생성
        X_sequences, y_sequences = [], []
        
        for i in range(len(X_scaled) - self.sequence_length):
            X_sequences.append(X_scaled[i:(i + self.sequence_length)])
            y_sequences.append(y_encoded[i + self.sequence_length])
        
        X_sequences = np.array(X_sequences)
        y_sequences = np.array(y_sequences)
        
        print(f"✅ 시퀀스 데이터 생성 완료:")
        print(f"  X shape: {X_sequences.shape}")
        print(f"  y shape: {y_sequences.shape}")
        
        # 훈련/테스트 분할
        X_train, X_test, y_train, y_test = train_test_split(
            X_sequences, y_sequences, test_size=0.2, random_state=42, stratify=y_sequences
        )
        
        print(f"📊 데이터 분할:")
        print(f"  훈련 세트: {X_train.shape[0]:,}개")
        print(f"  테스트 세트: {X_test.shape[0]:,}개")
        
        self.X_train, self.X_test = X_train, X_test
        self.y_train, self.y_test = y_train, y_test
        self.n_features = X_sequences.shape[2]
        self.n_classes = len(np.unique(y_sequences))
        
        return X_train, X_test, y_train, y_test
    
    def build_lstm_model(self):
        """LSTM 모델 구축"""
        
        print(f"\n🏗️ LSTM 모델 구축 중...")
        print(f"  입력 형태: ({self.sequence_length}, {self.n_features})")
        print(f"  출력 클래스: {self.n_classes}개")
        
        model = Sequential([
            # 첫 번째 LSTM 레이어
            LSTM(128, return_sequences=True, input_shape=(self.sequence_length, self.n_features)),
            Dropout(0.2),
            BatchNormalization(),
            
            # 두 번째 LSTM 레이어  
            LSTM(64, return_sequences=True),
            Dropout(0.2),
            BatchNormalization(),
            
            # 세 번째 LSTM 레이어
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            
            # 완전연결 레이어
            Dense(50, activation='relu'),
            Dropout(0.2),
            Dense(25, activation='relu'),
            
            # 출력 레이어 (3클래스 분류)
            Dense(self.n_classes, activation='softmax')
        ])
        
        # 모델 컴파일
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',  # 정수 라벨용
            metrics=['accuracy']
        )
        
        print("✅ 모델 구축 완료!")
        model.summary()
        
        self.model = model
        return model
    
    def train_model(self, epochs=50, batch_size=32):
        """모델 훈련"""
        
        if self.model is None:
            print("❌ 먼저 모델을 구축해주세요.")
            return
        
        print(f"\n🚀 모델 훈련 시작... (에포크: {epochs}, 배치: {batch_size})")
        
        # 콜백 설정
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.0001)
        ]
        
        # 훈련 실행
        history = self.model.fit(
            self.X_train, self.y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(self.X_test, self.y_test),
            callbacks=callbacks,
            verbose=1
        )
        
        print("✅ 모델 훈련 완료!")
        
        # 훈련 곡선 시각화
        self._plot_training_history(history)
        
        return history
    
    def evaluate_model(self):
        """모델 평가"""
        
        if self.model is None:
            print("❌ 훈련된 모델이 없습니다.")
            return
        
        print("\n📊 모델 평가 중...")
        
        # 예측
        y_pred_proba = self.model.predict(self.X_test)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # 정확도
        accuracy = accuracy_score(self.y_test, y_pred)
        print(f"🎯 테스트 정확도: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # 분류 리포트
        target_names = self.label_encoder.classes_
        print(f"\n📋 분류 리포트:")
        print(classification_report(self.y_test, y_pred, target_names=target_names))
        
        # 혼동 행렬
        cm = confusion_matrix(self.y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=target_names, yticklabels=target_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()
        
        return accuracy, y_pred, y_pred_proba
    
    def _plot_training_history(self, history):
        """훈련 곡선 시각화"""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss 곡선
        ax1.plot(history.history['loss'], label='Train Loss')
        ax1.plot(history.history['val_loss'], label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        
        # Accuracy 곡선
        ax2.plot(history.history['accuracy'], label='Train Accuracy')
        ax2.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax2.set_title('Model Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        
        plt.tight_layout()
        plt.show()
    
    def predict_next_direction(self, n_predictions=24):
        """향후 N시간의 주가 방향성 예측"""
        
        if self.model is None:
            print("❌ 훈련된 모델이 없습니다.")
            return
        
        print(f"\n🔮 향후 {n_predictions}시간 주가 방향성 예측...")
        
        # 최근 시퀀스 데이터 준비
        recent_data = self.combined_data.tail(self.sequence_length)
        
        feature_cols = [
            'news_pos', 'news_neu', 'news_neg',
            'twitter_pos', 'twitter_neu', 'twitter_neg', 
            'combined_pos', 'combined_neu', 'combined_neg',
            'news_sentiment_intensity', 'twitter_sentiment_intensity', 'combined_sentiment_intensity',
            'Close', 'Volume', 'price_change', 'price_volatility',
            'rsi', 'sma_5', 'sma_10',
            'hour_of_day', 'day_of_week', 'is_trading_hours'
        ]
        
        available_cols = [col for col in feature_cols if col in recent_data.columns]
        X_recent = recent_data[available_cols].values
        X_recent_scaled = self.scaler.transform(X_recent)
        
        predictions = []
        current_sequence = X_recent_scaled.copy()
        
        for i in range(n_predictions):
            # 예측
            X_input = current_sequence.reshape(1, self.sequence_length, self.n_features)
            pred_proba = self.model.predict(X_input, verbose=0)
            pred_class = np.argmax(pred_proba, axis=1)[0]
            pred_label = self.label_encoder.inverse_transform([pred_class])[0]
            
            predictions.append({
                'hour': i + 1,
                'prediction': pred_label,
                'confidence': np.max(pred_proba),
                'probabilities': {
                    label: prob for label, prob in zip(self.label_encoder.classes_, pred_proba[0])
                }
            })
            
            # 시퀀스 업데이트 (단순화: 마지막 값 복사)
            current_sequence = np.roll(current_sequence, -1, axis=0)
            current_sequence[-1] = current_sequence[-2]  # 단순 복사
        
        # 결과 출력
        print("\n🎯 예측 결과:")
        for pred in predictions:
            print(f"  시간 +{pred['hour']:2d}: {pred['prediction']:8s} (신뢰도: {pred['confidence']:.3f})")
        
        # 예측 분포
        pred_counts = {}
        for pred in predictions:
            label = pred['prediction']
            pred_counts[label] = pred_counts.get(label, 0) + 1
        
        print(f"\n📊 {n_predictions}시간 예측 분포:")
        for label, count in pred_counts.items():
            print(f"  {label}: {count}시간 ({count/n_predictions*100:.1f}%)")
        
        return predictions
    
    def save_model(self, model_path="sentiment_stock_lstm_model.h5"):
        """모델 저장"""
        if self.model is not None:
            self.model.save(model_path)
            print(f"💾 모델이 '{model_path}'에 저장되었습니다.")
        else:
            print("❌ 저장할 모델이 없습니다.")
    
    def run_full_pipeline(self, epochs=50, batch_size=32):
        """전체 파이프라인 실행"""
        
        print("🚀 감정분석 기반 주가 예측 시스템 실행!")
        print("=" * 60)
        
        # 1. 데이터 로드 및 동기화
        if self.load_and_sync_data() is None:
            return
        
        # 2. 특성 생성
        self.create_combined_features()
        
        # 3. LSTM 데이터 준비
        self.prepare_lstm_data()
        
        # 4. 모델 구축
        self.build_lstm_model()
        
        # 5. 모델 훈련
        self.train_model(epochs=epochs, batch_size=batch_size)
        
        # 6. 모델 평가
        accuracy, _, _ = self.evaluate_model()
        
        # 7. 미래 예측
        self.predict_next_direction(n_predictions=24)
        
        # 8. 모델 저장
        self.save_model()
        
        print(f"\n🎉 전체 파이프라인 완료! 최종 정확도: {accuracy:.4f}")
        
        return accuracy

# 사용 예시
if __name__ == "__main__":
    
    print("🎯 감정분석 기반 주가 예측 시스템")
    print("=" * 60)
    
    # 파일 경로 설정 (실제 파일 경로로 수정 필요)
    stock_file = "stock/AAPL_1hour_data_365days.csv"  # 주가 데이터
    news_file = "finnhub/AAPL_finnhub_processed_final.csv"  # 뉴스 감정분석
    twitter_file = "X_data/merged_tweets_with_sentiment.csv"  # 트위터 감정분석
    
    # 시스템 초기화
    predictor = SentimentStockPredictor(
        stock_file=stock_file,
        news_file=news_file, 
        twitter_file=twitter_file
    )
    
    # 전체 파이프라인 실행
    try:
        final_accuracy = predictor.run_full_pipeline(epochs=30, batch_size=32)
        print(f"\n✅ 시스템 구축 완료! 최종 성능: {final_accuracy:.4f}")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        print("파일 경로와 데이터 형식을 확인해주세요.")