#!/usr/bin/env python3
"""
Safe X (Twitter) API v2 Tweets Crawler - All-in-One Version
Windows 호환 + 디버그 기능 + 안전한 rate limiting 통합 버전
"""

import requests
import json
import time
import os
from datetime import datetime, timezone
import csv
from typing import Dict, List, Optional
import logging

# Setup logging with Windows compatibility
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('twitter_crawler_safe.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SafeXTweetsCrawler:
    def __init__(self, bearer_token: str, debug_mode: bool = False):
        """
        Initialize X API crawler with enhanced safety features
        
        Args:
            bearer_token: X API Bearer Token from developer portal
            debug_mode: Enable detailed debug output
        """
        self.bearer_token = bearer_token
        self.base_url = "https://api.twitter.com/2"
        self.headers = {
            "Authorization": f"Bearer {bearer_token}",
            "Content-Type": "application/json"
        }
        
        # Enhanced rate limiting tracking
        self.last_user_lookup = 0
        self.last_tweets_fetch = 0
        self.rate_limit_window = 16 * 60  # 16 minutes (1 minute buffer)
        self.debug_mode = debug_mode
        
    def _safe_wait_for_rate_limit(self, endpoint_type: str, force_wait: bool = False) -> None:
        """
        Enhanced rate limit waiting with safety buffer
        
        Args:
            endpoint_type: 'user_lookup' or 'tweets_fetch'
            force_wait: Force wait regardless of time elapsed
        """
        current_time = time.time()
        
        if endpoint_type == 'user_lookup':
            time_since_last = current_time - self.last_user_lookup
            if time_since_last < self.rate_limit_window or force_wait:
                wait_time = self.rate_limit_window - time_since_last if not force_wait else self.rate_limit_window
                logger.info(f"[SAFE] 안전한 대기: {wait_time:.0f}초 (사용자 조회)")
                for i in range(int(wait_time), 0, -1):
                    if i % 60 == 0:
                        logger.info(f"[TIME] {i//60}분 {i%60}초 남음...")
                    time.sleep(1)
                    
        elif endpoint_type == 'tweets_fetch':
            time_since_last = current_time - self.last_tweets_fetch
            if time_since_last < self.rate_limit_window or force_wait:
                wait_time = self.rate_limit_window - time_since_last if not force_wait else self.rate_limit_window
                logger.info(f"[SAFE] 안전한 대기: {wait_time:.0f}초 (트윗 조회)")
                for i in range(int(wait_time), 0, -1):
                    if i % 60 == 0:
                        logger.info(f"[TIME] {i//60}분 {i%60}초 남음...")
                    time.sleep(1)
    
    def get_user_id_by_username(self, username: str) -> Optional[str]:
        """
        Get user ID from username with enhanced error handling
        """
        self._safe_wait_for_rate_limit('user_lookup')
        
        url = f"{self.base_url}/users/by/username/{username}"
        params = {
            "user.fields": "id,name,username,description,public_metrics,verified,created_at"
        }
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                logger.info(f"[SEARCH] 사용자 조회 시도 {attempt + 1}/{max_retries}")
                response = requests.get(url, headers=self.headers, params=params, timeout=30)
                self.last_user_lookup = time.time()
                
                if self.debug_mode:
                    logger.info(f"[DEBUG] 사용자 조회 응답 코드: {response.status_code}")
                    logger.info(f"[DEBUG] 사용자 조회 URL: {url}")
                    logger.info(f"[DEBUG] 사용자 조회 파라미터: {params}")
                
                if response.status_code == 200:
                    data = response.json()
                    
                    if self.debug_mode:
                        logger.info(f"[DEBUG] 사용자 조회 응답 데이터: {json.dumps(data, indent=2, ensure_ascii=False)}")
                    
                    if 'data' in data:
                        user_info = data['data']
                        logger.info(f"[SUCCESS] 사용자 발견: @{user_info['username']} (ID: {user_info['id']})")
                        logger.info(f"[INFO] 이름: {user_info['name']}")
                        logger.info(f"[INFO] 팔로워: {user_info.get('public_metrics', {}).get('followers_count', 'N/A'):,}")
                        return user_info['id']
                    else:
                        logger.error(f"[ERROR] 사용자 @{username}을 찾을 수 없습니다")
                        if self.debug_mode:
                            logger.error(f"[DEBUG] 응답 데이터: {data}")
                        return None
                        
                elif response.status_code == 429:
                    logger.warning(f"[WARNING] Rate limit (시도 {attempt + 1}). 16분 대기...")
                    self._safe_wait_for_rate_limit('user_lookup', force_wait=True)
                    continue
                    
                else:
                    logger.error(f"[ERROR] API 에러: {response.status_code} - {response.text}")
                    if attempt < max_retries - 1:
                        logger.info(f"[RETRY] {5}초 후 재시도...")
                        time.sleep(5)
                        continue
                    return None
                    
            except Exception as e:
                logger.error(f"[ERROR] 오류: {e}")
                if attempt < max_retries - 1:
                    logger.info(f"[RETRY] {5}초 후 재시도...")
                    time.sleep(5)
                    continue
                return None
                
        return None
    
    def get_user_tweets_safe(self, user_id: str, max_results: int = 100, 
                            pagination_token: Optional[str] = None) -> Dict:
        """
        Enhanced tweet fetching with automatic 429 recovery
        """
        self._safe_wait_for_rate_limit('tweets_fetch')
        
        url = f"{self.base_url}/users/{user_id}/tweets"
        params = {
            "max_results": min(max_results, 100),
            "tweet.fields": "id,text,created_at,public_metrics,context_annotations,lang,possibly_sensitive,reply_settings",
            "expansions": "author_id",
            "user.fields": "id,name,username,verified"
        }
        
        if pagination_token:
            params["pagination_token"] = pagination_token
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                logger.info(f"[FETCH] 트윗 조회 시도 {attempt + 1}/{max_retries}")
                
                if self.debug_mode:
                    logger.info(f"[DEBUG] 트윗 조회 URL: {url}")
                    logger.info(f"[DEBUG] 트윗 조회 파라미터: {params}")
                
                response = requests.get(url, headers=self.headers, params=params, timeout=30)
                self.last_tweets_fetch = time.time()
                
                if self.debug_mode:
                    logger.info(f"[DEBUG] 트윗 조회 응답 코드: {response.status_code}")
                    logger.info(f"[DEBUG] 트윗 조회 응답 헤더: {dict(response.headers)}")
                
                if response.status_code == 200:
                    response_data = response.json()
                    
                    if self.debug_mode:
                        logger.info(f"[DEBUG] 트윗 조회 전체 응답:")
                        logger.info(f"[DEBUG] {json.dumps(response_data, indent=2, ensure_ascii=False)}")
                        
                        # 데이터 구조 분석
                        if 'data' in response_data:
                            tweets = response_data['data']
                            logger.info(f"[DEBUG] 트윗 개수: {len(tweets)}")
                            for i, tweet in enumerate(tweets[:3]):  # 첫 3개만 샘플 출력
                                logger.info(f"[DEBUG] 트윗 {i+1}: ID={tweet.get('id')}, 텍스트 길이={len(tweet.get('text', ''))}")
                        else:
                            logger.warning(f"[WARNING] 응답에 'data' 필드가 없음!")
                            logger.info(f"[DEBUG] 응답 키들: {list(response_data.keys())}")
                        
                        if 'meta' in response_data:
                            meta = response_data['meta']
                            logger.info(f"[DEBUG] 메타 정보: {meta}")
                        else:
                            logger.warning(f"[WARNING] 응답에 'meta' 필드가 없음!")
                    
                    logger.info("[SUCCESS] 트윗 조회 성공!")
                    return response_data
                    
                elif response.status_code == 429:
                    logger.warning(f"[WARNING] Rate limit 초과 (시도 {attempt + 1}). 16분 강제 대기...")
                    self._safe_wait_for_rate_limit('tweets_fetch', force_wait=True)
                    continue
                    
                else:
                    logger.error(f"[ERROR] API 에러: {response.status_code} - {response.text}")
                    if attempt < max_retries - 1:
                        logger.info(f"[RETRY] {10}초 후 재시도...")
                        time.sleep(10)
                        continue
                    return {}
                    
            except Exception as e:
                logger.error(f"[ERROR] 오류: {e}")
                if attempt < max_retries - 1:
                    logger.info(f"[RETRY] {10}초 후 재시도...")
                    time.sleep(10)
                    continue
                return {}
                
        return {}
    
    def save_tweets_to_json(self, tweets_data: List[Dict], filename: str) -> None:
        """Save tweets data to JSON file"""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(tweets_data, f, ensure_ascii=False, indent=2, default=str)
            logger.info(f"[SAVE] JSON 저장 완료: {filename}")
            logger.info(f"[SAVE] 저장된 트윗 수: {len(tweets_data)}")
        except Exception as e:
            logger.error(f"[ERROR] JSON 저장 실패: {e}")
    
    def save_tweets_to_csv(self, tweets_data: List[Dict], filename: str) -> None:
        """Save tweets data to CSV file"""
        try:
            with open(filename, 'w', newline='', encoding='utf-8') as f:
                if tweets_data:
                    writer = csv.DictWriter(f, fieldnames=tweets_data[0].keys())
                    writer.writeheader()
                    writer.writerows(tweets_data)
            logger.info(f"[SAVE] CSV 저장 완료: {filename}")
            logger.info(f"[SAVE] 저장된 트윗 수: {len(tweets_data)}")
        except Exception as e:
            logger.error(f"[ERROR] CSV 저장 실패: {e}")
    
    def crawl_user_tweets_safe(self, username: str, max_tweets: int = 500) -> List[Dict]:
        """
        안전한 트윗 수집 메인 메서드
        """
        logger.info(f"[START] @{username} 트윗 수집 시작")
        
        # Get user ID
        user_id = self.get_user_id_by_username(username)
        if not user_id:
            logger.error(f"[ERROR] @{username} 사용자 ID 조회 실패")
            return []
        
        all_tweets = []
        pagination_token = None
        fetched_count = 0
        batch_number = 1
        
        while fetched_count < max_tweets:
            logger.info(f"[BATCH] 배치 {batch_number}: 트윗 조회 중... ({fetched_count}/{max_tweets})")
            
            # Calculate batch size
            remaining = max_tweets - fetched_count
            batch_size = min(100, remaining)
            
            # Fetch tweets
            response = self.get_user_tweets_safe(user_id, batch_size, pagination_token)
            
            if not response or 'data' not in response:
                logger.warning("[WARNING] 더 이상 트윗이 없거나 API 오류")
                if self.debug_mode and response:
                    logger.info(f"[DEBUG] 응답에 데이터 없음. 전체 응답: {response}")
                break
            
            # Process tweets
            tweets = response['data']
            
            if self.debug_mode:
                logger.info(f"[DEBUG] 배치 {batch_number}에서 {len(tweets)}개 트윗 처리 중...")
            
            for i, tweet in enumerate(tweets):
                if self.debug_mode and i < 3:  # 첫 3개만 디버그
                    logger.info(f"[DEBUG] 트윗 {i+1} 처리: ID={tweet.get('id')}, 텍스트={tweet.get('text', '')[:50]}...")
                
                tweet_data = {
                    'id': tweet['id'],
                    'text': tweet['text'],
                    'created_at': tweet['created_at'],
                    'retweet_count': tweet.get('public_metrics', {}).get('retweet_count', 0),
                    'like_count': tweet.get('public_metrics', {}).get('like_count', 0),
                    'reply_count': tweet.get('public_metrics', {}).get('reply_count', 0),
                    'quote_count': tweet.get('public_metrics', {}).get('quote_count', 0),
                    'lang': tweet.get('lang', 'unknown'),
                    'possibly_sensitive': tweet.get('possibly_sensitive', False),
                    'username': username,
                    'batch_number': batch_number,
                    'crawled_at': datetime.now(timezone.utc).isoformat()
                }
                all_tweets.append(tweet_data)
            
            fetched_count += len(tweets)
            logger.info(f"[SUCCESS] 배치 {batch_number} 완료: {len(tweets)}개 수집. 총합: {fetched_count}개")
            
            # Check pagination
            if 'meta' in response and 'next_token' in response['meta']:
                pagination_token = response['meta']['next_token']
                batch_number += 1
                
                if fetched_count < max_tweets:
                    logger.info(f"[WAIT] 다음 배치까지 16분 대기... (현재: {fetched_count}/{max_tweets})")
                    self._safe_wait_for_rate_limit('tweets_fetch', force_wait=True)
            else:
                logger.info("[COMPLETE] 모든 트윗 수집 완료 (더 이상 없음)")
                break
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_filename = f"{username}_tweets_safe_{timestamp}.json"
        csv_filename = f"{username}_tweets_safe_{timestamp}.csv"
        
        self.save_tweets_to_json(all_tweets, json_filename)
        self.save_tweets_to_csv(all_tweets, csv_filename)
        
        logger.info(f"[COMPLETE] 수집 완료! 총 {len(all_tweets)}개 트윗 수집됨")
        return all_tweets

def main():
    """
    통합 크롤러 실행
    """
    print("=" * 70)
    print("  X API 트윗 크롤러 - 통합 버전 (Windows 호환 + 디버그)")
    print("=" * 70)
    
    # API Token
    bearer_token = os.getenv('X_BEARER_TOKEN')
    if not bearer_token:
        print("[INPUT] X API Bearer Token이 필요합니다.")
        bearer_token = input("Bearer Token 입력: ").strip()
    
    if not bearer_token:
        print("[ERROR] 토큰 없이는 실행할 수 없습니다.")
        return
    
    # 디버그 모드 선택
    debug_input = input("디버그 모드를 사용하시겠습니까? (y/n): ").strip().lower()
    debug_mode = debug_input in ['y', 'yes', '예', 'ㅇ']
    
    # 설정
    username = "elonmusk"
    max_tweets = 500
    
    print(f"[TARGET] 대상: @{username}")
    print(f"[GOAL] 목표: {max_tweets}개 트윗")
    print(f"[TIME] 예상 시간: 약 {(max_tweets // 100) * 16}분")
    print(f"[MODE] 안전 모드: 16분 간격 + 자동 재시도")
    print(f"[DEBUG] 디버그 모드: {'ON' if debug_mode else 'OFF'}")
    print("=" * 70)
    
    # 실행
    crawler = SafeXTweetsCrawler(bearer_token, debug_mode=debug_mode)
    tweets = crawler.crawl_user_tweets_safe(username, max_tweets)
    
    print("=" * 70)
    if tweets:
        print(f"[SUCCESS] 성공! {len(tweets)}개 트윗 수집 완료")
        print(f"[FILES] 파일: {username}_tweets_safe_*.json/.csv")
        
        # 샘플 출력
        if tweets:
            print(f"[SAMPLE] 첫 번째 트윗:")
            first_tweet = tweets[0]
            print(f"  ID: {first_tweet['id']}")
            print(f"  텍스트: {first_tweet['text'][:100]}...")
            print(f"  좋아요: {first_tweet['like_count']}")
    else:
        print("[FAILED] 수집 실패")

if __name__ == "__main__":
    main() 