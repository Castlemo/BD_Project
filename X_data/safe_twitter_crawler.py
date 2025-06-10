#!/usr/bin/env python3
"""
Safe X (Twitter) API v2 Tweets Crawler - All-in-One Version
Windows 호환 + 디버그 기능 + 안전한 rate limiting 통합 버전
"""

import requests
import json
import time
import os
from datetime import datetime, timezone, timedelta
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
                            pagination_token: Optional[str] = None, since_id: Optional[str] = None,
                            until_id: Optional[str] = None,
                            start_time: Optional[str] = None, end_time: Optional[str] = None) -> Dict:
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
        
        if since_id:
            params["since_id"] = since_id
        
        if until_id:
            params["until_id"] = until_id
        
        if start_time:
            params["start_time"] = start_time
        
        if end_time:
            params["end_time"] = end_time
        
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
    
    def save_tweets_to_csv(self, tweets_data: List[Dict], filename: str, append_mode: bool = False) -> None:
        """Save tweets data to CSV file"""
        try:
            mode = 'a' if append_mode else 'w'
            with open(filename, mode, newline='', encoding='utf-8') as f:
                if tweets_data:
                    writer = csv.DictWriter(f, fieldnames=tweets_data[0].keys())
                    # 새 파일인 경우에만 헤더 작성
                    if not append_mode:
                        writer.writeheader()
                    writer.writerows(tweets_data)
            
            action = "추가" if append_mode else "생성"
            logger.info(f"[SAVE] CSV {action} 완료: {filename}")
            logger.info(f"[SAVE] 저장된 트윗 수: {len(tweets_data)}")
        except Exception as e:
            logger.error(f"[ERROR] CSV 저장 실패: {e}")
    
    def crawl_user_tweets_safe(self, username: str, max_tweets: int = 100, since_id: Optional[str] = None) -> List[Dict]:
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
            response = self.get_user_tweets_safe(user_id, batch_size, pagination_token, since_id)
            
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
    
    def load_existing_tweets(self, username: str) -> tuple[List[Dict], Optional[str]]:
        """기존 수집된 트윗 파일 로드하고 최신 트윗 ID 반환"""
        import glob
        
        # 기존 JSON 파일들 찾기
        pattern = f"{username}_tweets_safe_*.json"
        files = glob.glob(pattern)
        
        if not files:
            logger.info("[INFO] 기존 파일이 없습니다. 새로 시작합니다.")
            return [], None
        
        # 가장 최근 파일 선택
        latest_file = max(files, key=os.path.getctime)
        logger.info(f"[INFO] 기존 파일 발견: {latest_file}")
        
        try:
            with open(latest_file, 'r', encoding='utf-8') as f:
                existing_tweets = json.load(f)
            
            if existing_tweets:
                # 가장 최신 트윗 ID (첫 번째 트윗이 최신)
                latest_id = existing_tweets[0]['id']
                logger.info(f"[INFO] 기존 트윗 {len(existing_tweets)}개 로드. 최신 ID: {latest_id}")
                return existing_tweets, latest_id
            else:
                logger.info("[INFO] 기존 파일이 비어있습니다.")
                return [], None
                
        except Exception as e:
            logger.error(f"[ERROR] 기존 파일 로드 실패: {e}")
            return [], None
    
    def crawl_user_tweets_incremental(self, username: str, max_new_tweets: int = 100) -> List[Dict]:
        """기존 트윗에 이어서 새로운 트윗만 수집"""
        logger.info(f"[START] @{username} 증분 트윗 수집 시작")
        
        # 기존 트윗 로드
        existing_tweets, latest_id = self.load_existing_tweets(username)
        
        # 새 트윗 수집 (since_id 사용)
        new_tweets = self.crawl_user_tweets_safe(username, max_new_tweets, latest_id)
        
        if new_tweets:
            # 기존 트윗과 합치기 (새 트윗이 앞에)
            all_tweets = new_tweets + existing_tweets
            
            # 중복 제거 (ID 기준)
            seen_ids = set()
            unique_tweets = []
            for tweet in all_tweets:
                if tweet['id'] not in seen_ids:
                    unique_tweets.append(tweet)
                    seen_ids.add(tweet['id'])
            
            # 날짜순 정렬 (최신이 앞에)
            unique_tweets.sort(key=lambda x: x['created_at'], reverse=True)
            
            # 새 파일로 저장
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            json_filename = f"{username}_tweets_safe_{timestamp}.json"
            csv_filename = f"{username}_tweets_safe_{timestamp}.csv"
            
            self.save_tweets_to_json(unique_tweets, json_filename)
            self.save_tweets_to_csv(unique_tweets, csv_filename)
            
            logger.info(f"[COMPLETE] 증분 수집 완료!")
            logger.info(f"[STATS] 새 트윗: {len(new_tweets)}개, 전체: {len(unique_tweets)}개")
            
            return unique_tweets
        else:
            logger.info("[INFO] 새로운 트윗이 없습니다.")
            return existing_tweets
    
    def crawl_tweets_by_date_range(self, username: str, days: int = 10, tweets_per_day: int = 100) -> List[Dict]:
        """날짜별로 트윗 수집 (오늘부터 N일 전까지)"""
        logger.info(f"[START] @{username} 날짜별 트윗 수집 시작")
        logger.info(f"[PLAN] {days}일간 × {tweets_per_day}개/일 = 총 {days * tweets_per_day}개 예상")
        logger.info(f"[TIME] 예상 소요시간: 약 {days * 16}분 ({days * 16 // 60}시간 {days * 16 % 60}분)")
        
        # Get user ID
        user_id = self.get_user_id_by_username(username)
        if not user_id:
            logger.error(f"[ERROR] @{username} 사용자 ID 조회 실패")
            return []
        
        # 파일명 미리 생성 (타임스탬프 고정)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_filename = f"{username}_tweets_daterange_{days}days_{timestamp}.json"
        csv_filename = f"{username}_tweets_daterange_{days}days_{timestamp}.csv"
        
        all_tweets = []
        today = datetime.now(timezone.utc)
        
        # 각 날짜별로 수집
        for day_offset in range(days):
            # 날짜 계산 (오늘부터 거꾸로)
            target_date = today - timedelta(days=day_offset)
            start_time = target_date.replace(hour=0, minute=0, second=0, microsecond=0)
            end_time = target_date.replace(hour=23, minute=59, second=59, microsecond=0)
            
            # RFC3339 형식으로 변환 (X API 호환)
            start_time_str = start_time.strftime("%Y-%m-%dT%H:%M:%SZ")
            end_time_str = end_time.strftime("%Y-%m-%dT%H:%M:%SZ")
            
            date_str = target_date.strftime("%Y-%m-%d")
            logger.info(f"[DAY {day_offset + 1}/{days}] {date_str} 트윗 수집 중...")
            
            if self.debug_mode:
                logger.info(f"[DEBUG] 시간 범위: {start_time_str} ~ {end_time_str}")
            
            # 해당 날짜의 트윗 수집
            response = self.get_user_tweets_safe(
                user_id, 
                tweets_per_day, 
                start_time=start_time_str, 
                end_time=end_time_str
            )
            
            if not response or 'data' not in response:
                logger.warning(f"[WARNING] {date_str}: 트윗이 없거나 API 오류")
                if self.debug_mode and response:
                    logger.info(f"[DEBUG] {date_str} 응답: {response}")
                continue
            
            # 트윗 처리
            daily_tweets = response['data']
            logger.info(f"[SUCCESS] {date_str}: {len(daily_tweets)}개 트윗 수집")
            
            daily_tweet_data = []
            for i, tweet in enumerate(daily_tweets):
                if self.debug_mode and i < 2:  # 각 날짜별로 2개씩만 디버그
                    logger.info(f"[DEBUG] {date_str} 트윗 {i+1}: ID={tweet.get('id')}")
                
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
                    'collection_date': date_str,
                    'day_number': day_offset + 1,
                    'crawled_at': datetime.now(timezone.utc).isoformat()
                }
                daily_tweet_data.append(tweet_data)
                all_tweets.append(tweet_data)
            
            # 매 100개씩 즉시 CSV에 저장
            if daily_tweet_data:
                append_mode = day_offset > 0  # 첫 번째 날은 새 파일, 나머지는 추가
                self.save_tweets_to_csv(daily_tweet_data, csv_filename, append_mode)
                logger.info(f"[BATCH] {date_str}: CSV 파일에 {len(daily_tweet_data)}개 저장 완료")
            
            # 다음 날짜 수집 전 대기 (마지막 날짜가 아닌 경우)
            if day_offset < days - 1:
                logger.info(f"[WAIT] 다음 날짜 수집까지 16분 대기... ({day_offset + 1}/{days} 완료)")
                self._safe_wait_for_rate_limit('tweets_fetch', force_wait=True)
        
        # 전체 JSON 파일만 저장 (CSV는 이미 저장됨)
        self.save_tweets_to_json(all_tweets, json_filename)
        
        logger.info(f"[COMPLETE] 날짜별 수집 완료!")
        logger.info(f"[STATS] 총 {len(all_tweets)}개 트윗 수집 ({days}일간)")
        logger.info(f"[FILES] 저장 파일: {json_filename}, {csv_filename}")
        logger.info(f"[NOTE] CSV는 매 100개씩 실시간 저장 완료")
        
        return all_tweets
    
    def crawl_tweets_sequential(self, username: str, total_batches: int = 10, tweets_per_batch: int = 100) -> List[Dict]:
        """최신 트윗부터 과거로 거슬러 올라가며 연속 수집 (중복 없음)"""
        logger.info(f"[START] @{username} 연속 트윗 수집 시작")
        logger.info(f"[PLAN] {total_batches}회 × {tweets_per_batch}개/회 = 총 {total_batches * tweets_per_batch}개 예상")
        logger.info(f"[TIME] 예상 소요시간: 약 {total_batches * 16}분 ({total_batches * 16 // 60}시간 {total_batches * 16 % 60}분)")
        
        # Get user ID
        user_id = self.get_user_id_by_username(username)
        if not user_id:
            logger.error(f"[ERROR] @{username} 사용자 ID 조회 실패")
            return []
        
        # 파일명 미리 생성 (타임스탬프 고정)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_filename = f"{username}_tweets_sequential_{total_batches}batches_{timestamp}.json"
        csv_filename = f"{username}_tweets_sequential_{total_batches}batches_{timestamp}.csv"
        
        all_tweets = []
        until_id = None  # 첫 번째는 최신 트윗부터
        
        # 각 배치별로 수집
        for batch_num in range(total_batches):
            logger.info(f"[BATCH {batch_num + 1}/{total_batches}] 트윗 수집 중...")
            
            if until_id:
                logger.info(f"[INFO] 기준 ID: {until_id} 이전 트윗들 수집")
            else:
                logger.info(f"[INFO] 최신 트윗부터 수집")
            
            if self.debug_mode:
                logger.info(f"[DEBUG] until_id: {until_id}")
            
            # 트윗 수집
            response = self.get_user_tweets_safe(
                user_id, 
                tweets_per_batch, 
                until_id=until_id
            )
            
            if not response or 'data' not in response:
                logger.warning(f"[WARNING] 배치 {batch_num + 1}: 트윗이 없거나 API 오류")
                if self.debug_mode and response:
                    logger.info(f"[DEBUG] 배치 {batch_num + 1} 응답: {response}")
                break
            
            # 트윗 처리
            batch_tweets = response['data']
            logger.info(f"[SUCCESS] 배치 {batch_num + 1}: {len(batch_tweets)}개 트윗 수집")
            
            batch_tweet_data = []
            for i, tweet in enumerate(batch_tweets):
                if self.debug_mode and i < 2:  # 각 배치별로 2개씩만 디버그
                    logger.info(f"[DEBUG] 배치 {batch_num + 1} 트윗 {i+1}: ID={tweet.get('id')}")
                
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
                    'batch_number': batch_num + 1,
                    'crawled_at': datetime.now(timezone.utc).isoformat()
                }
                batch_tweet_data.append(tweet_data)
                all_tweets.append(tweet_data)
            
            # 매 100개씩 즉시 CSV에 저장
            if batch_tweet_data:
                append_mode = batch_num > 0  # 첫 번째 배치는 새 파일, 나머지는 추가
                self.save_tweets_to_csv(batch_tweet_data, csv_filename, append_mode)
                logger.info(f"[BATCH] 배치 {batch_num + 1}: CSV 파일에 {len(batch_tweet_data)}개 저장 완료")
            
            # 다음 배치를 위한 until_id 설정 (가장 오래된 트윗 ID)
            if batch_tweets:
                until_id = batch_tweets[-1]['id']  # 마지막 트윗이 가장 오래된 트윗
                logger.info(f"[INFO] 다음 배치 기준 ID: {until_id}")
            
            # 다음 배치 수집 전 대기 (마지막 배치가 아닌 경우)
            if batch_num < total_batches - 1:
                logger.info(f"[WAIT] 다음 배치까지 16분 대기... ({batch_num + 1}/{total_batches} 완료)")
                self._safe_wait_for_rate_limit('tweets_fetch', force_wait=True)
        
        # 전체 JSON 파일만 저장 (CSV는 이미 저장됨)
        self.save_tweets_to_json(all_tweets, json_filename)
        
        logger.info(f"[COMPLETE] 연속 수집 완료!")
        logger.info(f"[STATS] 총 {len(all_tweets)}개 트윗 수집 ({total_batches}배치)")
        logger.info(f"[FILES] 저장 파일: {json_filename}, {csv_filename}")
        logger.info(f"[NOTE] CSV는 매 100개씩 실시간 저장 완료")
        
        return all_tweets

def main():
    """
    연속 트윗 크롤러 실행
    """
    print("=" * 70)
    print("  X API 연속 트윗 크롤러 (최신→과거, 10배치 × 100개 = 1000개)")
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
    
    # 설정 (연속 수집 전용)
    username = "elonmusk"
    total_batches = 10
    tweets_per_batch = 100
    total_tweets = total_batches * tweets_per_batch
    estimated_time = total_batches * 16
    
    print(f"[METHOD] 연속 수집 (최신 트윗부터 과거로 거슬러 올라가며)")
    print(f"[FLOW] 배치1(최신 100개) → 배치2(그 이전 100개) → ... → 배치10")
    
    print(f"[TARGET] 대상: @{username}")
    print(f"[GOAL] 목표: {total_tweets}개 트윗")
    print(f"[TIME] 예상 시간: 약 {estimated_time}분 ({estimated_time // 60}시간 {estimated_time % 60}분)")
    print(f"[MODE] 안전 모드: 16분 간격 + 자동 재시도")
    print(f"[DEBUG] 디버그 모드: {'ON' if debug_mode else 'OFF'}")
    print("=" * 70)
    
    # 실행 (연속 수집 전용)
    crawler = SafeXTweetsCrawler(bearer_token, debug_mode=debug_mode)
    tweets = crawler.crawl_tweets_sequential(username, total_batches, tweets_per_batch)
    
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