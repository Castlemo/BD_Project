#!/usr/bin/env python3
"""
RapidAPI - twitter241 엔드포인트를 사용한 트윗 크롤러
"""

import requests
import json
import csv
import os
import time
import logging
from datetime import datetime
from typing import List, Dict, Optional, Any

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rapidapi_crawler.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class RapidAPITweetCrawler:
    """
    RapidAPI의 twitter241 엔드포인트를 사용하여 트윗을 수집하고 CSV로 저장하는 크롤러.
    페이지네이션(cursor)을 처리하여 지정된 개수만큼 트윗을 수집합니다.
    """
    def __init__(self, api_key: str):
        """
        크롤러를 초기화합니다.
        
        Args:
            api_key: RapidAPI에서 발급받은 API 키
        """
        if not api_key:
            raise ValueError("API 키가 제공되지 않았습니다.")
            
        self.api_key = api_key
        self.base_url = "https://twitter241.p.rapidapi.com/user-tweets"
        self.headers = {
            "x-rapidapi-key": self.api_key,
            "x-rapidapi-host": "twitter241.p.rapidapi.com"
        }
        # count를 증가시켜 한번에 더 많은 트윗 요청 (최대 200까지 시도)
        self.count_per_request = 200
        
        # cursor 캐시 및 중복 방지
        self.used_cursors = set()

    def _parse_tweets_from_response(self, response_json: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        API 응답 JSON에서 트윗 데이터를 파싱합니다.
        
        Args:
            response_json: API로부터 받은 JSON 응답
        
        Returns:
            추출된 트윗 데이터 리스트 ({'created_at': ..., 'full_text': ...})
        """
        tweets_data = []
        
        try:
            # 'instructions' 리스트에서 'TimelineAddEntries' 타입의 항목을 찾습니다.
            instructions = response_json.get('result', {}).get('timeline', {}).get('instructions', [])
            
            timeline_entries = []
            for instruction in instructions:
                if instruction.get('type') == 'TimelineAddEntries':
                    timeline_entries = instruction.get('entries', [])
                    break
            
            if not timeline_entries:
                logger.warning("응답에서 'entries'를 찾을 수 없습니다.")
                return []

            for entry in timeline_entries:
                # 'TimelineTweet' 타입의 콘텐츠만 처리
                item_content = entry.get('content', {}).get('itemContent', {})
                if item_content and item_content.get('itemType') == 'TimelineTweet':
                    tweet_results = item_content.get('tweet_results', {})
                    result = tweet_results.get('result', {})
                    
                    # legacy 필드에 실제 데이터가 있습니다.
                    legacy_data = result.get('legacy', {})
                    
                    if legacy_data:
                        created_at = legacy_data.get('created_at', 'N/A')
                        full_text = ""
                        
                        # 리트윗(RT)인 경우 원본 트윗의 full_text를 가져옵니다.
                        # 'retweeted_status_result' 키가 있는지 확인합니다.
                        if 'retweeted_status_result' in legacy_data:
                            # 원본 트윗의 legacy 데이터를 찾습니다.
                            original_tweet_legacy = legacy_data.get('retweeted_status_result', {}).get('result', {}).get('legacy', {})
                            full_text = original_tweet_legacy.get('full_text', '')
                        else:
                            # 일반 트윗은 기존 방식대로 full_text를 가져옵니다.
                            full_text = legacy_data.get('full_text', '')

                        # 줄바꿈 문자를 공백으로 변환하고 양 끝 공백 제거
                        full_text = full_text.replace('\n', ' ').strip()
                        
                        tweets_data.append({
                            'created_at': created_at,
                            'full_text': full_text
                        })
        except (AttributeError, KeyError, IndexError) as e:
            logger.error(f"트윗 데이터 파싱 중 오류 발생: {e}")
            logger.debug(f"오류 발생 지점의 JSON 구조: {json.dumps(response_json, indent=2, ensure_ascii=False)}")
            
        return tweets_data

    def _find_next_cursor(self, response_json: Dict[str, Any]) -> Optional[str]:
        """
        API 응답에서 다음 페이지를 위한 cursor 값을 찾습니다.
        개선된 cursor 파싱으로 더 많은 cursor 타입을 처리합니다.
        
        Args:
            response_json: API로부터 받은 JSON 응답
            
        Returns:
            다음 페이지 cursor 문자열 또는 None
        """
        try:
            instructions = response_json.get('result', {}).get('timeline', {}).get('instructions', [])
            
            # 모든 instruction 타입에서 cursor 찾기
            all_cursors = []
            
            for instruction in instructions:
                # TimelineAddEntries에서 cursor 찾기
                if instruction.get('type') == 'TimelineAddEntries':
                    entries = instruction.get('entries', [])
                    for entry in entries:
                        content = entry.get('content', {})
                        if content.get('entryType') == 'TimelineTimelineCursor':
                            cursor_value = content.get('value')
                            cursor_type = content.get('cursorType', '')
                            
                            if cursor_value and cursor_value not in self.used_cursors:
                                all_cursors.append({
                                    'value': cursor_value,
                                    'type': cursor_type,
                                    'priority': 1 if cursor_type == 'Bottom' else 2
                                })
                
                # TimelineReplaceEntry에서도 cursor 찾기
                elif instruction.get('type') == 'TimelineReplaceEntry':
                    entry = instruction.get('entry', {})
                    content = entry.get('content', {})
                    if content.get('entryType') == 'TimelineTimelineCursor':
                        cursor_value = content.get('value')
                        cursor_type = content.get('cursorType', '')
                        
                        if cursor_value and cursor_value not in self.used_cursors:
                            all_cursors.append({
                                'value': cursor_value,
                                'type': cursor_type,
                                'priority': 1 if cursor_type == 'Bottom' else 2
                            })
            
            # cursor를 우선순위에 따라 정렬 (Bottom이 우선)
            if all_cursors:
                all_cursors.sort(key=lambda x: x['priority'])
                selected_cursor = all_cursors[0]['value']
                self.used_cursors.add(selected_cursor)
                logger.debug(f"선택된 cursor: {selected_cursor[:50]}... (타입: {all_cursors[0]['type']})")
                return selected_cursor
                
        except (AttributeError, KeyError, IndexError) as e:
            logger.error(f"Cursor 파싱 중 오류 발생: {e}")
            
        return None

    def fetch_user_tweets(self, user_id: str, max_tweets: int = 1000):
        """
        특정 사용자의 트윗을 수집하여 CSV 파일로 저장합니다.
        개선된 페이지네이션으로 더 많은 트윗을 효율적으로 수집합니다.
        
        Args:
            user_id: 트윗을 수집할 사용자의 ID
            max_tweets: 수집할 최대 트윗 수
        """
        logger.info(f"사용자 ID {user_id}의 트윗 수집을 시작합니다. 목표: {max_tweets}개")
        logger.info(f"한 번의 요청당 {self.count_per_request}개 트윗 요청")
        
        all_tweets = []
        cursor = None
        request_count = 0
        max_requests = 100  # 무한 루프 방지
        consecutive_empty_responses = 0
        
        # cursor 캐시 초기화
        self.used_cursors.clear()
        
        while len(all_tweets) < max_tweets and request_count < max_requests:
            # count를 동적으로 조정 (남은 트윗 수에 따라)
            remaining_tweets = max_tweets - len(all_tweets)
            current_count = min(self.count_per_request, remaining_tweets)
            
            querystring = {
                "user": user_id,
                "count": str(current_count)
            }
            if cursor:
                querystring["cursor"] = cursor
            
            logger.info(f"API 요청 #{request_count + 1}: {len(all_tweets)} / {max_tweets} 수집됨. Count: {current_count}")
            
            try:
                response = requests.get(self.base_url, headers=self.headers, params=querystring, timeout=45)
                request_count += 1
                
                if response.status_code == 429:  # Rate limit
                    logger.warning("Rate limit에 도달했습니다. 60초 대기...")
                    time.sleep(60)
                    continue
                elif response.status_code != 200:
                    logger.error(f"API 에러: {response.status_code} - {response.text}")
                    if response.status_code >= 500:  # 서버 에러인 경우 재시도
                        logger.info("서버 에러로 인한 10초 후 재시도...")
                        time.sleep(10)
                        continue
                    else:
                        break
                    
                data = response.json()
                
                newly_fetched_tweets = self._parse_tweets_from_response(data)
                
                if not newly_fetched_tweets:
                    consecutive_empty_responses += 1
                    logger.warning(f"이번 응답에서 트윗을 찾을 수 없습니다. ({consecutive_empty_responses}/3)")
                    
                    if consecutive_empty_responses >= 3:
                        logger.info("연속 3회 빈 응답으로 수집을 종료합니다.")
                        break
                else:
                    consecutive_empty_responses = 0
                    logger.info(f"이번 요청에서 {len(newly_fetched_tweets)}개 트윗 수집")
                
                all_tweets.extend(newly_fetched_tweets)
                
                # 중복 제거 (created_at + full_text 기준)
                seen = set()
                unique_tweets = []
                for tweet in all_tweets:
                    tweet_key = (tweet['created_at'], tweet['full_text'])
                    if tweet_key not in seen:
                        seen.add(tweet_key)
                        unique_tweets.append(tweet)
                
                all_tweets = unique_tweets
                logger.info(f"중복 제거 후: {len(all_tweets)}개 트윗")
                
                # 다음 cursor 찾기
                next_cursor = self._find_next_cursor(data)
                if not next_cursor or next_cursor == cursor:
                    logger.info("더 이상 사용 가능한 cursor가 없습니다. 수집을 종료합니다.")
                    break
                
                cursor = next_cursor

                # API rate limit를 고려한 대기 시간 (요청 수에 따라 조정)
                if request_count % 10 == 0:  # 10번째마다 긴 대기
                    wait_time = 5
                else:
                    wait_time = 1
                    
                logger.debug(f"{wait_time}초 대기 중...")
                time.sleep(wait_time)

            except requests.exceptions.Timeout:
                logger.warning("요청 타임아웃. 5초 후 재시도...")
                time.sleep(5)
                continue
            except requests.exceptions.RequestException as e:
                logger.error(f"네트워크 오류 발생: {e}")
                time.sleep(10)
                continue
            except json.JSONDecodeError:
                logger.error("JSON 디코딩 오류. 응답이 올바른 JSON 형식이 아닙니다.")
                time.sleep(5)
                continue

        logger.info(f"총 {len(all_tweets)}개의 트윗을 {request_count}번의 요청으로 수집했습니다.")
        logger.info(f"평균 요청당 트윗 수: {len(all_tweets) / request_count if request_count > 0 else 0:.1f}개")
        
        if all_tweets:
            filename = f"user_{user_id}_tweets_ReTweet.csv"
            self._save_to_csv(all_tweets, filename)
            
    def _save_to_csv(self, tweets_list: List[Dict[str, str]], filename: str):
        """
        수집된 트윗 데이터를 CSV 파일로 저장합니다.
        
        Args:
            tweets_list: 저장할 트윗 데이터 리스트
            filename: 저장할 파일 이름
        """
        try:
            with open(filename, 'w', newline='', encoding='utf-8-sig') as f:
                # 'utf-8-sig'는 Excel에서 한글이 깨지지 않도록 BOM을 추가합니다.
                writer = csv.DictWriter(f, fieldnames=['created_at', 'full_text'])
                writer.writeheader()
                writer.writerows(tweets_list)
            logger.info(f"CSV 파일 저장 완료: {filename}")
        except IOError as e:
            logger.error(f"파일 저장 중 오류 발생: {e}")

def main():
    """
    스크립트 실행을 위한 메인 함수
    """
    print("=" * 70)
    print("  RapidAPI(twitter241) 기반 트윗 크롤러 (개선된 버전)")
    print("=" * 70)
    
    # --- 설정 ---
    # 보안을 위해 API 키는 환경 변수에서 가져오는 것을 권장합니다.
    # 예: api_key = os.getenv("RAPIDAPI_KEY")
    API_KEY = "5fac920861msh988e449f8d91b60p10459bjsnba691d3d2d81" # 사용자 요청에 따라 하드코딩
    USER_ID = "86437069"
    # @WhiteHouse 1879644163769335808
    # @SecScottBessent 1889019333960998912
    # @JDVance 1542228578
    # @marcorubio 15745368
    # @elonmusk 44196397
    MAX_TWEETS = 1000
    
    if not API_KEY:
        print("[ERROR] API 키가 설정되지 않았습니다. 스크립트를 종료합니다.")
        return
        
    print(f"대상 사용자 ID: {USER_ID}")
    print(f"수집 목표 트윗 수: {MAX_TWEETS}")
    print("-" * 70)
    
    crawler = RapidAPITweetCrawler(api_key=API_KEY)
    crawler.fetch_user_tweets(user_id=USER_ID, max_tweets=MAX_TWEETS)
    
    print("=" * 70)
    print("크롤링 작업이 완료되었습니다.")
    print(f"결과는 user_{USER_ID}_tweets_ReTweet.csv 파일에 저장되었습니다.")
    print("=" * 70)


if __name__ == "__main__":
    main() 