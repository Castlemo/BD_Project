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
        # API는 한번에 약 20개의 트윗을 반환하는 것으로 보입니다.
        self.count_per_request = 20

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
        
        Args:
            response_json: API로부터 받은 JSON 응답
            
        Returns:
            다음 페이지 cursor 문자열 또는 None
        """
        try:
            instructions = response_json.get('result', {}).get('timeline', {}).get('instructions', [])
            
            timeline_entries = []
            for instruction in instructions:
                if instruction.get('type') == 'TimelineAddEntries':
                    timeline_entries = instruction.get('entries', [])
                    break

            for entry in timeline_entries:
                if entry.get('content', {}).get('entryType') == 'TimelineTimelineCursor':
                    if entry.get('content', {}).get('cursorType') == 'Bottom':
                        return entry.get('content', {}).get('value')
        except (AttributeError, KeyError, IndexError) as e:
            logger.error(f"Cursor 파싱 중 오류 발생: {e}")
            
        return None

    def fetch_user_tweets(self, user_id: str, max_tweets: int = 1000):
        """
        특정 사용자의 트윗을 수집하여 CSV 파일로 저장합니다.
        
        Args:
            user_id: 트윗을 수집할 사용자의 ID
            max_tweets: 수집할 최대 트윗 수
        """
        logger.info(f"사용자 ID {user_id}의 트윗 수집을 시작합니다. 목표: {max_tweets}개")
        
        all_tweets = []
        cursor = None
        
        while len(all_tweets) < max_tweets:
            querystring = {
                "user": user_id,
                "count": str(self.count_per_request) 
            }
            if cursor:
                querystring["cursor"] = cursor
            
            logger.info(f"API 요청: {len(all_tweets)} / {max_tweets} 수집됨. Cursor: {'있음' if cursor else '없음'}")
            
            try:
                response = requests.get(self.base_url, headers=self.headers, params=querystring, timeout=30)
                
                if response.status_code != 200:
                    logger.error(f"API 에러: {response.status_code} - {response.text}")
                    break
                    
                data = response.json()
                
                newly_fetched_tweets = self._parse_tweets_from_response(data)
                if not newly_fetched_tweets:
                    logger.info("이번 응답에서 더 이상 트윗을 찾을 수 없습니다.")
                    break
                
                all_tweets.extend(newly_fetched_tweets)
                
                cursor = self._find_next_cursor(data)
                if not cursor:
                    logger.info("마지막 페이지에 도달했습니다. 수집을 종료합니다.")
                    break

                # API의 rate limit를 존중하기 위한 약간의 대기 시간
                time.sleep(2)

            except requests.exceptions.RequestException as e:
                logger.error(f"네트워크 오류 발생: {e}")
                time.sleep(10) # 10초 후 재시도
                continue
            except json.JSONDecodeError:
                logger.error("JSON 디코딩 오류. 응답이 올바른 JSON 형식이 아닙니다.")
                break

        logger.info(f"총 {len(all_tweets)}개의 트윗을 수집했습니다.")
        
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
    print("  RapidAPI(twitter241) 기반 트윗 크롤러")
    print("=" * 70)
    
    # --- 설정 ---
    # 보안을 위해 API 키는 환경 변수에서 가져오는 것을 권장합니다.
    # 예: api_key = os.getenv("RAPIDAPI_KEY")
    API_KEY = "5fac920861msh988e449f8d91b60p10459bjsnba691d3d2d81" # 사용자 요청에 따라 하드코딩
    USER_ID = "44196397"
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
    print(f"결과는 user_{USER_ID}_tweets.csv 파일에 저장되었습니다.")
    print("=" * 70)


if __name__ == "__main__":
    main() 