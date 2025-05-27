import urllib.request
import urllib.parse
import json
import requests
from bs4 import BeautifulSoup
import pandas as pd
from collections import deque
import config
import re
import html

# 하루에 몇 건만 가져올지 설정 (현재 3건)
MAX_NEWS = 3

# 탭 이름 ↔ 네이버 검색어 매핑
CATEGORY_QUERIES = {
    "society": "사회",
    "economy": "경제",
    "it":      "AI"
}

def crawl_links_and_dates_titles(category: str):
    """
    1) Naver News Open API를 호출해서
       - 링크(link)
       - 발행일(pubDate)
       - 제목(title)
    를 최대 MAX_NEWS개수만큼 수집합니다.
    """
    query   = urllib.parse.quote(CATEGORY_QUERIES[category])
    display = 100  # 한 번에 가져올 기사 수
    start   = 1    # 검색 시작 인덱스
    links, dates, titles = [], [], []

    while len(links) < MAX_NEWS:
        api_url = (
            f"https://openapi.naver.com/v1/search/news?"
            f"query={query}&display={display}&start={start}&sort=sim"
        )
        req = urllib.request.Request(api_url)
        req.add_header("X-Naver-Client-Id",     config.NAVER_CLIENT_ID)
        req.add_header("X-Naver-Client-Secret", config.NAVER_CLIENT_SECRET)

        with urllib.request.urlopen(req) as res:
            if res.getcode() != 200:
                break
            data = json.loads(res.read().decode("utf-8"))

        # 응답에서 items 추출
        for item in data.get("items", []):
            link    = item.get("link", "")
            pubdate = item.get("pubDate", "")
            raw_title   = item.get("title", "")
            clean_title1 = re.sub(r"</?b>", "", raw_title)
            clean_title = html.unescape(clean_title1)  # HTML 엔티티 복원
            if "news.naver.com" in link:
                links.append(link)
                dates.append(pubdate)
                titles.append(clean_title)
                if len(links) >= MAX_NEWS:
                    break

        start += display
        if start > 1000:  # API 한계 방어
            break

    return links, dates, titles

def fetch_content(link: str) -> str:
    """
    2) 단일 뉴스 페이지로 접속하여
       <div id="newsct_article"> 내의 텍스트만 추출합니다.
    """
    try:
        resp = requests.get(link, timeout=5)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        article = soup.find("div", id="newsct_article")
        if not article:
            return ""

        # 스크립트·스타일 태그 제거
        for tag in article(["script", "style"]):
            tag.decompose()

        return article.get_text(strip=True)

    except Exception:
        return ""

def crawl_news(category: str) -> pd.DataFrame:
    """
    3) 링크 및 발행일, 제목을 가져오고,
       각 링크에서 본문을 크롤링하여 DataFrame으로 반환합니다.
    """
    links, dates, titles = crawl_links_and_dates_titles(category)
    records = []
    for link, date, title in zip(links, dates, titles):
        content = fetch_content(link)
        records.append({
            "link":    link,
            "date":    date,
            "title":   title,
            "content": content
        })
    return pd.DataFrame(records)
