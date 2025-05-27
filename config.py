from dotenv import load_dotenv
load_dotenv()

import os

# ── 네이버 검색 API
NAVER_CLIENT_ID     = os.getenv("NAVER_CLIENT_ID")
NAVER_CLIENT_SECRET = os.getenv("NAVER_CLIENT_SECRET")

# ── Gemini API
GENAI_API_KEY       = os.getenv("GENAI_API_KEY")

# ── DB 파일 경로
DB_PATH             = os.getenv("DB_PATH", "news.db")