#!/bin/bash
# RAG Compare POC 서버 시작 스크립트

cd "$(dirname "$0")"

# .env 파일 확인
if [ ! -f .env ]; then
  echo "[!] .env 파일이 없습니다. .env.example을 복사하여 수정하세요."
  echo "    cp .env.example .env"
  exit 1
fi

# 가상환경 확인
if [ ! -d .venv ]; then
  echo "[*] 가상환경 생성 중..."
  python3 -m venv .venv
fi

source .venv/bin/activate

# 패키지 설치
echo "[*] 패키지 설치 확인 중..."
pip install -q -r requirements.txt

# 샘플 데이터 로드 (DB가 비어있는 경우)
DB_PATH=$(grep SQLITE_DB_PATH .env | cut -d= -f2 | tr -d ' ')
DB_PATH=${DB_PATH:-./data/poc.db}

if [ ! -f "$DB_PATH" ]; then
  echo "[*] 샘플 데이터 로드 중..."
  python seed_data.py
fi

# 서버 시작
echo "[*] 서버 시작: http://0.0.0.0:8000"
echo "[*] 외부 접속: http://$(hostname -I | awk '{print $1}'):8000"
echo ""
PYTHONPATH="$(pwd)" uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
