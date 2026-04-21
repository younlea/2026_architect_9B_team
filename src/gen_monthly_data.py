"""
월간 대화 데이터 생성기
AI 에이전트가 5명의 페르소나로 약 30일치 대화를 생성합니다.
각 세션은 DB에 즉시 저장됩니다.

실행: cd src && python gen_monthly_data.py
"""
import sys, os, uuid, json
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(__file__))
os.chdir(os.path.dirname(__file__))

from dotenv import load_dotenv
load_dotenv()

from backend.db.database import init_db, get_conn
from backend.rag.llm_client import get_llm_answer

# ── 페르소나 정의 ──────────────────────────────────────────────────────────────
PERSONAS = {
    "김재원": {
        "role": "신입 백엔드 개발자 (입사 1개월차)",
        "age": 25,
        "background": "컴퓨터공학 전공 / Python·FastAPI 경험 1년 / 반려묘 '나비' 키움 / 집은 마포구",
        "personality": "열정적이고 질문이 많음. 실수를 두려워하지만 극복 의지 강함. 점심은 항상 편의점 도시락",
        "speaking": "공손한 존댓말, 가끔 '헉', '대박' 같은 감탄사 사용, 배운 것에 흥분",
    },
    "이현수": {
        "role": "시니어 개발자 / 멘토",
        "age": 32,
        "background": "개발 경력 7년 / RAG·LLM 전문 / RAPTOR 논문 구현 경험 / 강아지 '몽이' 키움",
        "personality": "차분하고 친절, 설명을 좋아함, 가끔 아재 개그, 코드 품질에 깐깐",
        "speaking": "반말과 존댓말 혼용(상황에 따라), 기술적 설명 상세, 예시 들기 좋아함",
    },
    "박민지": {
        "role": "팀장 / PM 출신",
        "age": 38,
        "background": "개발 경력 12년 / PM→개발리드 전환 / 비즈니스 마인드 강함 / 마라톤 취미",
        "personality": "결단력 있고 명확, 팀 분위기 중시, 데드라인에 민감, 칭찬을 잊지 않음",
        "speaking": "간결하고 명확, 미팅 진행 스타일, 불필요한 말 없음",
    },
    "최지훈": {
        "role": "AI 연구원",
        "age": 29,
        "background": "박사과정 중퇴 / RAPTOR·GraphRAG 논문 연구 / arXiv 매일 읽음 / 커피 중독",
        "personality": "이론 중심, 논문 인용 즐김, 약간 내성적이지만 기술 토론엔 적극적",
        "speaking": "전문 용어 많이 사용, 논리적이고 구조적, 가끔 논문 제목 언급",
    },
    "정수진": {
        "role": "프론트엔드 개발자",
        "age": 27,
        "background": "UX 디자인 전공 후 개발 전환 / React 전문 / 사용자 관점 중시 / 카페 투어 취미",
        "personality": "창의적이고 감성적, 사용자 경험 집착, 밝고 유머러스",
        "speaking": "캐주얼, 이모지·줄임말 사용, 디자인 관점 발언 많음",
    },
}

# ── 30일 시나리오 (주 5일 × 6주) ─────────────────────────────────────────────
DAILY_SCENARIOS = [
    # Week 1: 온보딩
    {"day": 1,  "title": "첫 출근 - 팀 소개와 환경 세팅",       "speakers": ["이현수","김재원"],             "mode": "2person",
     "context": "김재원의 첫 출근. 이현수가 개발환경 세팅을 도와주며 팀을 소개한다. 사용 기술 스택(Python/FastAPI/ChromaDB/LangChain)을 설명하고 GitHub 접근 권한을 준다."},
    {"day": 1,  "title": "첫날 팀 점심 - 자기소개 시간",         "speakers": ["박민지","이현수","김재원","정수진","최지훈"], "mode": "group",
     "context": "팀 전체가 점심 식사. 각자 자기소개와 함께 현재 진행 중인 RAG 비교 프로젝트에 대해 간략히 소개. 김재원은 프로젝트에 기대감을 표현."},
    {"day": 2,  "title": "Git 워크플로우와 코드 컨벤션 학습",    "speakers": ["이현수","김재원"],             "mode": "2person",
     "context": "이현수가 브랜치 전략(feature/fix/hotfix), PR 규칙, 코드 리뷰 기준을 설명. 김재원은 노트 필기하며 질문 폭발. 첫 브랜치 생성까지 완료."},
    {"day": 3,  "title": "기존 코드베이스 리뷰 - BasicRAG 분석", "speakers": ["최지훈","김재원"],             "mode": "2person",
     "context": "최지훈이 현재 Basic RAG 파이프라인 코드를 설명. 청킹 전략, 임베딩 모델 선택 이유, ChromaDB 사용 이유를 설명. 김재원은 왜 RAPTOR가 더 나은지 질문."},
    {"day": 4,  "title": "주간 팀 스탠드업 - 프로젝트 현황",     "speakers": ["박민지","이현수","최지훈","정수진","김재원"], "mode": "group",
     "context": "주간 스탠드업. 각자 이번 주 완료 사항과 다음 주 계획 공유. 박민지가 데모 일정(3주 후)을 공지. 김재원은 온보딩 진행 상황 보고."},
    {"day": 5,  "title": "퇴근 후 1:1 - 첫 주 회고",            "speakers": ["이현수","김재원"],             "mode": "2person",
     "context": "금요일 오후, 이현수와 김재원이 첫 주를 돌아봄. 김재원이 어려웠던 점(Docker 설정, ChromaDB API)을 이야기하고 이현수가 조언. 주말 공부할 것 추천."},

    # Week 2: 첫 실제 개발
    {"day": 8,  "title": "월요일 아침 - RAPTOR 개념 특강",       "speakers": ["최지훈","김재원"],             "mode": "2person",
     "context": "최지훈이 RAPTOR 논문(Sarthi et al., 2024)을 설명. 재귀 요약, 클러스터링, 트리 검색의 원리. 김재원이 Basic RAG와 차이를 이해하기 시작. K-Means vs GMM 논의."},
    {"day": 8,  "title": "오후 - 첫 이슈 할당과 분석",           "speakers": ["이현수","김재원"],             "mode": "2person",
     "context": "이현수가 김재원에게 첫 이슈를 할당: 세션 인덱싱 시 진행률 표시 기능 추가. 요구사항 분석 방법, 어떤 파일을 수정해야 하는지 함께 파악."},
    {"day": 9,  "title": "구현 중 막힌 부분 - asyncio 질문",     "speakers": ["이현수","김재원"],             "mode": "2person",
     "context": "김재원이 FastAPI에서 asyncio와 동기 함수 혼용 문제로 막힘. run_in_executor 사용법을 이현수가 설명. 실제 코드로 예시 보여줌."},
    {"day": 10, "title": "첫 PR 코드 리뷰",                     "speakers": ["이현수","최지훈","김재원"],     "mode": "group",
     "context": "김재원의 첫 PR에 이현수와 최지훈이 리뷰. 변수명 컨벤션 지적, 에러 핸들링 누락 발견, 좋은 점도 칭찬. 김재원이 피드백 반영 계획 설명."},
    {"day": 11, "title": "점심 - 개인 근황 토크",               "speakers": ["정수진","김재원"],             "mode": "2person",
     "context": "정수진과 김재원이 점심 먹으며 개인 이야기. 김재원 고양이 나비 이야기, 정수진 카페 투어 취미, 서울 살이 적응, 월세 이야기, 회사 복지 이야기."},
    {"day": 12, "title": "PR 머지 - 첫 기능 완성!",             "speakers": ["이현수","김재원","박민지"],     "mode": "group",
     "context": "김재원의 첫 PR이 머지됨. 이현수와 박민지가 축하. 작은 기능이지만 의미있는 첫 기여. 김재원이 다음 이슈로 넘어갈 준비 표명."},

    # Week 3: 팀 프로젝트 본격 참여
    {"day": 15, "title": "RAPTOR 인덱싱 성능 이슈 발견",         "speakers": ["최지훈","이현수","김재원"],     "mode": "group",
     "context": "긴 대화(500+ 메시지) 인덱싱 시 메모리 부족 이슈 발견. 최지훈이 배치 처리 방안 제안. 이현수와 김재원이 구체적인 구현 방식 논의. ChromaDB 제한사항 파악."},
    {"day": 15, "title": "오후 - 멀티모델 지원 논의",            "speakers": ["박민지","최지훈","이현수"],     "mode": "group",
     "context": "박민지가 클라이언트 요청사항 공유: Ollama gemma, qwen 등 여러 모델로 RAG 답변 품질 비교 기능 필요. 최지훈이 아키텍처 방안 제시. 이현수가 일정 산정."},
    {"day": 16, "title": "UI 설계 논의 - 멀티모델 비교 화면",    "speakers": ["정수진","박민지","김재원"],     "mode": "group",
     "context": "정수진이 멀티모델 비교 UI 와이어프레임 공유. 4패널 레이아웃 vs 탭 방식 토론. 김재원이 API 응답 구조 설명. 최종적으로 탭+패널 혼합 방식 채택."},
    {"day": 17, "title": "성능 테스트 - llama3 vs gemma2",       "speakers": ["최지훈","이현수","김재원"],     "mode": "group",
     "context": "llama3와 gemma2:9b 모델로 같은 질문에 대한 RAG 답변 품질 비교 테스트. 응답 시간, 정확도, 한국어 품질 비교. gemma2가 한국어에서 미묘하게 나음을 발견."},
    {"day": 18, "title": "프로덕션 버그 - 인덱싱 중 서버 크래시","speakers": ["이현수","김재원"],             "mode": "2person",
     "context": "RAPTOR 인덱싱 도중 서버가 크래시되는 버그 발생. 이현수와 김재원이 함께 로그 분석. 원인: K-Means에 빈 클러스터 예외처리 누락. 함께 핫픽스 작성."},
    {"day": 19, "title": "팀 회식 - 치킨집",                    "speakers": ["박민지","이현수","최지훈","정수진","김재원"], "mode": "group",
     "context": "팀 회식. 맥주와 치킨. 각자 회사 다니기 전 에피소드, 개발자 밈, 프로젝트 썰. 김재원이 처음으로 편하게 웃으며 팀에 녹아듦. 박민지가 중간 성과 칭찬."},

    # Week 4: 성장과 책임
    {"day": 22, "title": "월간 리뷰 준비 - KPI 정리",           "speakers": ["박민지","이현수"],             "mode": "2person",
     "context": "박민지와 이현수가 월간 KPI 리뷰 준비. Basic RAG vs RAPTOR 비교 수치 정리. 응답 시간 개선율(20%), 정확도 개선율(35%) 데이터 확인. 경영진 보고 방향 논의."},
    {"day": 22, "title": "김재원 1개월 성과 면담",              "speakers": ["박민지","김재원"],             "mode": "2person",
     "context": "박민지와 김재원의 1개월 면담. 잘한 점(적극적 질문, 빠른 학습), 개선할 점(문서화 습관, 테스트 코드 작성). 2개월차 목표 설정. 김재원 감사 표현."},
    {"day": 23, "title": "데이터셋 선정 논의 - Locomo vs MSC",  "speakers": ["최지훈","이현수","김재원"],     "mode": "group",
     "context": "POC 평가에 쓸 데이터셋 선정 논의. 최지훈이 MSC(다중세션), Locomo(장기기억), LongBench(긴문서) 특징 설명. 목적에 따른 선택 기준 논의. MSC로 최종 결정."},
    {"day": 24, "title": "데모 준비 - 시연 스크립트 작성",       "speakers": ["박민지","정수진","이현수"],     "mode": "group",
     "context": "3일 후 경영진 데모 준비. 시연 순서 정하기, 예상 질문 대비, 어떤 질문으로 RAG 차이를 극적으로 보여줄지 논의. 정수진이 UI 최종 점검."},
    {"day": 25, "title": "경영진 데모 데이",                    "speakers": ["박민지","이현수","최지훈","정수진","김재원"], "mode": "group",
     "context": "경영진 앞 데모. 긴장된 시작이지만 순조롭게 진행. Basic RAG vs RAPTOR 비교에서 RAPTOR가 복잡한 질문에서 확연히 나은 결과를 보여줌. 경영진 긍정적 반응."},
    {"day": 26, "title": "데모 후 회고",                        "speakers": ["박민지","이현수","최지훈","정수진","김재원"], "mode": "group",
     "context": "데모 직후 팀 회고. 잘된 점: RAPTOR 결과 차이가 명확히 드러남. 아쉬운 점: 멀티모델 비교 화면 로딩 느림. 다음 스프린트 백로그 정리."},

    # Week 5: 심화와 자립
    {"day": 29, "title": "qwen3:8b 모델 추가 테스트",           "speakers": ["최지훈","김재원"],             "mode": "2person",
     "context": "Ollama qwen3:8b 모델을 멀티모델 비교에 추가. 한국어 처리 능력 비교. qwen3가 긴 답변 생성에 강점, llama3가 짧고 정확한 답변에 강점 확인."},
    {"day": 30, "title": "1개월 기술 회고 - RAG 인사이트",       "speakers": ["최지훈","이현수","김재원"],     "mode": "group",
     "context": "1개월간 RAG 비교 실험 인사이트 정리. RAPTOR가 전체 맥락 파악에 우월, Basic RAG가 단순 팩트 검색에는 빠름. 한국어 데이터셋 부재 문제 논의. 논문 작성 가능성 언급."},
    {"day": 30, "title": "퇴근 전 - 김재원과 이현수 개인 대화",  "speakers": ["이현수","김재원"],             "mode": "2person",
     "context": "퇴근 전 1:1. 김재원이 한 달 동안 얼마나 성장했는지 이야기. 나비 근황, 이현수 몽이 이야기, 다음 달 목표(GraphRAG 스터디), 서로 격려."},
    {"day": 31, "title": "다음달 계획 수립 - 스프린트 플래닝",   "speakers": ["박민지","이현수","최지훈","정수진","김재원"], "mode": "group",
     "context": "2개월차 스프린트 플래닝. GraphRAG 도입 검토, 한국어 데이터셋 자체 구축 방안, 실제 서비스 배포 준비. 김재원에게 GraphRAG 문서화 태스크 할당."},
]

# ── 대화 생성 ─────────────────────────────────────────────────────────────────
BASE_DATE = datetime(2026, 3, 2)  # 월요일 기준 시작일


def persona_desc(name: str) -> str:
    p = PERSONAS[name]
    return f"{name}({p['role']}, {p['age']}세): 성격={p['personality']} / 말투={p['speaking']}"


def build_prompt(scenario: dict) -> str:
    speakers = scenario["speakers"]
    persona_block = "\n".join(persona_desc(s) for s in speakers)
    turns_per_speaker = 8 if len(speakers) == 2 else 5
    total_turns = turns_per_speaker * len(speakers)

    return f"""[중요] 반드시 한국어로만 작성하세요. Do NOT use English at all.

당신은 현실감 있는 한국어 대화 시나리오 작가입니다.
아래 인물들의 한국어 대화를 작성해 주세요.

## 등장인물
{persona_block}

## 오늘의 상황
{scenario['context']}

## 작성 규칙 (반드시 준수)
1. 모든 대화는 반드시 한국어로 작성합니다.
2. 총 {total_turns}줄 이상 작성합니다.
3. 각 인물의 말투를 개성 있게 표현합니다.
4. 기술 용어는 한국어 설명과 함께 씁니다.
5. 줄마다 정확히 다음 형식으로만 작성합니다 (다른 형식 금지):
   화자이름: 대화내용

예시:
이현수: 안녕하세요, 오늘 환경 세팅부터 시작해 볼까요?
김재원: 네! 많이 기대됩니다. 어디서부터 시작하면 될까요?

[지금 바로 한국어로 시작하세요]:
"""


def parse_dialog(raw: str, speakers: list[str]) -> list[dict]:
    messages = []
    for line in raw.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        for sp in speakers:
            if line.startswith(f"{sp}:"):
                content = line[len(sp) + 1:].strip()
                if content:
                    messages.append({"speaker": sp, "content": content})
                break
        else:
            if messages and line and not any(line.startswith(f"{s}:") for s in PERSONAS):
                messages[-1]["content"] += " " + line
    return messages


def generate_session(scenario: dict, model: str = "llama3") -> str:
    prompt = build_prompt(scenario)
    print(f"    LLM 생성 중 (모델: {model})...", end="", flush=True)
    raw = get_llm_answer(prompt, model)
    messages = parse_dialog(raw, scenario["speakers"])
    print(f" {len(messages)}개 메시지 생성됨")

    day_offset = scenario["day"] - 1
    session_date = BASE_DATE + timedelta(days=day_offset)
    session_id = str(uuid.uuid4())

    with get_conn() as conn:
        conn.execute(
            "INSERT INTO sessions (id, title, mode, created_at) VALUES (?, ?, ?, ?)",
            (session_id, f"[Day{scenario['day']}] {scenario['title']}",
             scenario["mode"], session_date.isoformat()),
        )
        for i, msg in enumerate(messages):
            ts = (session_date + timedelta(minutes=i * 2)).isoformat()
            conn.execute(
                "INSERT INTO messages (session_id, speaker, content, timestamp) VALUES (?, ?, ?, ?)",
                (session_id, msg["speaker"], msg["content"], ts),
            )

    return session_id


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "llama3"

    print("=" * 60)
    print("월간 대화 데이터 생성기")
    print(f"LLM 모델: {model}")
    print(f"총 시나리오: {len(DAILY_SCENARIOS)}개")
    print("=" * 60)

    init_db()
    generated = []
    failed = []

    for i, scenario in enumerate(DAILY_SCENARIOS, 1):
        print(f"\n[{i:02d}/{len(DAILY_SCENARIOS)}] Day {scenario['day']}: {scenario['title']}")
        print(f"    화자: {', '.join(scenario['speakers'])}")
        try:
            sid = generate_session(scenario, model)
            generated.append(sid)
            print(f"    저장 완료 (session_id: {sid[:8]}...)")
        except Exception as e:
            print(f"    오류: {e}")
            failed.append(scenario["title"])

    print("\n" + "=" * 60)
    print(f"완료: {len(generated)}개 세션 생성")
    if failed:
        print(f"실패: {len(failed)}개 → {failed}")

    # 생성 결과 요약 JSON 저장
    summary = {
        "generated_at": datetime.now().isoformat(),
        "model": model,
        "total": len(generated),
        "session_ids": generated,
        "failed": failed,
    }
    with open("data/monthly_gen_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"요약 저장: data/monthly_gen_summary.json")


if __name__ == "__main__":
    main()
