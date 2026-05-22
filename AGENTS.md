# AGENTS.md

## 기본 원칙

- 이 프로젝트에서 Agent는 항상 한국어로 대화한다.
- 사용자가 영어로 요청하거나 영어 표현을 섞어 쓰더라도, 답변과 중간 진행 설명은 한국어로 작성한다.
- 이 프로젝트의 문서 작성, 검토, 요약, 수정 제안은 기본적으로 한국어로 작성한다.
- 코드, 파일명, 디렉터리명, 기술 약어, 표준명, 제품명, API 이름처럼 원문 유지가 필요한 표현은 그대로 사용할 수 있다.

## 적용 범위

- 이 파일은 `2026_architect_9B_team` 프로젝트에만 적용한다.
- 다른 프로젝트나 상위/인접 디렉터리의 `AGENTS.md`는 이 프로젝트 작업에 적용하지 않는다.
- 특히 `../Archi-Pers-Data/AGENTS.md`의 지침은 이 프로젝트에서 무시한다.
- `doc/readme.md`는 현재 팀 Architecture 문서 범위에서 제외한다.
- `doc/RAPTOR_Hierarchical_RAG.pptx`는 현재 팀 Architecture 문서 범위에서 제외한다.
- 사용자가 별도로 요청하지 않는 한, 문서 검토와 수정은 `doc/01_*.md`, `doc/02_*.md`, `doc/03_*.md`, `doc/04_*.md`를 대상으로 한다.

## 작업 방식

- 사용자의 요청이 문서 검토이면, 먼저 관련 문서 구조를 확인하고 현재 문서의 목적, 구성, 누락, 불일치, 개선점을 한국어로 정리한다.
- 문서를 수정할 때는 기존 문서의 의도와 표현 방식을 최대한 유지한다.
- 확정되지 않은 내용은 추측으로 단정하지 않고 `TBD`, `TODO`, `Open Question`, `Evidence Needed` 등으로 표시한다.
- 사용자가 명시적으로 요청하지 않은 대규모 구조 변경은 피한다.
- 변경 전에는 어떤 파일을 어떤 방향으로 수정할지 간단히 한국어로 설명한다.

## 문서 작성 기준

- 문장은 발표나 리뷰에 바로 활용할 수 있도록 명확하고 보수적으로 작성한다.
- 설계 근거, 요구사항, 품질속성, Design Point, Architecture Decision은 서로 추적 가능하도록 정리한다.
- 문서 간 용어, ID, 상태, 링크가 서로 어긋나지 않도록 확인한다.
- 외부 근거가 필요한 내용은 출처가 확인되기 전까지 확정 정보처럼 쓰지 않는다.

## HTML / PPT 백데이터 동기화 기준

- `doc/01_Project_Overview_Requirements_Overall_Architecture.md`, `doc/02_DP1_Dedup_Aware_RAG_Selection.md`, `doc/03_DP2_Permission_Aware_Dataset_Strategy.md`, `doc/04_DP3_Knowledge_Access_Strategy.md` 중 하나라도 업데이트되면 `docs/html/ppt_content_pages.html`도 같은 작업 범위에서 함께 업데이트한다.
- Markdown 문서의 요구사항, QA, KPI, Design Point, 선택안, Trade-off, References, PPT 필수 포함 포인트가 바뀌면 HTML의 해당 페이지 탭과 표/다이어그램/발표자용 문구를 함께 맞춘다.
- HTML에서 참조하는 시각 자료가 변경 필요하면 `docs/html/assets/` 아래 SVG 또는 기타 asset도 함께 업데이트한다.
- Mermaid 다이어그램은 `docs/html/assets/mermaid.min.js`를 사용하는 로컬 렌더링 방식을 유지한다.
- `docs/html/ppt_content_pages.html`은 최종 PPT가 아니라 PPT 제작을 위한 백데이터/페이지별 초안이므로, 정보가 다소 밀도 있게 들어가도 된다.
- `doc/readme.md`와 `doc/RAPTOR_Hierarchical_RAG.pptx`는 사용자가 별도로 요청하지 않는 한 HTML 동기화 대상에서 제외한다.

## 응답 기준

- 최종 답변은 작업 결과, 변경 파일, 확인한 사항을 간결하게 설명한다.
- 테스트나 검증을 실행하지 못했으면 그 사실을 명확히 말한다.
- 사용자가 추가 검토를 원할 경우 바로 이어서 작업할 수 있도록 다음 단계 후보를 짧게 제안한다.
