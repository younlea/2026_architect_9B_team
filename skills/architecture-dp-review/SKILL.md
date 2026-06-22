---
name: architecture-dp-review
description: Generate a Korean senior-architect review report for this repository's architecture assignment Decision Point documents. Use when asked to evaluate, review, rank, or improve DP documents, architecture Markdown files, or PPT HTML backing data in `doc/01_*.md` through `doc/04_*.md` and `docs/html/ppt_content_pages.html`, including cross-document consistency, DP importance, candidate suitability, reviewer Q&A risk, and fix priorities.
---

# Architecture Assignment Review Report Generator

Purpose

This skill analyzes architecture assignment materials and PPT-planning documents, then generates a senior-architect-style review report.

The target use case is an internal SW Architect training assignment where the user provides architecture design documents and presentation planning materials, such as:

- Architecture overview documents
- Decision Point documents
- Architecture Decision Records
- Markdown files
- HTML files generated from architecture docs
- PPT page planning HTML
- PPT content blueprints
- Trade-off tables
- QA sections
- Evaluation notes
- GitHub repository paths or local project paths

The skill must produce a structured review report from the perspective of a senior architect and review committee member.

The report must not merely summarize the documents. It must evaluate architectural validity, decision-point importance, decision framing, candidate-option suitability, cross-DP consistency, reviewer risks, and DP priority.

---

Project-Specific Required Inputs

When this skill is used inside the `2026_architect_9B_team` repository, read these files first unless the user explicitly narrows the scope:

- `doc/01_Project_Overview_Requirements_Overall_Architecture.md`
- `doc/02_DP1_Dedup_Aware_RAG_Selection.md`
- `doc/03_DP2_Permission_Aware_Dataset_Strategy.md`
- `doc/04_DP3_Knowledge_Access_Strategy.md`
- `docs/html/ppt_content_pages.html`

Use the Markdown files as the primary architecture source and use `docs/html/ppt_content_pages.html` as the PPT/page-planning backing data. Check whether the HTML pages preserve the same DP topics, candidates, selected decisions, QA/KPI claims, references, and speaker-facing messages from the Markdown documents.

Do not include `doc/readme.md` or `doc/RAPTOR_Hierarchical_RAG.pptx` in the default review scope unless the user explicitly asks for them.

Before writing the report, explicitly compare:

- DP titles and selected options between Markdown and HTML
- requirement, QA, KPI, and stakeholder mappings
- candidate option names and abstraction level
- trade-off criteria and selected decisions
- cross-DP dependencies such as DP1 Evidence Unit, DP2 Source Metadata, and DP3 Wiki/Cache
- presentation narrative pages for DP background and comparison slides

If one of the required files is missing, mark it as `Evidence Needed` in the report instead of inventing content.

---

Mandatory Output Language

The final report MUST be written in Korean.

The skill instructions are written in English for clarity, but the generated report content must be Korean unless the user explicitly requests another language.

Technical terms may be kept in English or written in Korean with English in parentheses.

Examples:

- 결정 지점 / Decision Point
- 근거 추적성 / Traceability
- 권한 기반 검색 / Permission-aware Retrieval
- 후보안 / Candidate Option
- 아키텍처 드라이버 / Architecture Driver

---

Mandatory Output File Rule

The final result MUST be saved as a Markdown file.

Default filename:

architecture_review_report.md

If the project name is known, use a more specific filename:

{project_name}_architecture_review_report.md

Examples:

rag_code_assistant_architecture_review_report.md
sw_architecture_assignment_review_report.md

After writing the file, report back to the user with:

- The saved Markdown file path
- The most important DP
- The weakest or most risky DP
- The top 1–3 recommended fixes

If the execution environment allows writing files, actually create the ".md" file.

Example using shell:

cat > architecture_review_report.md <<'EOF'
# Senior Architect Review Report

...
EOF

Example using Python:

from pathlib import Path

report = """# Senior Architect Review Report

...
"""

Path("architecture_review_report.md").write_text(report, encoding="utf-8")

---

Input Assumptions

The user may provide one or more of the following:

- GitHub repository path
- Local document paths
- Markdown files
- HTML files generated from docs
- PPT page planning HTML
- PPT content blueprint
- Architecture overview documents
- DP documents such as "DP1", "DP2", "DP3", etc.
- Decision records
- Trade-off tables
- QA sections
- Evaluation notes

The number of DPs is NOT fixed.

Never assume there are exactly three DPs.

Detect all DPs from the provided materials.

DP identifiers may appear as:

- "DP1", "DP2", "DP3"
- "DP-1", "DP-2"
- "Decision Point 1"
- "Architecture Decision"
- "ADR"
- Section titles containing "DP"
- Sections comparing multiple design candidates
- Sections that contain trade-off tables and selected options

If a section compares multiple architecture candidates and selects one, treat it as a possible DP even if it is not explicitly labeled as one.

---

Core Task

When invoked, perform the following:

1. Read and understand the assignment background.

2. Identify the architecture problem being solved.

3. Identify all Decision Points.

4. For each DP, evaluate the following:
   
   1. Whether the DP topic itself is important given the assignment background.
   2. Whether the DP background explanation successfully justifies why this DP is needed.
   3. Whether the detailed design candidates are appropriate for the DP topic.

5. Rank all DPs by architectural importance and review priority.

6. Evaluate cross-DP consistency.

7. Identify likely senior-reviewer questions.

8. Provide recommended fix priorities.

9. Save the final Korean report as a Markdown file.

---

QA and Trade-off Score Handling

If the user says that QA sections, trade-off scores, or evaluation numbers are provisional or arbitrary, do not focus heavily on validating those numbers.

In that case, prioritize:

- DP topic validity
- DP background justification
- Candidate design suitability
- Candidate abstraction-level consistency
- Architecture-level defensibility
- Reviewer Q&A readiness
- Cross-DP consistency
- DP ranking

Still mention trade-off tables briefly if they are structurally misaligned with the DP topic, but do not make score validation the center of the report unless the user explicitly asks for it.

---

Analysis Procedure

Step 1. Build assignment context

Extract and understand:

- Assignment background
- System goal
- Main users
- Main quality attributes
- Constraints
- Security assumptions
- Operational assumptions
- Deployment assumptions
- Why the problem requires architecture design
- What the final presentation is trying to prove

Do not over-focus on implementation details unless they affect architecture decisions.

The report should explain the assignment context only enough to support DP evaluation.

---

Step 2. Identify all DPs

Find every DP-like section or document.

For each DP, extract:

- DP number or normalized identifier
- DP title
- Problem statement
- Background / motivation
- Candidate options
- Selected option
- Claimed trade-offs
- Risks
- Dependencies on other DPs
- Related architecture drivers
- Related quality attributes

Normalize DP names when needed.

Example:

Detected name| Normalized name
DP-01 Dedup RAG| DP1
Decision Point 2 Permission| DP2
Knowledge Access Strategy| DP3
ADR: Deployment Model| DP4

If no explicit DP number exists, assign a temporary identifier:

Potential DP-A: {title}
Potential DP-B: {title}

Mention in the report that the DP label should be clarified.

---

Step 3. Evaluate DP topic importance

For each DP, answer:

«Given the assignment background, is this DP topic important enough to be treated as an Architecture Decision Point?»

Judge independently from the user's written explanation or candidate options.

Use the following rating scale:

- 매우 높음: Central to whether the system can work in production.
- 높음: Clearly tied to major quality attributes or architectural risks.
- 중간: Useful, but may be a sub-design rather than a top-level DP.
- 낮음: Looks closer to an implementation detail than an architecture-level decision.

For each DP, explain:

- Why the topic matters
- Which quality attributes it affects
- What could fail if the DP is not handled
- Whether it is architecture-level or implementation-level
- How a senior reviewer might challenge it

Also include:

**Reviewer attack point:**

> A likely question from a review committee member.

**Defense direction:**

> How the presenter should defend this DP.

The content inside these sections must be written in Korean.

---

Step 4. Evaluate DP background explanation

For each DP, answer:

«Does the current DP background explanation clearly justify why this DP is necessary?»

Use the following rating scale:

- 잘 설명됨
- 대체로 설명됨. 다만 framing 보강 필요
- 필요성은 있으나 설명이 약함
- 현재 설명으로는 DP 필요성이 잘 드러나지 않음

Evaluate whether the background connects to:

- Assignment background
- Architecture drivers
- Quality attributes
- Failure scenarios
- Operational risks
- Security risks
- Performance risks
- Maintainability risks
- User/developer workflow
- Production-readiness concerns

A good DP background explanation should include:

- The concrete problem
- Why a naive/default design fails
- What risk happens if this DP is not handled
- Why the decision must be made at architecture level
- Which quality attributes are affected

If the explanation is weak, provide improved wording:

**Recommended framing:**

> 발표나 문서에 바로 사용할 수 있는 개선된 설명 문장

The recommended framing must be written in Korean and should be usable directly in the user's document or presentation.

---

Step 5. Evaluate candidate design options

For each DP, evaluate whether the candidate options are appropriate.

Check:

- Are the candidates directly relevant to the DP topic?
- Are the candidates at the same abstraction level?
- Are they true alternatives?
- Are some options actually complementary rather than competing?
- Is a baseline option included?
- Is an important industry-standard option missing?
- Are option names clear and defensible?
- Are the comparison criteria aligned with the DP topic?
- Is the selected option defensible?

When necessary, use broader architecture knowledge, known design patterns, industry practices, and external technical knowledge.

Clearly distinguish:

- What the document explicitly says
- What is inferred from the document
- What is external architectural judgment

For each DP, include this table:

| Candidate | Suitability | Comment |
|---|---:|---|

Use these suitability values in Korean:

- 높음
- 중간
- 낮음
- 후보로는 부적절
- 보완 옵션에 가까움

Then provide:

- Missing candidates, if any
- Options that are not at the same abstraction level
- Options that are complementary rather than alternatives
- Naming issues
- Suggested candidate restructuring
- Whether the final selected option is defensible

End each DP with:

**Recommended adjustment:**

> 구체적인 수정 제안

The content must be written in Korean.

---

Step 6. Rank DPs by architectural importance

After evaluating all DPs, rank them.

The ranking must not simply follow document order.

Rank DPs using the following criteria:

Criterion| Description
Directness to assignment background| How directly the DP addresses the core problem
Quality attribute impact| Impact on security, performance, accuracy, maintainability, scalability, usability, etc.
Production-readiness impact| Whether the system can operate safely if this DP fails
Architectural blast radius| How much the DP affects other components and other DPs
Reviewer importance| How likely reviewers are to ask about it
Candidate trade-off strength| Whether the DP contains a real architecture trade-off
Risk reduction effect| How much the DP reduces major project risks
Dependency importance| Whether other DPs depend on this DP

Use the following importance labels:

- Critical
- High
- Medium
- Low

The report MUST include this table:

| Rank | DP | Importance | Reason |
|---:|---|---:|---|

Also include a ranking summary:

### Ranking Summary

- 가장 중요한 DP:
- 발표에서 가장 방어하기 쉬운 DP:
- 발표에서 가장 공격받기 쉬운 DP:
- 하위 설계로 내려도 되는 DP:
- 새로 DP로 승격할 만한 주제:

If useful, distinguish between:

Perspective| Recommended Order
Importance order| Highest architecture risk first
Presentation order| Most understandable narrative flow
Implementation order| Dependency-driven order

All ranking content must be written in Korean.

---

Step 7. Cross-DP consistency review

Evaluate how the DPs work together.

Check:

- Are DP dependencies clear?
- Does one DP produce metadata, constraints, or components required by another DP?
- Do DPs overlap?
- Are any DPs redundant?
- Do any DPs conflict?
- Are assumptions consistent?
- Does the overall architecture follow logically from the selected DP options?
- Are security, permission, versioning, citation, audit, and traceability assumptions carried across all DPs?
- Are there missing interfaces between DP results?

Include this table if helpful:

| Relationship | Assessment | Risk | Recommendation |
|---|---|---|---|

The content must be written in Korean.

---

Markdown Report Format

The final ".md" file MUST follow this structure.

# Senior Architect Review Report

## 0. Overall Assessment

| DP | Topic Importance | Background Explanation | Candidate Suitability | Reviewer Risk |
|---|---:|---:|---:|---|

요약:
- 가장 강한 DP:
- 가장 약한 DP:
- 가장 위험한 심사 질문:
- 발표 전 가장 먼저 고칠 부분:

---

## 1. Assignment Background Assessment

...

---

## 2. DP Importance Ranking

| Rank | DP | Importance | Reason |
|---:|---|---:|---|

### Ranking Summary

- 가장 중요한 DP:
- 발표에서 가장 방어하기 쉬운 DP:
- 발표에서 가장 공격받기 쉬운 DP:
- 하위 설계로 내려도 되는 DP:
- 새로 DP로 승격할 만한 주제:

---

## 3. DP-by-DP Evaluation

## DP1. {DP title}

### 1) Is this DP topic important?

...

**Reviewer attack point:**

> ...

**Defense direction:**

> ...

### 2) Does the background explanation justify this DP?

...

**Recommended framing:**

> ...

### 3) Are the candidate design options appropriate?

| Candidate | Suitability | Comment |
|---|---:|---|

...

**Recommended adjustment:**

> ...

---

## DP2. {DP title}

...

---

## 4. Cross-DP Consistency Review

| Relationship | Assessment | Risk | Recommendation |
|---|---|---|---|

...

---

## 5. Reviewer Q&A Risk List

| Question | Risk Level | Suggested Answer Direction |
|---|---:|---|

...

---

## 6. Recommended Fix Priority

| Priority | Fix | Reason |
|---:|---|---|

...

---

## 7. Final Verdict

...

All actual report content inside this structure must be Korean.

---

Report Writing Rules

The report must be written in Korean.

Use a tone that is:

- Senior architect-like
- Review committee-like
- Direct
- Practical
- Constructive
- Critical when necessary
- Focused on architectural defensibility

The report should not sound like a generic summary.

Prefer expressions like:

- “심사위원 관점에서는…”
- “이 DP는 방어하기 쉽습니다.”
- “이 DP는 주제 자체는 좋지만 framing이 약합니다.”
- “공격받을 수 있는 지점은…”
- “이건 DP라기보다 하위 설계 옵션으로 보일 수 있습니다.”
- “후보들이 같은 레벨의 대안인지 확인해야 합니다.”
- “방어 논리는 이렇게 잡는 것이 좋습니다.”
- “발표에서는 이 순서로 설명하는 편이 안전합니다.”
- “중요도 기준으로는 DP{n}이 가장 우선입니다.”

Avoid:

- Empty praise
- Generic summaries
- Repeating the source text too much
- Over-focusing on QA score validity unless asked
- Treating every candidate as equally good
- Ignoring abstraction-level mismatch
- Assuming exactly three DPs
- Hiding weak points
- Calling implementation details architecture decisions without challenge

---

DP Ranking Guidelines

Critical DP

Mark a DP as Critical if several of the following are true:

- If this DP fails, the system cannot be safely operated.
- It affects security, compliance, or data leakage risk.
- It determines the overall architecture structure.
- Other DPs depend on it.
- It directly addresses the core assignment background.
- It has a large architectural blast radius.
- Reviewers are very likely to ask about it.

High DP

Mark a DP as High if:

- It strongly affects major quality attributes.
- It directly affects system performance, accuracy, maintainability, or scalability.
- It includes a real design trade-off.
- It is likely to be important in presentation review.
- It reduces a meaningful project risk.

Medium DP

Mark a DP as Medium if:

- It is a meaningful design decision, but mostly limited to a specific component.
- It could be treated as a sub-design of another DP.
- Its options are partially complementary rather than mutually exclusive.
- It is useful but not central to the architecture story.

Low DP

Mark a DP as Low if:

- It is closer to implementation detail.
- It is already determined by another DP.
- The trade-off is weak.
- It has weak connection to the assignment background.
- It is unlikely to affect architecture-level review.

---

Reviewer Q&A Risk List

Generate likely senior-reviewer questions.

The questions should be concrete and challenging.

Use this format:

| Question | Risk Level | Suggested Answer Direction |
|---|---:|---|

Risk Level values:

- 높음
- 중간
- 낮음

Questions may include:

- “이게 정말 Architecture Decision인가요, 아니면 구현 세부사항인가요?”
- “후보들이 같은 레벨의 대안인가요?”
- “선택한 방식이 실패하면 fallback은 무엇인가요?”
- “권한이나 버전 정책이 다른 DP에도 동일하게 적용되나요?”
- “Cache나 Wiki가 stale해지면 어떻게 방지하나요?”
- “중복 제거 때문에 원본 코드의 세부 의미가 손실되지 않나요?”
- “성능 개선과 정확도 개선 중 어떤 품질 속성을 우선한 결정인가요?”
- “운영 복잡도는 어느 컴포넌트가 책임지나요?”

---

Recommended Fix Priority

Rank improvements by urgency.

Use this table:

| Priority | Fix | Reason |
|---:|---|---|

Priority 1 should be the most important fix before the presentation.

Fixes may include:

- Reframing weak DP background
- Renaming a DP
- Moving a DP to sub-design
- Adding a missing baseline candidate
- Removing a candidate that is not a true alternative
- Splitting a DP into two decisions
- Merging overlapping DPs
- Adding cross-DP metadata/interface assumptions
- Clarifying fallback strategy
- Clarifying security boundary
- Clarifying versioning or traceability assumptions

---

File Saving Instruction

After completing the analysis, write the final report to a Markdown file.

Use UTF-8 encoding.

Preferred filename:

architecture_review_report.md

If project name is known, use:

{project_name}_architecture_review_report.md

After saving, tell the user:

보고서를 저장했습니다.

- 파일: {path}
- 가장 중요한 DP: DP{n} - {title}
- 가장 위험한 DP: DP{n} - {title}
- 가장 먼저 고칠 부분: {fix}

The user-facing completion message must be Korean.

---

Final Output Quality Bar

The final report should be specific enough for the user to:

- Revise architecture documents
- Improve PPT narrative
- Rank and prioritize DPs
- Reframe weak DPs
- Restructure candidate options
- Prepare for senior-reviewer questions
- Defend why each DP matters
- Decide which DPs should remain top-level decisions
- Decide which DPs should be moved to sub-design
- Improve cross-DP consistency

The final report must be a practical review document, not a generic summary.
