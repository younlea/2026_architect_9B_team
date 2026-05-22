"""
SWE-bench Lite 데이터셋 다운로드 → 파싱 → ChromaDB 인덱싱 스크립트

실행: cd src && PYTHONPATH=$(pwd) python backend/scripts/load_swebench.py
"""
import re
import sys
import json
import ast
import textwrap
from pathlib import Path

# PYTHONPATH 확인
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.db.database import get_conn, init_db
from backend.rag import _ef as _shared_ef
from backend.config import CHROMA_PERSIST_DIR
import chromadb

FLAT_COL = "swebench_flat"
PART_PREFIX = "swebench_part"


# ── 패치 파싱 ─────────────────────────────────────────────────────────────

def _parse_patch_files(patch: str) -> list[str]:
    """패치에서 수정된 파일 경로 목록 추출 (정답 파일 목록)."""
    return re.findall(r'^diff --git a/(.+?) b/', patch, re.MULTILINE)


def _parse_patch_chunks(patch: str) -> list[dict]:
    """패치를 파일별 코드 청크로 분리.
    각 청크는 해당 파일의 변경 전후 컨텍스트 코드를 담습니다.
    """
    chunks = []
    current_file = None
    hunk_lines: list[str] = []

    for line in patch.splitlines():
        if line.startswith("diff --git"):
            if current_file and hunk_lines:
                chunks.append({"file_path": current_file, "content": "\n".join(hunk_lines)})
            m = re.match(r"diff --git a/(.+?) b/", line)
            current_file = m.group(1) if m else None
            hunk_lines = []
        elif line.startswith(("--- ", "+++ ", "index ")):
            continue
        elif line.startswith("@@"):
            if hunk_lines:
                hunk_lines.append("")  # 헝크 구분
            hunk_lines.append(line)
        else:
            # 컨텍스트(공백), 삭제(-), 추가(+) 라인 모두 포함 (diff 접두사 제거)
            if line and line[0] in (" ", "-", "+"):
                hunk_lines.append(line[1:])
            elif line:
                hunk_lines.append(line)

    if current_file and hunk_lines:
        chunks.append({"file_path": current_file, "content": "\n".join(hunk_lines)})

    return chunks


# ── Python AST 기반 함수/클래스 청킹 ──────────────────────────────────────

def _ast_chunk(source: str, file_path: str) -> list[dict]:
    """AST로 함수/클래스 단위 청킹. 실패 시 빈 리스트 반환."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    chunks = []
    lines = source.splitlines()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        start = node.lineno - 1
        end = node.end_lineno
        body = "\n".join(lines[start:end])
        if len(body.strip()) < 20:
            continue
        chunks.append({
            "content": body,
            "file_path": file_path,
            "start_line": node.lineno,
            "end_line": node.end_lineno,
            "chunk_type": type(node).__name__,
            "name": node.name,
        })
    return chunks


def _line_chunk(source: str, file_path: str, chunk_size: int = 50, overlap: int = 10) -> list[dict]:
    """줄 기반 fallback 청킹."""
    lines = source.splitlines()
    chunks = []
    step = chunk_size - overlap
    for i in range(0, max(1, len(lines)), max(1, step)):
        block = lines[i: i + chunk_size]
        if not any(l.strip() for l in block):
            continue
        chunks.append({
            "content": "\n".join(block),
            "file_path": file_path,
            "start_line": i + 1,
            "end_line": i + len(block),
            "chunk_type": "lines",
            "name": f"lines_{i+1}_{i+len(block)}",
        })
    return chunks


def chunk_code(source: str, file_path: str) -> list[dict]:
    """AST → fallback 줄 기반 청킹."""
    result = _ast_chunk(source, file_path)
    if result:
        return result
    return _line_chunk(source, file_path)


# ── ChromaDB 헬퍼 ─────────────────────────────────────────────────────────

def _get_client():
    return chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)


def _col_name(repo: str, version: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_]", "_", f"{PART_PREFIX}_{repo}_{version}")
    return safe[:63]  # ChromaDB 이름 길이 제한


def _get_or_create(client, name: str):
    ef = _shared_ef.get()
    return client.get_or_create_collection(name=name, embedding_function=ef)


# ── SQLite 저장 ───────────────────────────────────────────────────────────

def _save_issue(row: dict, answer_files: list[str]):
    with get_conn() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO swebench_issues
               (instance_id, repo, version, problem_statement, answer_files)
               VALUES (?, ?, ?, ?, ?)""",
            (
                row["instance_id"],
                row["repo"],
                row["version"],
                row["problem_statement"],
                json.dumps(answer_files, ensure_ascii=False),
            ),
        )


# ── 메인 인덱싱 ───────────────────────────────────────────────────────────

def load_and_index(max_issues: int = 300, verbose: bool = True):
    """SWE-bench Lite를 다운로드하여 Flat DB + Partitioned DB 양쪽에 인덱싱합니다."""
    from datasets import load_dataset

    init_db()
    ds = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
    client = _get_client()
    flat_col = _get_or_create(client, FLAT_COL)

    total = min(max_issues, len(ds))
    flat_docs, flat_ids, flat_metas = [], [], []

    for idx in range(total):
        row = ds[idx]
        instance_id = row["instance_id"]
        repo = row["repo"]
        version = str(row["version"])
        patch = row["patch"]
        problem = row["problem_statement"]

        answer_files = _parse_patch_files(patch)
        _save_issue(row, answer_files)

        patch_chunks = _parse_patch_chunks(patch)

        part_docs, part_ids, part_metas = [], [], []

        for chunk in patch_chunks:
            file_path = chunk["file_path"]
            source = chunk["content"]

            sub_chunks = chunk_code(source, file_path)
            if not sub_chunks:
                sub_chunks = [{"content": source, "file_path": file_path,
                               "start_line": 0, "end_line": 0,
                               "chunk_type": "raw", "name": "raw"}]

            for ci, sc in enumerate(sub_chunks):
                doc_id = f"{instance_id}__{file_path}__{ci}".replace("/", "_")[:200]
                meta = {
                    "instance_id": instance_id,
                    "repo": repo,
                    "version": version,
                    "file_path": file_path,
                    "start_line": sc.get("start_line", 0),
                    "end_line": sc.get("end_line", 0),
                    "chunk_type": sc.get("chunk_type", ""),
                    "name": sc.get("name", ""),
                }
                text = sc["content"]
                if not text.strip():
                    continue

                flat_docs.append(text)
                flat_ids.append(doc_id)
                flat_metas.append(meta)

                part_docs.append(text)
                part_ids.append(doc_id)
                part_metas.append(meta)

        # 파티션 DB 업서트
        if part_docs:
            col_name = _col_name(repo, version)
            part_col = _get_or_create(client, col_name)
            _batch_upsert(part_col, part_docs, part_ids, part_metas)

        if verbose:
            print(f"[{idx+1}/{total}] {instance_id} — {len(part_docs)} chunks")

    # Flat DB 업서트 (배치)
    _batch_upsert(flat_col, flat_docs, flat_ids, flat_metas)
    print(f"\n✅ 완료: {total}개 이슈, flat={flat_col.count()} 청크")


def _batch_upsert(col, docs, ids, metas, batch=200):
    for i in range(0, len(docs), batch):
        col.upsert(
            documents=docs[i: i + batch],
            ids=ids[i: i + batch],
            metadatas=metas[i: i + batch],
        )


def get_index_status() -> dict:
    """현재 인덱싱 상태 반환."""
    try:
        client = _get_client()
        ef = _shared_ef.get()
        flat = client.get_collection(name=FLAT_COL, embedding_function=ef)
        flat_count = flat.count()
    except Exception:
        flat_count = 0

    with get_conn() as conn:
        issue_count = conn.execute(
            "SELECT COUNT(*) as cnt FROM swebench_issues"
        ).fetchone()["cnt"]

    return {"issue_count": issue_count, "flat_chunk_count": flat_count, "indexed": flat_count > 0}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max", type=int, default=300, help="최대 이슈 수")
    args = parser.parse_args()
    load_and_index(max_issues=args.max)
