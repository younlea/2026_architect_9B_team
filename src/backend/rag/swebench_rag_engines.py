"""
SWE-bench RAG 엔진 × 검색 전략 조합

RAG 인덱싱 방식 (DB 구축):
  Legacy     — Python AST 청킹 → 직접 벡터 검색 (RAG 없음, 베이스라인)
  BasicRAG   — 고정 크기 청킹 (basic_rag 방식)
  RaptorRAG  — DBSCAN 계층 노드 (raptor_rag 방식, LLM 요약 생략)
  ROIRAG     — kNN EU 구성 (roi_rag 방식, LLM 요약 생략)

검색 전략 (TopK 찾는 방법):
  Flat       — 통합 DB에서 코사인 Top-K 직접 추출
  PostFilter — 통합 DB Top-N 추출 → repo+version 메타 필터 → Top-K
  Routed     — repo+version으로 파티션 DB 직접 선택 → Top-K

조합: 4 × 3 = 12가지 비교 가능
컬렉션: swebench_flat (Legacy) / sweb_basic / sweb_raptor / sweb_roi
"""
import re
import time
import numpy as np
import chromadb
from backend.config import CHROMA_PERSIST_DIR
from backend.rag import _ef as _shared_ef
from backend.scripts.load_swebench import _parse_patch_chunks, chunk_code

TOP_K = 3
PREFETCH_K = 10   # PostFilter용 초기 fetch 수
CHUNK_SIZE = 512
CHUNK_OVERLAP = 80

ALL_ENGINES    = ["Legacy", "BasicRAG", "RaptorRAG", "ROIRAG"]
ALL_STRATEGIES = ["Flat", "PostFilter", "Routed"]

# 컬렉션 이름 매핑
_FLAT_COL_MAP = {
    "Legacy":    "swebench_flat",   # 레거시 AST 청킹 컬렉션
    "BasicRAG":  "sweb_basic",
    "RaptorRAG": "sweb_raptor",
    "ROIRAG":    "sweb_roi",
}

# Legacy 파티션 컬렉션의 PART_PREFIX (load_swebench.py 와 동일)
_LEGACY_PART_PREFIX = "swebench_part"


# ── 컬렉션 이름 ────────────────────────────────────────────────────────────────

def _flat_col(rag: str) -> str:
    return _FLAT_COL_MAP.get(rag, f"sweb_{rag.lower()}")

def _part_col(rag: str, repo: str, version: str) -> str:
    if rag == "Legacy":
        # load_swebench._col_name() 과 동일한 방식
        safe = re.sub(r"[^a-zA-Z0-9_]", "_",
                      f"{_LEGACY_PART_PREFIX}_{repo}_{version}")[:60]
        return safe
    prefix = _FLAT_COL_MAP.get(rag, f"sweb_{rag.lower()}")
    safe = re.sub(r"[^a-zA-Z0-9_]", "_", f"{repo}_{version}")[:40]
    return f"{prefix}_{safe}"


# ── ChromaDB 헬퍼 ──────────────────────────────────────────────────────────────

def _client():
    return chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)

def _ef():
    return _shared_ef.get()

def _get_col(client, name: str):
    return client.get_or_create_collection(name=name, embedding_function=_ef())

def _embed(texts: list[str]) -> np.ndarray:
    return np.array(_ef()(texts), dtype=float)


# ── 패치 파싱 ──────────────────────────────────────────────────────────────────

def _parse_file_chunks(patch: str) -> list[dict]:
    result = []
    for rc in _parse_patch_chunks(patch):
        sub = chunk_code(rc["content"], rc["file_path"])
        if not sub:
            sub = [{"content": rc["content"], "file_path": rc["file_path"]}]
        for sc in sub:
            if sc.get("content", "").strip():
                result.append({"file_path": rc["file_path"], "content": sc["content"]})
    return result

def _basic_chunk(text: str) -> list[str]:
    chunks, start = [], 0
    while start < len(text):
        chunk = text[start: start + CHUNK_SIZE].strip()
        if chunk:
            chunks.append(chunk)
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


# ── 결과 포맷 & 메트릭 헬퍼 ────────────────────────────────────────────────────

def _fmt(results: dict, answer_files: list, top_k: int) -> list[dict]:
    docs  = results.get("documents", [[]])[0]
    metas = results.get("metadatas",  [[]])[0]
    dists = results.get("distances",  [[None] * len(docs)])[0]
    answer_set = set(answer_files)
    out = []
    for doc, meta, dist in zip(docs[:top_k], metas[:top_k], dists[:top_k]):
        fps_raw = meta.get("file_paths", meta.get("file_path", ""))
        fps = [f.strip() for f in str(fps_raw).split(",") if f.strip()]
        out.append({
            "content":       doc[:600] + ("..." if len(doc) > 600 else ""),
            "file_path":     fps[0] if fps else "",
            "file_paths":    fps,
            "distance":      round(dist, 4) if dist is not None else None,
            "chunk_type":    meta.get("chunk_type", ""),
            "level":         meta.get("level", 0),
            "segment_count": meta.get("segment_count", 1),
        })
    return out

def _hit(chunks: list, answer_files: list) -> bool:
    answer_set = set(answer_files)
    return any(
        any(fp in answer_set for fp in c.get("file_paths", [c.get("file_path", "")]))
        for c in chunks
    )

def _empty(name: str) -> dict:
    return {"retriever": name, "chunks": [], "hit": False,
            "latency_ms": 0, "error": "Not indexed yet"}

def _err(name: str, msg: str) -> dict:
    return {"retriever": name, "chunks": [], "hit": False,
            "latency_ms": 0, "error": msg}


# ── RAG 인덱싱: 노드 빌더 (공통 메타 형식) ────────────────────────────────────

def _build_basic_nodes(patch: str, instance_id: str,
                       repo: str, version: str) -> tuple[list, list, list]:
    """Basic RAG: 고정 크기 청킹."""
    docs, ids, metas = [], [], []
    for fc in _parse_file_chunks(patch):
        for i, chunk in enumerate(_basic_chunk(fc["content"])):
            did = f"basic_{instance_id}_{fc['file_path']}_{i}".replace("/", "_")[:200]
            docs.append(chunk)
            ids.append(did)
            metas.append({
                "instance_id": instance_id,
                "repo": repo, "version": version,
                "file_path": fc["file_path"],
                "file_paths": fc["file_path"],
                "chunk_type": "basic", "level": 0,
            })
    return docs, ids, metas


def _build_raptor_nodes(patch: str, instance_id: str,
                        repo: str, version: str) -> tuple[list, list, list]:
    """RAPTOR RAG: 리프 노드 + DBSCAN 클러스터 노드 (LLM 요약 없음)."""
    from sklearn.cluster import DBSCAN

    file_chunks = _parse_file_chunks(patch)
    leaves = []
    for fc in file_chunks:
        for chunk in _basic_chunk(fc["content"]):
            leaves.append({"text": chunk, "file_path": fc["file_path"]})

    if not leaves:
        return [], [], []

    docs, ids, metas = [], [], []

    # 리프 노드 (level=0)
    for i, leaf in enumerate(leaves):
        did = f"raptor_leaf_{instance_id}_{leaf['file_path']}_{i}".replace("/", "_")[:200]
        docs.append(leaf["text"])
        ids.append(did)
        metas.append({
            "instance_id": instance_id,
            "repo": repo, "version": version,
            "file_path": leaf["file_path"],
            "file_paths": leaf["file_path"],
            "chunk_type": "leaf", "level": 0,
        })

    # 클러스터 노드 (level=1) — 리프 4개 이상일 때
    if len(leaves) >= 4:
        embeddings = _embed([l["text"] for l in leaves])
        db = DBSCAN(eps=0.25, min_samples=2, metric="cosine")
        raw = db.fit_predict(embeddings).tolist()

        max_lbl = max(raw) if raw else -1
        labels = []
        for lbl in raw:
            if lbl == -1:
                max_lbl += 1
                labels.append(max_lbl)
            else:
                labels.append(lbl)

        clusters: dict[int, list] = {}
        for leaf, lbl in zip(leaves, labels):
            clusters.setdefault(lbl, []).append(leaf)

        for ci, cl in enumerate(clusters.values()):
            if len(cl) < 2:
                continue
            summary = "\n---\n".join(l["text"][:300] for l in cl[:5])
            fps = list(dict.fromkeys(l["file_path"] for l in cl))
            did = f"raptor_cluster_{instance_id}_{ci}".replace("/", "_")[:200]
            docs.append(summary)
            ids.append(did)
            metas.append({
                "instance_id": instance_id,
                "repo": repo, "version": version,
                "file_path": fps[0],
                "file_paths": ",".join(fps),
                "chunk_type": "cluster", "level": 1,
                "cluster_size": len(cl),
            })

    return docs, ids, metas


def _entropy_re(emb_subset: np.ndarray) -> float:
    """논문 Algorithm 1의 RE(중복 엔트로피). roi_rag._compute_entropy_indices와 동일.
    RE = 1 - H(거리분포)/log(n²).  n<2면 0.
    """
    n = len(emb_subset)
    if n < 2:
        return 0.0
    norms = np.linalg.norm(emb_subset, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    unit = emb_subset / norms
    dist = 1.0 - unit @ unit.T
    np.fill_diagonal(dist, 0.0)
    total = dist.sum() + 1e-10
    p = dist / total
    H = float(-np.sum(p * np.log(p + 1e-10)))
    log_n2 = np.log(float(n) ** 2)
    DE = float(np.clip(H / log_n2, 0.0, 1.0)) if log_n2 > 1e-10 else 0.0
    return 1.0 - DE


def _build_roi_nodes(patch: str, instance_id: str,
                     repo: str, version: str):
    """ROI-RAG: 논문 §3.4 Entropy-Guided EU 구성 + aggregated(mean-pool) 임베딩.
    (오프라인 LLM 요약은 SWE-bench 대량 인덱싱 비용상 생략 — 구조적 부분만 충실 반영)

    본 roi_rag.py와 동일한 절차:
      1) 각 세그먼트의 kNN 이웃 neighborhood 구성
      2) neighborhood RE(중복도)가 높은 순으로 seed 선택
      3) seed에 '증분 다양성 최대' 멤버를 budget까지 추가 (non-overlap)
    Returns (docs, embeds, ids, metas)
    """
    KNN_K, MAX_SEG = 5, 4
    file_chunks = _parse_file_chunks(patch)
    all_segs = []
    for fc in file_chunks:
        for chunk in _basic_chunk(fc["content"]):
            all_segs.append({"text": chunk, "file_path": fc["file_path"]})

    if not all_segs:
        return [], [], [], []

    docs, embeds, ids, metas = [], [], [], []

    if len(all_segs) < 3:
        for i, seg in enumerate(all_segs):
            did = f"roi_{instance_id}_eu_{i}".replace("/", "_")[:200]
            emb = _embed([seg["text"]])[0].tolist()
            docs.append(seg["text"])
            embeds.append(emb)
            ids.append(did)
            metas.append({
                "instance_id": instance_id,
                "repo": repo, "version": version,
                "file_path": seg["file_path"],
                "file_paths": seg["file_path"],
                "chunk_type": "eu", "segment_count": 1,
            })
        return docs, embeds, ids, metas

    embeddings = _embed([s["text"] for s in all_segs])
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    unit = embeddings / norms
    sim = unit @ unit.T
    n = len(all_segs)

    # 1) kNN 이웃 (자기 자신 제외 top-KNN_K)
    k_act = min(KNN_K, n - 1)
    neighborhoods = []
    for i in range(n):
        order = np.argsort(sim[i])[::-1]
        neighborhoods.append([int(j) for j in order if int(j) != i][:k_act])

    # 2) neighborhood RE로 seed 우선순위 (중복 많은 이웃 먼저 — 논문 §3.4)
    nbr_re = [_entropy_re(embeddings[[i] + neighborhoods[i]]) for i in range(n)]
    seed_order = sorted(range(n), key=lambda i: nbr_re[i], reverse=True)

    used: set = set()
    eu_groups: list[list[int]] = []
    for seed in seed_order:
        if seed in used:
            continue
        group = [seed]
        used.add(seed)
        cands = [j for j in neighborhoods[seed] if j not in used]
        # 3) 증분 다양성 최대 멤버 추가 (EU 평균과 유사도가 가장 낮은 후보)
        while len(group) < MAX_SEG and cands:
            mean_vec = embeddings[group].mean(axis=0)
            mean_unit = mean_vec / (np.linalg.norm(mean_vec) + 1e-10)
            best = max(cands, key=lambda j: 1.0 - float(unit[j] @ mean_unit))
            group.append(best)
            used.add(best)
            cands = [j for j in cands if j not in used]
        eu_groups.append(group)

    for ei, group in enumerate(eu_groups):
        eu_text = "\n".join(all_segs[j]["text"][:400] for j in group)
        fps = list(dict.fromkeys(all_segs[j]["file_path"] for j in group))
        eu_emb = embeddings[group].mean(axis=0).tolist()   # aggregated 임베딩 (논문)
        did = f"roi_{instance_id}_eu_{ei}".replace("/", "_")[:200]
        docs.append(eu_text)
        embeds.append(eu_emb)
        ids.append(did)
        metas.append({
            "instance_id": instance_id,
            "repo": repo, "version": version,
            "file_path": fps[0],
            "file_paths": ",".join(fps),
            "chunk_type": "eu", "segment_count": len(group),
            "re": round(float(_entropy_re(embeddings[group])), 4),
        })

    return docs, embeds, ids, metas


# ── 인덱싱: flat + partition 두 컬렉션에 동시 삽입 ────────────────────────────

def index_issue(client, rag_method: str, instance_id: str,
                patch: str, repo: str, version: str) -> int:
    """주어진 RAG 방식으로 이슈를 flat + partition 컬렉션에 인덱싱."""
    flat = _get_col(client, _flat_col(rag_method))
    part = _get_col(client, _part_col(rag_method, repo, version))

    if rag_method == "Legacy":
        # 레거시: Python AST 청킹 (load_swebench.py 방식과 동일)
        docs, ids, metas = [], [], []
        for chunk in _parse_patch_chunks(patch):
            fp = chunk["file_path"]
            sub_chunks = chunk_code(chunk["content"], fp)
            if not sub_chunks:
                sub_chunks = [{"content": chunk["content"], "file_path": fp,
                               "start_line": 0, "end_line": 0,
                               "chunk_type": "raw", "name": "raw"}]
            for ci, sc in enumerate(sub_chunks):
                if not sc.get("content", "").strip():
                    continue
                did = f"{instance_id}__{fp}__{ci}".replace("/", "_")[:200]
                docs.append(sc["content"])
                ids.append(did)
                metas.append({
                    "instance_id": instance_id,
                    "repo": repo, "version": version,
                    "file_path": fp, "file_paths": fp,
                    "start_line": sc.get("start_line", 0),
                    "end_line":   sc.get("end_line",   0),
                    "chunk_type": sc.get("chunk_type", ""),
                    "name":       sc.get("name", ""),
                })
        if not docs:
            return 0
        flat.upsert(documents=docs, ids=ids, metadatas=metas)
        part.upsert(documents=docs, ids=ids, metadatas=metas)
        return len(docs)

    if rag_method == "BasicRAG":
        docs, ids, metas = _build_basic_nodes(patch, instance_id, repo, version)
        if not docs:
            return 0
        flat.upsert(documents=docs, ids=ids, metadatas=metas)
        part.upsert(documents=docs, ids=ids, metadatas=metas)

    elif rag_method == "RaptorRAG":
        docs, ids, metas = _build_raptor_nodes(patch, instance_id, repo, version)
        if not docs:
            return 0
        flat.upsert(documents=docs, ids=ids, metadatas=metas)
        part.upsert(documents=docs, ids=ids, metadatas=metas)

    elif rag_method == "ROIRAG":
        docs, embs, ids, metas = _build_roi_nodes(patch, instance_id, repo, version)
        if not docs:
            return 0
        flat.upsert(documents=docs, embeddings=embs, ids=ids, metadatas=metas)
        part.upsert(documents=docs, embeddings=embs, ids=ids, metadatas=metas)

    else:
        return 0

    return len(docs)


# ── 검색 전략 ──────────────────────────────────────────────────────────────────

def retrieve(client, rag_method: str, strategy: str,
             query: str, repo: str, version: str,
             answer_files: list, top_k: int = TOP_K) -> dict:
    """(rag_method × strategy) 조합으로 검색."""
    name = f"{rag_method}+{strategy}"
    start = time.time()

    try:
        if strategy == "Flat":
            col = client.get_collection(_flat_col(rag_method), embedding_function=_ef())
            count = col.count()
            if count == 0:
                return _empty(name)
            results = col.query(
                query_texts=[query],
                n_results=min(top_k, count),
                include=["documents", "metadatas", "distances"],
            )
            chunks = _fmt(results, answer_files, top_k)
            extra = {
                "db_searched": _flat_col(rag_method),
                "criteria": "통합 DB 전체 검색 (조건 없음)",
            }

        elif strategy == "PostFilter":
            col = client.get_collection(_flat_col(rag_method), embedding_function=_ef())
            count = col.count()
            if count == 0:
                return _empty(name)
            results = col.query(
                query_texts=[query],
                n_results=min(PREFETCH_K, count),
                include=["documents", "metadatas", "distances"],
            )
            all_chunks = _fmt(results, answer_files, PREFETCH_K)
            # raw metadata에서 직접 repo/version 필터 (_fmt 출력엔 이 필드가 없음)
            raw_metas = results.get("metadatas", [[]])[0]
            filtered = [
                all_chunks[i]
                for i, m in enumerate(raw_metas[:len(all_chunks)])
                if m.get("repo") == repo and str(m.get("version", "")) == str(version)
            ]
            prefetch_n = len(all_chunks)
            if filtered:
                chunks = filtered[:top_k]
                filter_matched = True
                kept_n = len(filtered)
            else:
                kept_n = 0
                # 메타에 repo/version 없는 경우 → WHERE 절로 재시도 → 실패 시 전체 top-K
                try:
                    results2 = col.query(
                        query_texts=[query],
                        n_results=min(top_k, count),
                        where={"$and": [{"repo": {"$eq": repo}},
                                        {"version": {"$eq": version}}]},
                        include=["documents", "metadatas", "distances"],
                    )
                    chunks = _fmt(results2, answer_files, top_k)
                    if not chunks:
                        chunks = all_chunks[:top_k]
                except Exception:
                    chunks = all_chunks[:top_k]
                filter_matched = False
            extra = {
                "db_searched": _flat_col(rag_method),
                "filter_matched": filter_matched,
                "filter_condition": f'repo == "{repo}" AND version == "{version}"',
                "prefetch_k": prefetch_n,
                "kept": kept_n,
                "dropped": max(prefetch_n - kept_n, 0),
                "criteria": f"통합 DB Top-{PREFETCH_K} → repo/version 메타 필터 → Top-{top_k}",
            }

        elif strategy == "Routed":
            col_name = _part_col(rag_method, repo, version)
            try:
                col = client.get_collection(col_name, embedding_function=_ef())
                count = col.count()
                if count == 0:
                    raise Exception("empty partition")
                results = col.query(
                    query_texts=[query],
                    n_results=min(top_k, count),
                    include=["documents", "metadatas", "distances"],
                )
                chunks = _fmt(results, answer_files, top_k)
                extra = {
                    "db_searched": col_name,
                    "route_key": f"{repo}@{version}",
                    "route_condition": f'repo == "{repo}" AND version == "{version}"',
                    "criteria": f"라우팅 키 {repo}@{version} → 파티션 DB 직접 선택",
                }
            except Exception:
                # 파티션 없음 → flat 컬렉션 + where 필터로 폴백
                col = client.get_collection(_flat_col(rag_method), embedding_function=_ef())
                count = col.count()
                if count == 0:
                    return _empty(name)
                try:
                    results = col.query(
                        query_texts=[query],
                        n_results=min(top_k, count),
                        where={"$and": [{"repo": {"$eq": repo}},
                                        {"version": {"$eq": version}}]},
                        include=["documents", "metadatas", "distances"],
                    )
                    chunks = _fmt(results, answer_files, top_k)
                    if not chunks:
                        raise Exception("no matches")
                except Exception:
                    # where 필터도 실패 → flat 전체에서 top-K
                    results = col.query(
                        query_texts=[query],
                        n_results=min(top_k, count),
                        include=["documents", "metadatas", "distances"],
                    )
                    chunks = _fmt(results, answer_files, top_k)
                extra = {
                    "db_searched": _flat_col(rag_method),
                    "route_key": f"{repo}@{version}",
                    "route_condition": f'repo == "{repo}" AND version == "{version}"',
                    "fallback": "no_partition",
                    "criteria": f"파티션({repo}@{version}) 없음 → 통합 DB + where 필터 폴백",
                }

        else:
            return _err(name, f"Unknown strategy: {strategy}")

    except Exception as e:
        return _err(name, str(e))

    latency = int((time.time() - start) * 1000)
    return {
        "retriever": name,
        "rag_method": rag_method,
        "strategy": strategy,
        "chunks": chunks,
        "hit": _hit(chunks, answer_files),
        "latency_ms": latency,
        **extra,
    }


# ── 상태 확인 ──────────────────────────────────────────────────────────────────

def engine_status() -> dict:
    """각 RAG 엔진의 flat 컬렉션 인덱싱 여부 반환."""
    ef = _ef()
    result = {}
    try:
        client = _client()
        for rag in ALL_ENGINES:
            try:
                col = client.get_collection(_flat_col(rag), embedding_function=ef)
                cnt = col.count()
                result[rag] = {"indexed": cnt > 0, "chunk_count": cnt}
            except Exception:
                result[rag] = {"indexed": False, "chunk_count": 0}
    except Exception:
        for rag in ALL_ENGINES:
            result[rag] = {"indexed": False, "chunk_count": 0}
    return result
