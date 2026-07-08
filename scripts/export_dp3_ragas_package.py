#!/usr/bin/env python3
"""Create a portable RAGAS package from saved DP3 TC3 run logs."""

from __future__ import annotations

import argparse
import json
import shutil
import tarfile
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RUN_LOG_DIR = ROOT / "src" / "data" / "dp3_run_logs"
DEFAULT_OUTPUT_ROOT = ROOT / "ragas_export"
RUNNER_TEMPLATE_DIR = ROOT / "ragas_export" / "dp3_tc3_ragas_techqa_20260630"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def is_tc3_log(data: dict[str, Any]) -> bool:
    request_case = data.get("request", {}).get("test_case")
    result_case = data.get("result", {}).get("test_case")
    return request_case == "similar_pair_quality" or result_case == "similar_pair_quality"


def tc3_logs(log_dir: Path) -> list[Path]:
    paths = []
    for path in log_dir.glob("*.json"):
        try:
            data = load_json(path)
        except Exception:
            continue
        if is_tc3_log(data):
            paths.append(path)
    return sorted(paths, key=lambda item: item.stat().st_mtime)


def safe_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def resolve_ragas_input(log_path: Path, data: dict[str, Any]) -> Path | None:
    value = data.get("result", {}).get("ragas_input_path")
    if not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = (log_path.parent / path).resolve()
    return path if path.exists() else None


def write_readme(package_dir: Path, manifest: dict[str, Any]) -> None:
    readme = f"""# DP3 TC3 RAGAS Package

이 패키지는 DP3 TC3 Similar Query Pair Quality 실행 결과를 다른 PC에서 official RAGAS로 평가하기 위한 export다.

## 구성

| 경로 | 설명 |
|---|---|
| `inputs/` | run별 RAGAS JSONL 입력 |
| `run_logs/` | 원본 DP3 run log JSON |
| `run_official_ragas_slow.py` | standalone RAGAS 실행 스크립트 |
| `requirements-ragas-runner.txt` | 설치용 requirements |
| `manifest.json` | run log와 input 매핑 |

## 설치

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-ragas-runner.txt
```

## 단일 input 실행

```bash
export GROQ_API_KEY="..."
python run_official_ragas_slow.py \\
  --provider groq \\
  --model meta-llama/llama-4-scout-17b-16e-instruct \\
  --input inputs/{manifest['runs'][0]['input_file'] if manifest['runs'] else '<input-file>.jsonl'} \\
  --output-dir official_ragas_output/{manifest['runs'][0]['run_id'] if manifest['runs'] else '<run-id>'} \\
  --chunk-size 1 \\
  --seconds-per-request 65
```

## 모든 input 순차 실행 예시

```bash
export GROQ_API_KEY="..."
for input in inputs/*.jsonl; do
  name="$(basename "$input" .jsonl)"
  python run_official_ragas_slow.py \\
    --provider groq \\
    --model meta-llama/llama-4-scout-17b-16e-instruct \\
    --input "$input" \\
    --output-dir "official_ragas_output/$name" \\
    --chunk-size 1 \\
    --seconds-per-request 65
done
```

## 성공 판정

- 각 output의 `summary.json`에서 `rows_scored`가 input row 수와 같아야 한다.
- A/B row 수가 동일해야 한다.
- 특정 metric의 `missing_or_nan`이 0이 아니면 해당 metric은 최종 해석에서 제외하거나 partial로 표시한다.
"""
    package_dir.joinpath("README.md").write_text(readme, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", default=str(RUN_LOG_DIR))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--name", default=f"dp3_tc3_ragas_export_{time.strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument("--latest", type=int, default=0, help="0 means all TC3 logs.")
    parser.add_argument("--make-tar", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    logs = tc3_logs(Path(args.log_dir))
    if args.latest and args.latest > 0:
        logs = logs[-args.latest :]
    if not logs:
        raise SystemExit("No TC3 run logs found.")

    output_root = Path(args.output_root)
    package_dir = output_root / args.name
    inputs_dir = package_dir / "inputs"
    logs_dir = package_dir / "run_logs"
    package_dir.mkdir(parents=True, exist_ok=True)

    for filename in ["run_official_ragas_slow.py", "requirements-ragas-runner.txt"]:
        source = RUNNER_TEMPLATE_DIR / filename
        if not source.exists():
            raise SystemExit(f"Missing runner template: {source}")
        safe_copy(source, package_dir / filename)

    manifest: dict[str, Any] = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source_log_dir": str(Path(args.log_dir).resolve()),
        "runs": [],
    }

    for log_path in logs:
        data = load_json(log_path)
        run_id = log_path.stem
        safe_copy(log_path, logs_dir / log_path.name)
        input_path = resolve_ragas_input(log_path, data)
        input_file = None
        input_rows = 0
        if input_path is not None:
            input_file = f"{run_id}.jsonl"
            safe_copy(input_path, inputs_dir / input_file)
            input_rows = sum(1 for line in input_path.read_text(encoding="utf-8").splitlines() if line.strip())
        manifest["runs"].append(
            {
                "run_id": run_id,
                "run_log_file": log_path.name,
                "input_file": input_file,
                "input_rows": input_rows,
                "saved_at": data.get("saved_at"),
                "job_id": data.get("job_id"),
                "request": data.get("request", {}),
                "source_ragas_input_path": str(input_path) if input_path else None,
            }
        )

    package_dir.joinpath("manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_readme(package_dir, manifest)

    if args.make_tar:
        archive = output_root / f"{args.name}.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            tar.add(package_dir, arcname=package_dir.name)
        print(json.dumps({"package_dir": str(package_dir), "archive": str(archive), "runs": len(manifest["runs"])}, indent=2))
    else:
        print(json.dumps({"package_dir": str(package_dir), "runs": len(manifest["runs"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
