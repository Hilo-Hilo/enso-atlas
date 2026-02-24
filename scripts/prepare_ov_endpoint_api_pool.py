#!/usr/bin/env python3
"""
Build API-derived OV endpoint training pools from GDC + bucket manifests.

Supported endpoints:
  - survival_1y
  - survival_3y
  - survival_5y
  - tumor_grade

Outputs:
  - <out_prefix>_meta.csv
  - <out_prefix>_labels.csv
  - <out_prefix>_file_ids.txt
  - <out_prefix>_h5_uris.txt
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


GDC_FILES_URL = "https://api.gdc.cancer.gov/files"

FIELDS = [
    "file_id",
    "file_name",
    "cases.case_id",
    "cases.submitter_id",
    "cases.project.project_id",
    "cases.demographic.vital_status",
    "cases.demographic.days_to_death",
    "cases.diagnoses.days_to_last_follow_up",
    "cases.diagnoses.tumor_grade",
    "cases.follow_ups.days_to_follow_up",
]


def normalize(s: object) -> str:
    return re.sub(r"\s+", " ", str(s or "").strip().lower())


def _try_float(v: object) -> Optional[float]:
    try:
        if v is None or v == "":
            return None
        return float(v)
    except Exception:
        return None


def specimen_type(slide_id: str) -> str:
    if "-TS" in slide_id:
        return "TS"
    if "-BS" in slide_id:
        return "BS"
    return "OTHER"


def patient_id_from_slide(slide_id: str) -> str:
    parts = slide_id.split("-")
    if len(parts) >= 3 and parts[0] == "TCGA":
        return "-".join(parts[:3])
    return slide_id


def gdc_get_json(params: Dict[str, str]) -> Dict:
    query = urllib.parse.urlencode(params)
    with urllib.request.urlopen(f"{GDC_FILES_URL}?{query}", timeout=180) as resp:
        return json.loads(resp.read().decode("utf-8"))


def iter_tcga_ov_svs() -> Iterable[Dict]:
    filters = {
        "op": "and",
        "content": [
            {"op": "=", "content": {"field": "files.data_format", "value": ["SVS"]}},
            {"op": "=", "content": {"field": "cases.project.project_id", "value": ["TCGA-OV"]}},
        ],
    }
    size = 2000
    start = 0
    total = None
    while True:
        params = {
            "filters": json.dumps(filters),
            "fields": ",".join(FIELDS),
            "format": "JSON",
            "size": str(size),
            "from": str(start),
        }
        payload = gdc_get_json(params)
        warnings = payload.get("warnings")
        if warnings:
            print(f"WARN: {warnings}")

        data = payload.get("data", {})
        hits = data.get("hits", [])
        pagination = data.get("pagination", {})
        if total is None:
            total = int(pagination.get("total", 0))
            print(f"GDC OV SVS total={total}")

        if not hits:
            break

        for h in hits:
            yield h

        start += size
        if start >= total:
            break


def infer_survival_label(case_obj: Dict, years: int) -> Tuple[Optional[int], str]:
    """
    Binary endpoint:
      1 = survived > N years
      0 = deceased within N years
    """
    threshold_days = float(years) * 365.25
    demo = case_obj.get("demographic") or {}
    vital = normalize(demo.get("vital_status"))
    days_to_death = _try_float(demo.get("days_to_death"))

    follow_up_days: Optional[float] = None
    for diag in case_obj.get("diagnoses") or []:
        dtf = _try_float(diag.get("days_to_last_follow_up"))
        if dtf is not None and (follow_up_days is None or dtf > follow_up_days):
            follow_up_days = dtf
    for fu in case_obj.get("follow_ups") or []:
        dtf = _try_float(fu.get("days_to_follow_up"))
        if dtf is not None and (follow_up_days is None or dtf > follow_up_days):
            follow_up_days = dtf

    if vital == "dead" and days_to_death is not None:
        return (1 if days_to_death > threshold_days else 0), "demographic.days_to_death"
    if vital == "alive" and follow_up_days is not None and follow_up_days >= threshold_days:
        return 1, "follow_up_censoring"
    return None, "unlabeled"


def infer_tumor_grade_label(case_obj: Dict) -> Tuple[Optional[int], str]:
    """
    Binary endpoint:
      1 = high grade (G3/G4/GB)
      0 = low grade (G1/G2)
    """
    grade_tokens: List[str] = []
    for diag in case_obj.get("diagnoses") or []:
        g = normalize(diag.get("tumor_grade"))
        if g:
            grade_tokens.append(g)

    if not grade_tokens:
        return None, "unlabeled"

    has_high = any(g in {"g3", "g4", "gb"} or "g3" in g or "g4" in g for g in grade_tokens)
    has_low = any(g in {"g1", "g2"} or "g1" in g or "g2" in g for g in grade_tokens)

    if has_high:
        return 1, "diagnoses.tumor_grade"
    if has_low:
        return 0, "diagnoses.tumor_grade"
    return None, "unlabeled"


def infer_label(case_obj: Dict, endpoint: str) -> Tuple[Optional[int], str]:
    if endpoint == "survival_1y":
        return infer_survival_label(case_obj, years=1)
    if endpoint == "survival_3y":
        return infer_survival_label(case_obj, years=3)
    if endpoint == "survival_5y":
        return infer_survival_label(case_obj, years=5)
    if endpoint == "tumor_grade":
        return infer_tumor_grade_label(case_obj)
    raise ValueError(f"Unsupported endpoint: {endpoint}")


def status_column_name(endpoint: str) -> str:
    if endpoint.startswith("survival_"):
        return endpoint + "_status"
    if endpoint == "tumor_grade":
        return "grade_status"
    raise ValueError(f"Unsupported endpoint: {endpoint}")


def status_value(endpoint: str, label: int) -> str:
    if endpoint.startswith("survival_"):
        return "survived" if int(label) == 1 else "deceased"
    if endpoint == "tumor_grade":
        return "high" if int(label) == 1 else "low"
    raise ValueError(f"Unsupported endpoint: {endpoint}")


def read_available_file_ids(path: Path) -> set:
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def read_bucket_scan(path: Path, available: set) -> Dict[str, Dict[str, str]]:
    by_file_id: Dict[str, Dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            file_id = (row.get("file_id") or "").strip()
            if not file_id or file_id not in available:
                continue
            file_name = (row.get("file_name") or "").strip()
            slide_id = file_name.rsplit(".", 1)[0] if "." in file_name else file_name
            patient_id = patient_id_from_slide(slide_id)
            by_file_id[file_id] = {
                "file_id": file_id,
                "file_name": file_name,
                "slide_id": slide_id,
                "patient_id": patient_id,
                "site": patient_id.split("-")[1] if patient_id.startswith("TCGA-") and len(patient_id.split("-")) >= 3 else "",
                "specimen": specimen_type(slide_id),
            }
    return by_file_id


def write_csv(path: Path, rows: List[Dict[str, object]], cols: List[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in rows:
            w.writerow({c: row.get(c, "") for c in cols})


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare API OV endpoint pool from bucket manifests.")
    parser.add_argument(
        "--endpoint",
        type=str,
        required=True,
        choices=["survival_1y", "survival_3y", "survival_5y", "tumor_grade"],
    )
    parser.add_argument("--scan_csv", type=str, required=True, help="bucket_physical_scan.csv path")
    parser.add_argument("--available_file_ids", type=str, required=True, help="available_file_ids.txt path")
    parser.add_argument("--out_dir", type=str, required=True, help="output directory")
    parser.add_argument("--out_prefix", type=str, default=None, help="output prefix override")
    args = parser.parse_args()

    endpoint = args.endpoint
    out_prefix = args.out_prefix or f"trainval_pool_ov_{endpoint}_api"
    status_col = status_column_name(endpoint)

    scan_csv = Path(args.scan_csv)
    avail_path = Path(args.available_file_ids)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    available = read_available_file_ids(avail_path)
    scan_by_id = read_bucket_scan(scan_csv, available)
    print(f"Endpoint={endpoint}")
    print(f"Bucket available file_ids in scan: {len(scan_by_id)}")

    # Case-level label inference from API
    case_label: Dict[str, Tuple[int, str, str]] = {}
    seen_hits = 0
    used_hits = 0
    for hit in iter_tcga_ov_svs():
        seen_hits += 1
        file_id = (hit.get("file_id") or hit.get("id") or "").strip()
        if file_id not in scan_by_id:
            continue
        used_hits += 1
        cases = hit.get("cases") or []
        if not cases:
            continue
        case_obj = cases[0]
        case_id = (case_obj.get("case_id") or "").strip()
        if not case_id:
            continue
        if case_id not in case_label:
            lab, src = infer_label(case_obj, endpoint=endpoint)
            if lab is not None:
                case_label[case_id] = (lab, src, (case_obj.get("submitter_id") or "").strip())

    print(f"GDC hits scanned={seen_hits}, bucket-overlap hits={used_hits}, labeled cases={len(case_label)}")

    # Build slide-level rows
    meta_rows: List[Dict[str, object]] = []
    label_rows: List[Dict[str, object]] = []
    class_counter = Counter()
    source_counter = Counter()

    for hit in iter_tcga_ov_svs():
        file_id = (hit.get("file_id") or hit.get("id") or "").strip()
        scan = scan_by_id.get(file_id)
        if scan is None:
            continue
        cases = hit.get("cases") or []
        if not cases:
            continue
        case_obj = cases[0]
        case_id = (case_obj.get("case_id") or "").strip()
        if not case_id or case_id not in case_label:
            continue

        lab, src, case_submitter_id = case_label[case_id]
        stat = status_value(endpoint, lab)

        row = {
            **scan,
            "case_id": case_id,
            "case_submitter_id": case_submitter_id,
            "label": int(lab),
            status_col: stat,
            "label_source": src,
            "endpoint": endpoint,
        }
        meta_rows.append(row)
        label_rows.append(
            {
                "slide_id": row["slide_id"],
                "patient_id": row["patient_id"],
                "label": row["label"],
                status_col: row[status_col],
                "file_id": row["file_id"],
                "case_id": row["case_id"],
                "label_source": row["label_source"],
                "endpoint": endpoint,
            }
        )
        class_counter[int(lab)] += 1
        source_counter[src] += 1

    # Stable dedupe by (slide_id, file_id)
    seen = set()
    dedup_meta: List[Dict[str, object]] = []
    dedup_labels: List[Dict[str, object]] = []
    for m, l in zip(meta_rows, label_rows):
        key = (m["slide_id"], m["file_id"])
        if key in seen:
            continue
        seen.add(key)
        dedup_meta.append(m)
        dedup_labels.append(l)

    meta_cols = [
        "file_id",
        "file_name",
        "slide_id",
        "patient_id",
        "site",
        "specimen",
        "case_id",
        "case_submitter_id",
        "label",
        status_col,
        "label_source",
        "endpoint",
    ]
    label_cols = [
        "slide_id",
        "patient_id",
        "label",
        status_col,
        "file_id",
        "case_id",
        "label_source",
        "endpoint",
    ]

    prefix = out_prefix
    meta_path = out_dir / f"{prefix}_meta.csv"
    labels_path = out_dir / f"{prefix}_labels.csv"
    file_ids_path = out_dir / f"{prefix}_file_ids.txt"
    h5_uris_path = out_dir / f"{prefix}_h5_uris.txt"

    write_csv(meta_path, dedup_meta, meta_cols)
    write_csv(labels_path, dedup_labels, label_cols)

    file_ids = [str(r["file_id"]) for r in dedup_meta]
    file_ids_path.write_text("\n".join(file_ids) + ("\n" if file_ids else ""), encoding="utf-8")
    h5_uris_path.write_text(
        "\n".join(f"gs://embeddings-path/embeddings_fp32/{fid}.h5" for fid in file_ids)
        + ("\n" if file_ids else ""),
        encoding="utf-8",
    )

    print(f"Slides written: {len(dedup_meta)}")
    print(f"Unique patients: {len({r['patient_id'] for r in dedup_meta})}")
    print(f"Class distribution: {dict(sorted(class_counter.items()))} (0=negative,1=positive)")
    print(f"Label sources: {dict(source_counter)}")
    print(f"Wrote: {meta_path}")
    print(f"Wrote: {labels_path}")
    print(f"Wrote: {file_ids_path}")
    print(f"Wrote: {h5_uris_path}")


if __name__ == "__main__":
    main()
