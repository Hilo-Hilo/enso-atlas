#!/usr/bin/env python3
"""
Build an API-derived expanded platinum-sensitivity training pool from GDC + bucket manifests.

Labeling policy (case-level, then propagated to all case slides in bucket):
  1) Strict (preferred): if platinum treatment_outcome exists
     - sensitive (1): Complete Response / Partial Response only
     - resistant (0): Progressive Disease / Stable Disease / No Response / Resistant / Refractory only
  2) Expanded fallback: for platinum-exposed cases without strict label
     - resistant (0): any follow-up disease_response = WT-With Tumor
     - sensitive (1): otherwise any follow-up disease_response = TF-Tumor Free

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

PLATINUM_WORDS = (
    "carboplatin",
    "cisplatin",
    "oxaliplatin",
    "nedaplatin",
    "satraplatin",
    "lobaplatin",
    "heptaplatin",
)
POS_OUTCOMES = {"complete response", "partial response"}
NEG_OUTCOMES = {"progressive disease", "stable disease", "no response", "resistant", "refractory"}

FIELDS = [
    "file_id",
    "file_name",
    "cases.case_id",
    "cases.submitter_id",
    "cases.project.project_id",
    "cases.diagnoses.treatments.therapeutic_agents",
    "cases.diagnoses.treatments.treatment_outcome",
    "cases.follow_ups.disease_response",
]


def normalize(s: object) -> str:
    return re.sub(r"\s+", " ", str(s or "").strip().lower())


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
    q = urllib.parse.urlencode(params)
    with urllib.request.urlopen(f"{GDC_FILES_URL}?{q}", timeout=180) as resp:
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
        pag = data.get("pagination", {})
        if total is None:
            total = int(pag.get("total", 0))
            print(f"GDC OV SVS total={total}")

        if not hits:
            break

        for h in hits:
            yield h

        start += size
        if start >= total:
            break


def _platinum_treatments(case_obj: Dict) -> List[Dict]:
    out = []
    for diag in case_obj.get("diagnoses") or []:
        for tr in diag.get("treatments") or []:
            ag = normalize(tr.get("therapeutic_agents"))
            if ag and any(k in ag for k in PLATINUM_WORDS):
                out.append(tr)
    return out


def infer_platinum_label(case_obj: Dict) -> Tuple[Optional[int], str]:
    """
    Return (label, source) where source is one of:
      strict_outcome, expanded_followup, unlabeled
    """
    plats = _platinum_treatments(case_obj)
    if not plats:
        return None, "unlabeled"

    pos = False
    neg = False
    for tr in plats:
        out = normalize(tr.get("treatment_outcome"))
        if out in POS_OUTCOMES:
            pos = True
        if out in NEG_OUTCOMES:
            neg = True
    if pos and not neg:
        return 1, "strict_outcome"
    if neg and not pos:
        return 0, "strict_outcome"

    # Expanded fallback from follow-up disease response.
    has_wt = False
    has_tf = False
    for fu in case_obj.get("follow_ups") or []:
        dr = normalize(fu.get("disease_response"))
        if dr.startswith("wt-") or dr == "wt-with tumor" or dr == "with tumor":
            has_wt = True
        if dr.startswith("tf-") or dr == "tf-tumor free" or dr == "tumor free":
            has_tf = True

    if has_wt:
        return 0, "expanded_followup"
    if has_tf:
        return 1, "expanded_followup"
    return None, "unlabeled"


def read_available_file_ids(path: Path) -> set:
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def read_bucket_scan(path: Path, available: set) -> Dict[str, Dict[str, str]]:
    by_file_id: Dict[str, Dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            fid = (row.get("file_id") or "").strip()
            if not fid or fid not in available:
                continue
            fname = (row.get("file_name") or "").strip()
            slide_id = fname.rsplit(".", 1)[0] if "." in fname else fname
            patient_id = patient_id_from_slide(slide_id)
            by_file_id[fid] = {
                "file_id": fid,
                "file_name": fname,
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
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare API-expanded OV platinum pool from bucket manifests.")
    ap.add_argument("--scan_csv", type=str, required=True, help="bucket_physical_scan.csv path")
    ap.add_argument("--available_file_ids", type=str, required=True, help="available_file_ids.txt path")
    ap.add_argument("--out_dir", type=str, required=True, help="output directory")
    ap.add_argument("--out_prefix", type=str, default="trainval_pool_api_expanded")
    args = ap.parse_args()

    scan_csv = Path(args.scan_csv)
    avail_path = Path(args.available_file_ids)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    available = read_available_file_ids(avail_path)
    scan_by_id = read_bucket_scan(scan_csv, available)
    print(f"Bucket available file_ids in scan: {len(scan_by_id)}")

    # Build case-level labels from API.
    case_label: Dict[str, Tuple[int, str, str]] = {}
    # case_id -> (label, source, case_submitter_id)
    seen_hits = 0
    used_hits = 0
    for hit in iter_tcga_ov_svs():
        seen_hits += 1
        fid = (hit.get("file_id") or hit.get("id") or "").strip()
        if fid not in scan_by_id:
            continue
        used_hits += 1
        cases = hit.get("cases") or []
        if not cases:
            continue
        c = cases[0]
        cid = (c.get("case_id") or "").strip()
        if not cid:
            continue
        if cid not in case_label:
            lab, src = infer_platinum_label(c)
            if lab is not None:
                case_label[cid] = (lab, src, (c.get("submitter_id") or "").strip())

    print(f"GDC hits scanned={seen_hits}, bucket-overlap hits={used_hits}, labeled cases={len(case_label)}")

    # Build slide-level rows by propagating case label to each bucket slide.
    meta_rows: List[Dict[str, object]] = []
    label_rows: List[Dict[str, object]] = []
    class_counter = Counter()
    source_counter = Counter()

    # We need case_id per file_id, so query API hits again and emit rows directly.
    # (kept simple; dataset size is modest for OV)
    for hit in iter_tcga_ov_svs():
        fid = (hit.get("file_id") or hit.get("id") or "").strip()
        scan = scan_by_id.get(fid)
        if scan is None:
            continue
        cases = hit.get("cases") or []
        if not cases:
            continue
        c = cases[0]
        cid = (c.get("case_id") or "").strip()
        if not cid or cid not in case_label:
            continue

        lab, src, case_submitter_id = case_label[cid]
        platinum_status = "sensitive" if int(lab) == 1 else "resistant"

        row = {
            **scan,
            "case_id": cid,
            "case_submitter_id": case_submitter_id,
            "label": int(lab),
            "platinum_status": platinum_status,
            "label_source": src,
        }
        meta_rows.append(row)
        label_rows.append(
            {
                "slide_id": row["slide_id"],
                "patient_id": row["patient_id"],
                "label": row["label"],
                "platinum_status": row["platinum_status"],
                "file_id": row["file_id"],
                "case_id": row["case_id"],
                "label_source": row["label_source"],
            }
        )
        class_counter[int(lab)] += 1
        source_counter[src] += 1

    # De-duplicate by (slide_id, file_id) in stable order.
    seen = set()
    dedup_meta = []
    dedup_labels = []
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
        "platinum_status",
        "label_source",
    ]
    label_cols = [
        "slide_id",
        "patient_id",
        "label",
        "platinum_status",
        "file_id",
        "case_id",
        "label_source",
    ]

    prefix = args.out_prefix
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
    print(f"Class distribution: {dict(sorted(class_counter.items()))}")
    print(f"Label sources: {dict(source_counter)}")
    print(f"Wrote: {meta_path}")
    print(f"Wrote: {labels_path}")
    print(f"Wrote: {file_ids_path}")
    print(f"Wrote: {h5_uris_path}")


if __name__ == "__main__":
    main()

