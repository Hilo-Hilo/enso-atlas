#!/usr/bin/env python3
"""
Rebuild multi-model training pools by matching bucket slides to current GDC records via slide barcode,
then create class-balanced labels by undersampling the majority class (no augmentation).

Why barcode matching:
  - GDC file UUIDs can change over time.
  - Embeddings in bucket are keyed by historical UUID/file_id.
  - TCGA slide barcode in file_name is stable and used to refresh labels from current API records.

Outputs (per model):
  - <prefix>_meta.csv
  - <prefix>_labels.csv
  - <prefix>_file_ids.txt
  - <prefix>_h5_uris.txt
  - <prefix>_labels_balanced.csv
  - <prefix>_file_ids_balanced.txt
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple


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


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    project_id: str
    prefix: str
    embeddings_dir: str
    status_col: str
    status_pos: str
    status_neg: str
    infer_label: Callable[[Dict], Tuple[Optional[int], str]]


def normalize(s: object) -> str:
    return re.sub(r"\s+", " ", str(s or "").strip().lower())


def _try_float(v: object) -> Optional[float]:
    try:
        if v is None or v == "":
            return None
        return float(v)
    except Exception:
        return None


def barcode_from_filename(file_name: str) -> str:
    # Example:
    #   TCGA-XX-YYYY-01Z-00-DX1.svs -> TCGA-XX-YYYY-01Z-00-DX1
    #   TCGA-XX-YYYY-01Z-00-DX1.1234.svs -> TCGA-XX-YYYY-01Z-00-DX1
    base = Path(file_name).name
    return base.split(".", 1)[0]


def patient_id_from_slide(slide_id: str) -> str:
    parts = slide_id.split("-")
    if len(parts) >= 3 and parts[0] == "TCGA":
        return "-".join(parts[:3])
    return slide_id


def specimen_type(slide_id: str) -> str:
    if "-TS" in slide_id:
        return "TS"
    if "-BS" in slide_id:
        return "BS"
    return "OTHER"


def gdc_get_json(params: Dict[str, str]) -> Dict:
    query = urllib.parse.urlencode(params)
    with urllib.request.urlopen(f"{GDC_FILES_URL}?{query}", timeout=180) as resp:
        return json.loads(resp.read().decode("utf-8"))


def iter_tcga_svs(project_id: str, fields: Sequence[str]) -> Iterable[Dict]:
    filters = {
        "op": "and",
        "content": [
            {"op": "=", "content": {"field": "files.data_format", "value": ["SVS"]}},
            {"op": "=", "content": {"field": "cases.project.project_id", "value": [project_id]}},
        ],
    }
    size = 2000
    start = 0
    total = None
    while True:
        params = {
            "filters": json.dumps(filters),
            "fields": ",".join(fields),
            "format": "JSON",
            "size": str(size),
            "from": str(start),
        }
        payload = gdc_get_json(params)
        warnings = payload.get("warnings")
        if warnings:
            print(f"WARN({project_id}): {warnings}")
        data = payload.get("data", {})
        hits = data.get("hits", [])
        pag = data.get("pagination", {})
        if total is None:
            total = int(pag.get("total", 0))
            print(f"GDC {project_id} SVS total={total}")
        if not hits:
            break
        for h in hits:
            yield h
        start += size
        if start >= total:
            break


def choose_best_hit(hits: Sequence[Dict]) -> Optional[Dict]:
    if not hits:
        return None

    def key(h: Dict) -> Tuple[int, int, str, int]:
        access = normalize(h.get("access")) == "open"
        state = normalize(h.get("state")) == "released"
        created = str(h.get("created_datetime") or "")
        size = int(h.get("file_size") or 0)
        return (1 if access else 0, 1 if state else 0, created, size)

    return sorted(hits, key=key, reverse=True)[0]


def read_available_file_ids(path: Path) -> set:
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def read_bucket_scan(path: Path, available: set) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            file_id = (row.get("file_id") or "").strip()
            file_name = (row.get("file_name") or "").strip()
            full_path = (row.get("full_path") or "").strip()
            if not file_id or file_id not in available:
                continue
            if not file_name:
                continue
            slide_id = Path(file_name).stem
            slide_barcode = barcode_from_filename(file_name)
            patient_id = patient_id_from_slide(slide_barcode)
            rows.append(
                {
                    "bucket_file_id": file_id,
                    "file_name": file_name,
                    "full_path": full_path,
                    "slide_id": slide_id,
                    "slide_barcode": slide_barcode,
                    "patient_id": patient_id,
                    "site": patient_id.split("-")[1] if patient_id.startswith("TCGA-") and len(patient_id.split("-")) >= 3 else "",
                    "specimen": specimen_type(slide_barcode),
                }
            )
    return rows


def _platinum_treatments(case_obj: Dict) -> List[Dict]:
    out: List[Dict] = []
    for diag in case_obj.get("diagnoses") or []:
        for tr in diag.get("treatments") or []:
            agents = normalize(tr.get("therapeutic_agents"))
            if agents and any(k in agents for k in PLATINUM_WORDS):
                out.append(tr)
    return out


def infer_platinum_label(case_obj: Dict) -> Tuple[Optional[int], str]:
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


def infer_survival_label(case_obj: Dict, years: int) -> Tuple[Optional[int], str]:
    threshold_days = float(years) * 365.25
    demo = case_obj.get("demographic") or {}
    vital = normalize(demo.get("vital_status"))
    days_to_death = _try_float(demo.get("days_to_death"))

    follow_up_days: Optional[float] = None
    for diag in case_obj.get("diagnoses") or []:
        v = _try_float(diag.get("days_to_last_follow_up"))
        if v is not None and (follow_up_days is None or v > follow_up_days):
            follow_up_days = v
    for fu in case_obj.get("follow_ups") or []:
        v = _try_float(fu.get("days_to_follow_up"))
        if v is not None and (follow_up_days is None or v > follow_up_days):
            follow_up_days = v

    if vital == "dead" and days_to_death is not None:
        return (1 if days_to_death > threshold_days else 0), "demographic.days_to_death"
    if vital == "alive" and follow_up_days is not None and follow_up_days >= threshold_days:
        return 1, "follow_up_censoring"
    return None, "unlabeled"


def infer_survival_1y_label(case_obj: Dict) -> Tuple[Optional[int], str]:
    return infer_survival_label(case_obj, years=1)


def infer_survival_3y_label(case_obj: Dict) -> Tuple[Optional[int], str]:
    return infer_survival_label(case_obj, years=3)


def infer_survival_5y_label(case_obj: Dict) -> Tuple[Optional[int], str]:
    return infer_survival_label(case_obj, years=5)


def infer_tumor_grade_label(case_obj: Dict) -> Tuple[Optional[int], str]:
    grades: List[str] = []
    for diag in case_obj.get("diagnoses") or []:
        g = normalize(diag.get("tumor_grade"))
        if g:
            grades.append(g)
    if not grades:
        return None, "unlabeled"
    has_high = any(g in {"g3", "g4", "gb"} or "g3" in g or "g4" in g for g in grades)
    has_low = any(g in {"g1", "g2"} or "g1" in g or "g2" in g for g in grades)
    if has_high:
        return 1, "diagnoses.tumor_grade"
    if has_low:
        return 0, "diagnoses.tumor_grade"
    return None, "unlabeled"


def infer_lung_stage_label(case_obj: Dict) -> Tuple[Optional[int], str]:
    stages: List[str] = []
    for diag in case_obj.get("diagnoses") or []:
        st = normalize(diag.get("ajcc_pathologic_stage"))
        if st and st not in {"not reported", "unknown", "stage x"}:
            stages.append(st)
    if not stages:
        return None, "unlabeled"
    for st in stages:
        if "stage iii" in st or "stage iv" in st or re.search(r"\biii[a-z]*\b", st) or re.search(r"\biv[a-z]*\b", st):
            return 1, "ajcc_pathologic_stage"
    for st in stages:
        if "stage i" in st or "stage ii" in st or re.search(r"\bi[a-z]*\b", st) or re.search(r"\bii[a-z]*\b", st):
            return 0, "ajcc_pathologic_stage"
    return None, "unlabeled"


def write_csv(path: Path, rows: List[Dict[str, object]], cols: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cols))
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})


def dedup_rows(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    seen = set()
    for r in rows:
        key = (str(r.get("slide_id", "")), str(r.get("bucket_file_id", "")))
        if key in seen:
            continue
        seen.add(key)
        out.append(dict(r))
    return out


def _sample_majority_stratified(
    maj_rows: List[Dict[str, object]],
    min_rows: List[Dict[str, object]],
    target_n: int,
    rng: random.Random,
) -> List[Dict[str, object]]:
    if target_n >= len(maj_rows):
        return list(maj_rows)

    strata_cols = [c for c in ("specimen", "site") if any(str(r.get(c, "")) for r in (maj_rows + min_rows))]
    if not strata_cols:
        return rng.sample(maj_rows, target_n)

    def key_fn(r: Dict[str, object]) -> Tuple[str, ...]:
        return tuple(str(r.get(c, "")) for c in strata_cols)

    maj_by = defaultdict(list)
    for r in maj_rows:
        maj_by[key_fn(r)].append(r)
    min_by = Counter(key_fn(r) for r in min_rows)

    picked: List[Dict[str, object]] = []
    used_ids = set()

    # First pass: try to mirror minority strata counts.
    for k, want in min_by.items():
        pool = maj_by.get(k, [])
        if not pool:
            continue
        take = min(len(pool), int(want))
        chosen = rng.sample(pool, take) if take < len(pool) else list(pool)
        for r in chosen:
            rid = id(r)
            if rid not in used_ids:
                used_ids.add(rid)
                picked.append(r)

    # Fill remainder from all remaining majority rows.
    if len(picked) < target_n:
        remaining = [r for r in maj_rows if id(r) not in used_ids]
        need = target_n - len(picked)
        if need > 0 and remaining:
            extra = rng.sample(remaining, need) if need < len(remaining) else remaining
            picked.extend(extra)

    # Rare edge case: overshoot due duplicated keys / bookkeeping.
    if len(picked) > target_n:
        picked = rng.sample(picked, target_n)
    return picked


def create_balanced_rows(
    rows: Sequence[Dict[str, object]],
    embeddings_dir: Path,
    seed: int,
) -> Tuple[List[Dict[str, object]], Dict[str, int], Dict[str, int]]:
    have_stems = {p.stem for p in embeddings_dir.glob("*.npy")}
    have_stems_l = {s.lower() for s in have_stems}

    def _match_embedding(row: Dict[str, object]) -> bool:
        bucket_file_id = str(row.get("bucket_file_id", "")).strip()
        slide_id = str(row.get("slide_id", "")).strip()
        file_name = str(row.get("file_name", "")).strip()
        file_name_stem = Path(file_name).stem if file_name else ""
        candidates = [bucket_file_id, slide_id, file_name_stem]
        for c in candidates:
            if not c:
                continue
            if c in have_stems or c.lower() in have_stems_l:
                return True
        return False

    filtered = [dict(r) for r in rows if _match_embedding(r)]
    filtered = dedup_rows(filtered)

    pre = Counter(int(r["label"]) for r in filtered if str(r.get("label", "")).strip() != "")
    if not pre:
        return [], dict(pre), {}
    if len(pre) < 2:
        return filtered, dict(pre), dict(pre)

    labels_sorted = sorted(pre.items(), key=lambda kv: kv[1])
    minority_label, minority_n = labels_sorted[0]
    majority_label, _ = labels_sorted[-1]

    min_rows = [r for r in filtered if int(r["label"]) == minority_label]
    maj_rows = [r for r in filtered if int(r["label"]) == majority_label]

    rng = random.Random(seed)
    maj_sampled = _sample_majority_stratified(maj_rows, min_rows, minority_n, rng)
    balanced = min_rows + maj_sampled
    rng.shuffle(balanced)

    post = Counter(int(r["label"]) for r in balanced if str(r.get("label", "")).strip() != "")
    return balanced, dict(pre), dict(post)


def build_barcode_map(project_id: str, fields: Sequence[str]) -> Dict[str, Dict]:
    by_barcode: Dict[str, List[Dict]] = defaultdict(list)
    n = 0
    for h in iter_tcga_svs(project_id, fields):
        n += 1
        file_name = str(h.get("file_name") or "")
        if not file_name:
            continue
        bc = barcode_from_filename(file_name)
        by_barcode[bc].append(h)
    out: Dict[str, Dict] = {}
    for bc, hits in by_barcode.items():
        best = choose_best_hit(hits)
        if best is not None:
            out[bc] = best
    print(f"{project_id}: API hits={n}, unique_barcodes={len(by_barcode)}, mapped_barcodes={len(out)}")
    return out


def materialize_pool_for_model(
    spec: ModelSpec,
    scan_rows: Sequence[Dict[str, str]],
    barcode_map: Dict[str, Dict],
    out_dir: Path,
    seed: int,
) -> None:
    rows: List[Dict[str, object]] = []
    missing_barcode = 0
    unlabeled = 0
    for s in scan_rows:
        barcode = str(s.get("slide_barcode") or "")
        hit = barcode_map.get(barcode)
        if hit is None:
            missing_barcode += 1
            continue
        cases = hit.get("cases") or []
        if not cases:
            unlabeled += 1
            continue
        case_obj = cases[0]
        lab, src = spec.infer_label(case_obj)
        if lab is None:
            unlabeled += 1
            continue

        status_val = spec.status_pos if int(lab) == 1 else spec.status_neg
        case_id = (case_obj.get("case_id") or "").strip()
        case_submitter_id = (case_obj.get("submitter_id") or "").strip()
        api_file_id = (hit.get("file_id") or hit.get("id") or "").strip()
        rows.append(
            {
                "slide_id": s["slide_id"],
                "slide_barcode": s["slide_barcode"],
                "patient_id": s["patient_id"],
                "label": int(lab),
                spec.status_col: status_val,
                "bucket_file_id": s["bucket_file_id"],
                "api_file_id": api_file_id,
                "case_id": case_id,
                "case_submitter_id": case_submitter_id,
                "site": s["site"],
                "specimen": s["specimen"],
                "file_name": s["file_name"],
                "full_path": s["full_path"],
                "label_source": src,
                "project_id": spec.project_id,
                "model_id": spec.model_id,
            }
        )

    rows = dedup_rows(rows)
    cls = Counter(int(r["label"]) for r in rows)
    src = Counter(str(r.get("label_source", "")) for r in rows)
    pats = {str(r.get("patient_id", "")) for r in rows if str(r.get("patient_id", ""))}
    print(
        f"{spec.model_id}: rows={len(rows)} patients={len(pats)} class={dict(sorted(cls.items()))} "
        f"missing_barcode={missing_barcode} unlabeled={unlabeled} sources={dict(src)}"
    )

    meta_cols = [
        "bucket_file_id",
        "api_file_id",
        "file_name",
        "full_path",
        "slide_id",
        "slide_barcode",
        "patient_id",
        "site",
        "specimen",
        "case_id",
        "case_submitter_id",
        "label",
        spec.status_col,
        "label_source",
        "project_id",
        "model_id",
    ]
    labels_cols = [
        "slide_id",
        "slide_barcode",
        "patient_id",
        "label",
        spec.status_col,
        "bucket_file_id",
        "api_file_id",
        "case_id",
        "site",
        "specimen",
        "label_source",
    ]

    prefix = spec.prefix
    meta_path = out_dir / f"{prefix}_meta.csv"
    labels_path = out_dir / f"{prefix}_labels.csv"
    file_ids_path = out_dir / f"{prefix}_file_ids.txt"
    h5_uris_path = out_dir / f"{prefix}_h5_uris.txt"
    labels_bal_path = out_dir / f"{prefix}_labels_balanced.csv"
    file_ids_bal_path = out_dir / f"{prefix}_file_ids_balanced.txt"

    write_csv(meta_path, rows, meta_cols)
    write_csv(labels_path, rows, labels_cols)

    file_ids = [str(r["bucket_file_id"]) for r in rows]
    file_ids_path.write_text("\n".join(file_ids) + ("\n" if file_ids else ""), encoding="utf-8")
    h5_uris_path.write_text(
        "\n".join(f"gs://embeddings-path/embeddings_fp32/{fid}.h5" for fid in file_ids)
        + ("\n" if file_ids else ""),
        encoding="utf-8",
    )

    emb_dir = Path(spec.embeddings_dir)
    balanced_rows, pre_cls, post_cls = create_balanced_rows(rows, emb_dir, seed=seed)
    write_csv(labels_bal_path, balanced_rows, labels_cols)
    bal_ids = [str(r["bucket_file_id"]) for r in balanced_rows]
    file_ids_bal_path.write_text("\n".join(bal_ids) + ("\n" if bal_ids else ""), encoding="utf-8")
    print(
        f"{spec.model_id}: balanced_rows={len(balanced_rows)} pre={pre_cls} post={post_cls} "
        f"(embeddings_dir={emb_dir})"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Rebuild all 6 model pools with barcode API remap + balanced labels.")
    ap.add_argument("--scan_csv", required=True, help="bucket_physical_scan.csv")
    ap.add_argument("--available_file_ids", required=True, help="available_file_ids.txt")
    ap.add_argument("--out_dir", required=True, help="Output directory for pool files")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    available = read_available_file_ids(Path(args.available_file_ids))
    scan_rows = read_bucket_scan(Path(args.scan_csv), available)
    print(f"Scan rows with available file_id: {len(scan_rows)}")

    # Build per-project barcode maps with all fields needed by downstream label functions.
    ov_fields = [
        "file_id",
        "file_name",
        "access",
        "state",
        "created_datetime",
        "file_size",
        "cases.case_id",
        "cases.submitter_id",
        "cases.project.project_id",
        "cases.demographic.vital_status",
        "cases.demographic.days_to_death",
        "cases.diagnoses.days_to_last_follow_up",
        "cases.diagnoses.tumor_grade",
        "cases.diagnoses.treatments.therapeutic_agents",
        "cases.diagnoses.treatments.treatment_outcome",
        "cases.follow_ups.disease_response",
        "cases.follow_ups.days_to_follow_up",
    ]
    luad_fields = [
        "file_id",
        "file_name",
        "access",
        "state",
        "created_datetime",
        "file_size",
        "cases.case_id",
        "cases.submitter_id",
        "cases.project.project_id",
        "cases.diagnoses.ajcc_pathologic_stage",
    ]

    ov_map = build_barcode_map("TCGA-OV", ov_fields)
    luad_map = build_barcode_map("TCGA-LUAD", luad_fields)

    models = [
        ModelSpec(
            model_id="platinum_sensitivity",
            project_id="TCGA-OV",
            prefix="trainval_pool_ov_platinum_api_barcode_v2",
            embeddings_dir="data/tcga_full/bucket_training/npy_pool_api_expanded",
            status_col="platinum_status",
            status_pos="sensitive",
            status_neg="resistant",
            infer_label=infer_platinum_label,
        ),
        ModelSpec(
            model_id="tumor_grade",
            project_id="TCGA-OV",
            prefix="trainval_pool_ov_tumor_grade_api_barcode_v2",
            embeddings_dir="data/tcga_full/bucket_training/npy_pool_api_expanded",
            status_col="grade_status",
            status_pos="high",
            status_neg="low",
            infer_label=infer_tumor_grade_label,
        ),
        ModelSpec(
            model_id="survival_1y",
            project_id="TCGA-OV",
            prefix="trainval_pool_ov_survival_1y_api_barcode_v2",
            embeddings_dir="data/tcga_full/bucket_training/npy_pool_api_expanded",
            status_col="survival_1y_status",
            status_pos="survived",
            status_neg="deceased",
            infer_label=infer_survival_1y_label,
        ),
        ModelSpec(
            model_id="survival_3y",
            project_id="TCGA-OV",
            prefix="trainval_pool_ov_survival_3y_api_barcode_v2",
            embeddings_dir="data/tcga_full/bucket_training/npy_pool_api_expanded",
            status_col="survival_3y_status",
            status_pos="survived",
            status_neg="deceased",
            infer_label=infer_survival_3y_label,
        ),
        ModelSpec(
            model_id="survival_5y",
            project_id="TCGA-OV",
            prefix="trainval_pool_ov_survival_5y_api_barcode_v2",
            embeddings_dir="data/tcga_full/bucket_training/npy_pool_api_expanded",
            status_col="survival_5y_status",
            status_pos="survived",
            status_neg="deceased",
            infer_label=infer_survival_5y_label,
        ),
        ModelSpec(
            model_id="lung_stage",
            project_id="TCGA-LUAD",
            prefix="trainval_pool_luad_stage_api_barcode_v2",
            embeddings_dir="data/tcga_full/bucket_training/npy_pool_luad_stage_api",
            status_col="stage_status",
            status_pos="advanced",
            status_neg="early",
            infer_label=infer_lung_stage_label,
        ),
    ]

    for spec in models:
        barcode_map = ov_map if spec.project_id == "TCGA-OV" else luad_map
        materialize_pool_for_model(spec, scan_rows, barcode_map, out_dir=out_dir, seed=args.seed)

    print("Done.")


if __name__ == "__main__":
    main()
