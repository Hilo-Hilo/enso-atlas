#!/usr/bin/env python3
"""
Prepare a platinum-sensitivity labeled pool from bucket embeddings.

Modes:
  - slide labels: join on exact slide_id (strict)
  - patient labels: propagate patient label to all patient slides (expanded pool)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import pandas as pd


def specimen_type(slide_id: str) -> str:
    if "-TS" in slide_id:
        return "TS"
    if "-BS" in slide_id:
        return "BS"
    return "OTHER"


def counts(df: pd.DataFrame, col: str) -> Dict[str, int]:
    return {str(k): int(v) for k, v in df[col].value_counts().to_dict().items()}


def main() -> None:
    ap = argparse.ArgumentParser(description="Build platinum labeled pool from bucket files.")
    ap.add_argument("--labels_csv", required=True, type=str)
    ap.add_argument("--scan_csv", required=True, type=str)
    ap.add_argument("--available_file_ids", required=True, type=str)
    ap.add_argument("--out_dir", required=True, type=str)
    ap.add_argument(
        "--label_granularity",
        type=str,
        choices=["slide", "patient"],
        default="patient",
        help="Use exact slide labels or patient-level label expansion.",
    )
    ap.add_argument(
        "--max_slides",
        type=int,
        default=0,
        help="Optional cap after shuffling (0 means no cap).",
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = pd.read_csv(args.labels_csv)
    scan = pd.read_csv(args.scan_csv, usecols=["file_id", "file_name"])
    avail = set(Path(args.available_file_ids).read_text().splitlines())

    scan = scan[scan["file_id"].isin(avail)].copy()
    scan["slide_id"] = scan["file_name"].str.replace(r"\.svs$", "", regex=True)
    scan["patient_id"] = scan["slide_id"].str.split("-").str[:3].str.join("-")
    scan["site"] = scan["slide_id"].str.split("-").str[1]
    scan["specimen"] = scan["slide_id"].map(specimen_type)

    if args.label_granularity == "slide":
        merged = scan.merge(
            labels[["slide_id", "patient_id", "label", "platinum_status"]],
            on=["slide_id", "patient_id"],
            how="inner",
        )
    else:
        pat = labels[["patient_id", "label", "platinum_status"]].drop_duplicates("patient_id")
        merged = scan.merge(pat, on="patient_id", how="inner")

    merged = merged.drop_duplicates(subset=["slide_id", "file_id"]).reset_index(drop=True)
    merged = merged.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    if int(args.max_slides) > 0 and len(merged) > int(args.max_slides):
        merged = merged.iloc[: int(args.max_slides)].copy()

    pool_labels = merged[["slide_id", "patient_id", "label", "platinum_status"]].copy()
    out_prefix = f"trainval_pool_{args.label_granularity}"
    (out_dir / f"{out_prefix}_labels.csv").write_text(pool_labels.to_csv(index=False))
    (out_dir / f"{out_prefix}_file_ids.txt").write_text("\n".join(merged["file_id"]) + ("\n" if len(merged) else ""))
    (out_dir / f"{out_prefix}_h5_uris.txt").write_text(
        "\n".join(f"gs://embeddings-path/embeddings_fp32/{fid}.h5" for fid in merged["file_id"])
        + ("\n" if len(merged) else "")
    )
    (out_dir / f"{out_prefix}_meta.csv").write_text(merged.to_csv(index=False))

    print("Granularity:", args.label_granularity)
    print("Slides:", len(merged), "| patients:", merged["patient_id"].nunique())
    print("Class:", counts(merged, "label"))
    print("Specimen:", counts(merged, "specimen"))
    print("Site:", counts(merged, "site"))
    print("Wrote:", out_dir / f"{out_prefix}_labels.csv")
    print("Wrote:", out_dir / f"{out_prefix}_h5_uris.txt")


if __name__ == "__main__":
    main()

