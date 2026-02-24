#!/usr/bin/env python3
"""
Prepare a capped, stratified platinum-sensitivity training subset from bucket embeddings.

Inputs:
  - labels CSV with columns: slide_id, patient_id, label, platinum_status
  - bucket physical scan CSV with columns: file_id, file_name
  - available file_id list from GCS (`embeddings_fp32/*.h5`)

Outputs:
  - matched_labels_available.csv
  - train_subset_labels.csv
  - train_subset_file_ids.txt
  - train_subset_h5_uris.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd


def specimen_type(slide_id: str) -> str:
    if "-TS" in slide_id:
        return "TS"
    if "-BS" in slide_id:
        return "BS"
    return "OTHER"


def proportional_group_sample(
    df: pd.DataFrame,
    group_col: str,
    target_n: int,
    *,
    seed: int,
) -> pd.DataFrame:
    """Sample target_n rows using proportional allocation across groups."""
    if target_n <= 0 or len(df) == 0:
        return df.iloc[0:0].copy()
    if target_n >= len(df):
        return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    rng = np.random.RandomState(seed)
    group_sizes = df.groupby(group_col).size().sort_values(ascending=False)
    frac = group_sizes / group_sizes.sum()
    alloc = np.floor(frac * target_n).astype(int)

    # Keep representation for sparse groups where possible.
    for g in alloc.index:
        if alloc[g] == 0:
            alloc[g] = 1

    while int(alloc.sum()) > target_n:
        g = alloc.idxmax()
        alloc[g] -= 1

    while int(alloc.sum()) < target_n:
        remaining = (group_sizes - alloc).clip(lower=0)
        if int(remaining.sum()) <= 0:
            break
        g = remaining.idxmax()
        alloc[g] += 1

    parts = []
    for g, n in alloc.items():
        if n <= 0:
            continue
        chunk = df[df[group_col] == g]
        n = int(min(n, len(chunk)))
        if n > 0:
            parts.append(chunk.sample(n=n, random_state=int(rng.randint(0, 1_000_000))))

    sampled = pd.concat(parts, ignore_index=True) if parts else df.iloc[0:0].copy()

    # In rare rounding paths we can still miss/exceed by a few rows.
    if len(sampled) > target_n:
        sampled = sampled.sample(n=target_n, random_state=seed).reset_index(drop=True)
    elif len(sampled) < target_n:
        missing = target_n - len(sampled)
        remaining = df.drop(index=sampled.index, errors="ignore")
        if len(remaining) > 0:
            add_n = min(missing, len(remaining))
            add_df = remaining.sample(n=add_n, random_state=seed + 1)
            sampled = pd.concat([sampled, add_df], ignore_index=True)
    return sampled.reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare capped platinum subset from bucket embeddings.")
    ap.add_argument("--labels_csv", required=True, type=str)
    ap.add_argument("--scan_csv", required=True, type=str)
    ap.add_argument("--available_file_ids", required=True, type=str)
    ap.add_argument("--out_dir", required=True, type=str)
    ap.add_argument("--max_slides", type=int, default=60)
    ap.add_argument("--pos_to_neg_ratio", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = pd.read_csv(args.labels_csv)
    scan = pd.read_csv(args.scan_csv, usecols=["file_id", "file_name"])
    available_ids = set(Path(args.available_file_ids).read_text().splitlines())

    scan = scan[scan["file_id"].isin(available_ids)].copy()
    scan["slide_id"] = scan["file_name"].str.replace(r"\.svs$", "", regex=True)

    merged = labels.merge(scan[["slide_id", "file_id"]], on="slide_id", how="inner").copy()
    merged["site"] = merged["slide_id"].str.split("-").str[1]
    merged["specimen"] = merged["slide_id"].map(specimen_type)
    merged["strat_key"] = merged["site"].astype(str) + "|" + merged["specimen"].astype(str)

    neg = merged[merged["label"] == 0].copy()
    pos = merged[merged["label"] == 1].copy()

    # Keep all negatives, cap positives to target ratio and overall max.
    max_pos_by_ratio = int(np.floor(args.pos_to_neg_ratio * len(neg)))
    max_pos_by_total = max(int(args.max_slides) - len(neg), 0)
    target_pos = min(len(pos), max_pos_by_ratio, max_pos_by_total)

    pos_sel = proportional_group_sample(pos, group_col="strat_key", target_n=target_pos, seed=args.seed)
    subset = pd.concat([neg, pos_sel], ignore_index=True)
    subset = subset.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    # Persist
    (out_dir / "matched_labels_available.csv").write_text(merged.to_csv(index=False))
    (out_dir / "train_subset_labels.csv").write_text(
        subset[["slide_id", "patient_id", "label", "platinum_status"]].to_csv(index=False)
    )
    (out_dir / "train_subset_file_ids.txt").write_text(
        "\n".join(subset["file_id"].tolist()) + ("\n" if len(subset) else "")
    )
    (out_dir / "train_subset_h5_uris.txt").write_text(
        "\n".join(f"gs://embeddings-path/embeddings_fp32/{fid}.h5" for fid in subset["file_id"])
        + ("\n" if len(subset) else "")
    )

    def counts(df: pd.DataFrame, col: str) -> Dict[str, int]:
        return {str(k): int(v) for k, v in df[col].value_counts().to_dict().items()}

    print("Matched slides:", len(merged), "| patients:", merged["patient_id"].nunique(), "| class:", counts(merged, "label"))
    print("Subset slides:", len(subset), "| patients:", subset["patient_id"].nunique(), "| class:", counts(subset, "label"))
    print("Subset specimen:", counts(subset, "specimen"))
    print("Subset site:", counts(subset, "site"))
    print("Wrote:", out_dir / "train_subset_labels.csv")
    print("Wrote:", out_dir / "train_subset_h5_uris.txt")


if __name__ == "__main__":
    main()

