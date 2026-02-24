#!/usr/bin/env python3
"""
Convert bucket-produced H5 embeddings to per-slide NPY format for TransMIL training.

Expected H5 layout (from embedder_32-path.py):
  - dataset: "features" (n_patches, 384)
  - dataset: "coords_level0" (n_patches, 2) preferred for API heatmaps
  - dataset: "coords" (n_patches, 2) fallback when level0 coords are unavailable
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert H5 embeddings to NPY.")
    ap.add_argument("--h5_dir", type=str, required=True, help="Directory containing <file_id>.h5 files.")
    ap.add_argument(
        "--matched_csv",
        type=str,
        required=True,
        help="CSV with columns including slide_id and file_id (e.g., matched_labels_available.csv).",
    )
    ap.add_argument(
        "--labels_csv",
        type=str,
        required=True,
        help="Subset labels CSV with column slide_id (e.g., train_subset_labels.csv).",
    )
    ap.add_argument("--out_dir", type=str, required=True, help="Output directory for <slide_id>.npy files.")
    ap.add_argument(
        "--skip_coords",
        action="store_true",
        help="Skip writing <slide_id>_coords.npy sidecars (not recommended for API heatmaps).",
    )
    ap.add_argument(
        "--max_patches",
        type=int,
        default=0,
        help="Optional cap per slide (0 = keep all patches).",
    )
    ap.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "float32"],
        help="Output dtype for npy embeddings.",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for capped patch subsampling.",
    )
    args = ap.parse_args()

    h5_dir = Path(args.h5_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    matched_raw = pd.read_csv(args.matched_csv)
    if "slide_id" not in matched_raw.columns:
        raise KeyError(f"'slide_id' column missing in {args.matched_csv}")
    file_id_col = None
    for candidate in ("file_id", "api_file_id", "bucket_file_id"):
        if candidate in matched_raw.columns:
            file_id_col = candidate
            break
    if file_id_col is None:
        raise KeyError(
            f"No file-id column found in {args.matched_csv}. "
            "Expected one of: file_id, api_file_id, bucket_file_id"
        )
    matched = (
        matched_raw.loc[:, ["slide_id", file_id_col]]
        .rename(columns={file_id_col: "file_id"})
        .drop_duplicates()
    )
    subset = pd.read_csv(args.labels_csv, usecols=["slide_id"]).drop_duplicates()
    plan = subset.merge(matched, on="slide_id", how="inner")
    rng = np.random.RandomState(args.seed)
    out_dtype = np.float16 if args.dtype == "float16" else np.float32

    converted = 0
    missing = 0
    skipped = 0
    coords_written = 0
    coords_missing = 0
    coords_mismatch = 0
    for row in plan.itertuples(index=False):
        slide_id = str(row.slide_id)
        file_id = str(row.file_id)
        h5_path = h5_dir / f"{file_id}.h5"
        out_path = out_dir / f"{slide_id}.npy"
        coords_out_path = out_dir / f"{slide_id}_coords.npy"

        if not h5_path.exists():
            missing += 1
            continue

        with h5py.File(h5_path, "r") as f:
            if "features" not in f:
                raise KeyError(f"'features' dataset missing in {h5_path}")
            feats = f["features"][:]
            coords = None
            if not args.skip_coords:
                if "coords_level0" in f:
                    coords = f["coords_level0"][:]
                elif "coords" in f:
                    coords = f["coords"][:]

        if feats.ndim != 2 or feats.shape[0] == 0:
            skipped += 1
            continue

        if coords is not None:
            if coords.ndim != 2 or coords.shape[1] != 2:
                print(f"[warn] Invalid coords shape for {slide_id} in {h5_path}: {coords.shape}")
                coords = None
                coords_missing += 1
            elif coords.shape[0] != feats.shape[0]:
                print(
                    f"[warn] Feature/coords length mismatch for {slide_id} in {h5_path}: "
                    f"features={feats.shape[0]} coords={coords.shape[0]}"
                )
                coords = None
                coords_mismatch += 1
        elif not args.skip_coords:
            coords_missing += 1

        if args.max_patches > 0 and feats.shape[0] > args.max_patches:
            idx = rng.choice(feats.shape[0], size=args.max_patches, replace=False)
            feats = feats[idx]
            if coords is not None:
                coords = coords[idx]

        feats = feats.astype(out_dtype, copy=False)

        np.save(out_path, feats)
        if coords is not None:
            np.save(coords_out_path, coords.astype(np.int32, copy=False))
            coords_written += 1
        converted += 1

    print(f"Converted: {converted}")
    print(f"Missing H5: {missing}")
    print(f"Skipped empty/invalid: {skipped}")
    if args.skip_coords:
        print("Coords export: skipped by --skip_coords")
    else:
        print(f"Coords written: {coords_written}")
        print(f"Coords missing in H5: {coords_missing}")
        print(f"Coords mismatched length: {coords_mismatch}")
    print(f"Output dir: {out_dir}")


if __name__ == "__main__":
    main()
