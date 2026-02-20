# Enso Atlas WSI Datasets

This document describes the datasets used and considered for Enso Atlas.

---

## Primary Project Dataset: TCGA Ovarian Cancer (TCGA-OV)

**Project:** `ovarian-platinum`

### Overview

- **Source:** Genomic Data Commons (GDC) / The Cancer Genome Atlas
- **Project ID:** TCGA-OV
- **Slides Used:** 208 whole-slide images with platinum sensitivity labels
- **Format:** SVS (open access)
- **License:** CC BY 3.0

### Labels

- **Platinum Sensitivity:** Binary (Sensitive vs Resistant), derived from PLATINUM_STATUS in clinical records
- **Tumor Grade:** High vs Low grade
- **Survival:** 1-year, 3-year, 5-year overall survival (binary)

### Usage in Enso Atlas

- 208 slides with platinum sensitivity labels for the primary ovarian response model (AUC 0.907)
- Larger labeled subsets used for grade and survival heads
- Level-0 patch extraction (224x224), typically 6,000-30,000 patches/slide
- Path Foundation embeddings (384-dimensional, FP16 cache)

---

## Secondary Project Dataset: TCGA Lung Adenocarcinoma (TCGA-LUAD)

**Project:** `lung-stage`

### Overview

- **Source:** Genomic Data Commons (GDC) / The Cancer Genome Atlas
- **Project ID:** TCGA-LUAD
- **Slides Used:** 130 whole-slide images
- **Format:** SVS (open access)
- **License:** TCGA/GDC open-access terms

### Labels

- **Tumor Stage:** Binary stage grouping for classification
  - **Early:** Stage I/II
  - **Advanced:** Stage III/IV

### Usage in Enso Atlas

- 130 slides used for LUAD stage classification
- TransMIL model ID: `lung_stage`
- AUC-ROC: **0.648**
- Uses the same level-0 Path Foundation embedding pipeline and project-scoped storage/routing

---

## Blocked Dataset: Ovarian Bevacizumab Response (PathDB/TCIA)

**Originally planned as primary ovarian treatment-response dataset. Download was blocked.**

### Overview

- **Source:** The Cancer Imaging Archive (TCIA) / PathDB
- **DOI:** 10.7937/TCIA.985G-EY35
- **Subjects:** 78 patients
- **Slides:** 288 H&E WSIs (162 effective, 126 invalid)
- **Labels:** Bevacizumab treatment response (Effective vs Invalid)

### Why It Was Blocked

The PathDB download server returned 0-byte files for 217 out of 286 requested slides. Only 69 slides downloaded successfully, which was insufficient for training. Repeated attempts produced the same failure mode.

### Reference

Wang et al. (2022), *Histopathological whole slide image dataset for classification of treatment effectiveness to ovarian cancer*. Scientific Data. DOI: 10.1038/s41597-022-01127-6

---

## Other Considered Datasets

### PTRC-HGSOC (TCIA)

- 158 patients, 348 WSIs, platinum sensitivity labels
- Candidate for future multi-cohort validation

### CPTAC-OV

- 102 patients with proteomic + histopathology data
- Candidate for future multimodal extension

---

## Summary

| Dataset | Project | Status | Slides | Labels | Used |
|---------|---------|--------|--------|--------|------|
| TCGA-OV | ovarian-platinum | Available | 208+ | Platinum response, grade, survival | Yes (primary ovarian project) |
| TCGA-LUAD | lung-stage | Available | 130 | Early vs advanced stage | Yes (lung project) |
| Bevacizumab Response (PathDB/TCIA) | N/A | Blocked | 288 | Treatment response | No |
| PTRC-HGSOC | N/A | Available | 348 | Platinum sensitivity | No (future work) |
| CPTAC-OV | N/A | Available | ~200 | Survival, proteomics | No (future work) |
