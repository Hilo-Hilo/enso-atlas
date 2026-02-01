# Ovarian Cancer WSI Datasets for Treatment Response Prediction

This document catalogs publicly available ovarian cancer whole slide image (WSI) datasets with treatment response labels or survival outcomes that can be used as proxies.

---

## Summary Table

| Dataset | Subjects | WSIs | Labels | Access | Quality | Priority |
|---------|----------|------|--------|--------|---------|----------|
| Ovarian Bevacizumab Response | 78 | 288 | Treatment Response | Public | HIGH | 1 |
| PTRC-HGSOC | 158 | 348 | Chemo Sensitivity | Public | HIGH | 2 |
| TCGA-OV (GDC) | 587 | 1,481 | Survival + Response | Public | HIGH | 3 |
| CMB-OV | 47 | 74 | Clinical (dbGaP) | Public | MEDIUM | 4 |
| CPTAC-OV | 102 | ~200 | Survival | Public | MEDIUM | 5 |

---

## 1. Ovarian Bevacizumab Response (TCIA)

**HIGHLY RECOMMENDED - Best dataset for treatment response prediction**

### Overview
- **Source:** The Cancer Imaging Archive (TCIA)
- **DOI:** 10.7937/TCIA.985G-EY35
- **Size:** 253.8 GB
- **Subjects:** 78 patients
- **Slides:** 288 H&E stained WSIs (162 effective, 126 invalid)
- **Magnification:** 20x (0.5 um/pixel)
- **Format:** SVS
- **License:** CC BY 4.0

### Labels Available
- **Treatment Response:** Binary classification (Effective vs Invalid)
  - Effective: 162 slides (56.3%)
  - Invalid: 126 slides (43.7%)
- **Clinical Data:**
  - Age
  - Pre/post-treatment CA-125 levels
  - Histologic subtype
  - FIGO stage
  - Recurrence status
  - Survival status
  - Date of death (if applicable)
  - BMI
  - Number of bevacizumab doses

### Response Definition
- **Bevacizumab-resistant:** Persistently high CA-125 during therapy OR tumor progression/recurrence within 6 months post-treatment (CT/PET confirmed)
- **Bevacizumab-sensitive:** Normal CA-125 and no tumor progression/recurrence during or within 6 months of treatment

### Cancer Types
- Epithelial Ovarian Cancer (EOC): 70 patients
  - High-grade serous: 58
  - Endometrioid: 4
  - Clear cell: 7
  - Mucinous: 2
  - Unclassified adenocarcinoma: 7
- Peritoneal Serous Papillary Carcinoma (PSPC): 8 patients

### Access Method
```bash
# Download via IBM Aspera Connect
# Direct download link: https://faspex.cancerimagingarchive.net/aspera/faspex/public/package?context=...

# Clinical data (Excel):
# - Patient list: https://www.cancerimagingarchive.net/wp-content/uploads/Final-patient_list.xlsx
# - CA-125 data: https://www.cancerimagingarchive.net/wp-content/uploads/new_CA125-data_20230207.xlsx
```

### Publications
1. Wang et al. (2022) "Weakly Supervised Deep Learning for Prediction of Treatment Effectiveness on Ovarian Cancer from Histopathology Images." Computerized Medical Imaging and Graphics. DOI: 10.1016/j.compmedimag.2022.102093
2. Wang et al. (2022) "Histopathological whole slide image dataset for classification of treatment effectiveness to ovarian cancer." Scientific Data. DOI: 10.1038/s41597-022-01127-6

### Assessment
- **Quality:** HIGH - Purpose-built for treatment response prediction
- **Relevance:** EXCELLENT - Direct treatment response labels
- **Ease of Download:** MEDIUM - Requires Aspera Connect plugin
- **Recommendation:** Primary dataset for the hackathon

---

## 2. PTRC-HGSOC (TCIA)

**HIGHLY RECOMMENDED - Platinum chemotherapy response labels**

### Overview
- **Source:** The Cancer Imaging Archive (TCIA)
- **DOI:** 10.7937/6RDA-P940
- **Size:** 120.5 GB
- **Subjects:** 158 unique patients (174 tumors)
- **Slides:** 348 H&E WSIs (bookend sections)
- **Magnification:** 20x (0.5 um resolution)
- **Format:** SVS
- **License:** CC BY 4.0

### Labels Available
- **Chemotherapy Response:** Binary (Sensitive vs Refractory)
  - Refractory: Clinical ovarian cancer that progresses while on platinum-based therapy or within 4 weeks
- **Clinical Data:**
  - Neo-adjuvant treatment status
  - Tumor location
  - Tumor grade, stage, substage
  - Patient age
  - Patient ethnicity/race
  - Age of sample

### Sample Type
- Formalin-fixed, paraffin-embedded (FFPE) tissues
- Bookend sections (first and last from each FFPE block)
- HALO Image Analysis Platform validated

### Access Method
```bash
# Download via IBM Aspera Connect
# Clinical data: https://www.cancerimagingarchive.net/wp-content/uploads/PTRC-HGSOC_List_clincal_data.xlsx
```

### Publication
Chowdhury et al. (2023) "Proteogenomic analysis of chemo-refractory high-grade serous ovarian cancer." Cell. DOI: 10.1016/j.cell.2023.07.004

### Assessment
- **Quality:** HIGH - Well-annotated with proteogenomic data
- **Relevance:** EXCELLENT - Platinum sensitivity labels (standard of care)
- **Ease of Download:** MEDIUM - Requires Aspera Connect
- **Recommendation:** Secondary dataset, complements Bevacizumab Response

---

## 3. TCGA-OV (GDC Portal)

**RECOMMENDED - Largest dataset with survival and therapy outcome labels**

### Overview
- **Source:** Genomic Data Commons (GDC)
- **Project ID:** TCGA-OV
- **DOI:** 10.7937/K9/TCIA.2016.NDO1MDFQ
- **Subjects:** 587 cases (143 with radiology on TCIA)
- **Slides:** 1,481 diagnostic slide images
- **Format:** SVS (open access)
- **License:** CC BY 3.0

### Labels Available (via cBioPortal API)
- **Survival Data:**
  - `OS_MONTHS`: Overall survival in months
  - `OS_STATUS`: Living/Deceased
  - `DFS_MONTHS`: Disease-free survival in months
  - `DFS_STATUS`: Disease-free/Recurred
- **Treatment Response:**
  - `PLATINUM_STATUS`: Sensitive/Resistant/TooEarly
  - `PRIMARY_THERAPY_OUTCOME_SUCCESS`: Complete Response / Partial Response / Stable Disease / Progressive Disease

### Using Survival as Proxy for Treatment Response
Can create binary labels:
- **Good Response:** PFS > 12 months OR PRIMARY_THERAPY_OUTCOME = "Complete Response"
- **Poor Response:** PFS <= 12 months OR PRIMARY_THERAPY_OUTCOME = "Progressive Disease"

### Access Method
```bash
# GDC Data Portal API for slides
curl 'https://api.gdc.cancer.gov/files?filters={"op":"and","content":[{"op":"in","content":{"field":"cases.project.project_id","value":["TCGA-OV"]}},{"op":"=","content":{"field":"data_type","value":"Slide Image"}}]}&size=10&pretty=true'

# cBioPortal API for clinical data
curl 'https://www.cbioportal.org/api/studies/ov_tcga_pub/clinical-data?clinicalDataType=PATIENT'

# Download with gdc-client
gdc-client download -m manifest.txt
```

### Clinical Data Fields from cBioPortal
- DFS_MONTHS, DFS_STATUS
- OS_MONTHS, OS_STATUS
- PLATINUM_STATUS
- PRIMARY_THERAPY_OUTCOME_SUCCESS
- Sample metadata (tumor type, stage, grade)

### Assessment
- **Quality:** HIGH - TCGA gold standard
- **Relevance:** HIGH - Comprehensive clinical + genomic data
- **Ease of Download:** EASY - Direct API access
- **Recommendation:** Use for large-scale validation

---

## 4. CMB-OV (Cancer Moonshot Biobank)

### Overview
- **Source:** TCIA / dbGaP
- **DOI:** 10.7937/4nx6-e061
- **Size:** 131.49 GB
- **Subjects:** 47 patients
- **Slides:** 74 WSIs
- **Format:** SVS + JSON metadata
- **License:** CC BY 4.0 (images), Controlled (clinical via dbGaP)
- **Status:** Ongoing collection

### Labels Available
- Clinical data available through dbGaP (controlled access)
- Longitudinal imaging with days from enrollment
- Treatment information linked via NCI CRDC

### Access Method
```bash
# Imaging data (public):
# Download via Aspera Connect from TCIA

# Clinical/genomic data (controlled access):
# https://www.ncbi.nlm.nih.gov/gap/ - Study phs002192
```

### Assessment
- **Quality:** MEDIUM - Ongoing collection, smaller size
- **Relevance:** MEDIUM - Controlled access for clinical data
- **Ease of Download:** MEDIUM - Split access model
- **Recommendation:** Consider for future work when clinical data access obtained

---

## 5. CPTAC-OV (Clinical Proteomic Tumor Analysis Consortium)

### Overview
- **Source:** TCIA / GDC / NCI CPTAC
- **Subjects:** 102 patients
- **Slides:** Histopathology WSIs available
- **Cancer Type:** Ovarian Cancer
- **Additional Data:** Proteomics, Genomics
- **License:** Public (images), varies for genomic data

### Labels Available
- Integrated proteomic characterization
- Clinical outcomes linked to proteogenomic data

### Publication
Hu et al. (2020) "Integrated Proteomic and Glycoproteomic Characterization of Human High-Grade Serous Ovarian Carcinoma." Cell Reports. DOI: 10.1016/j.celrep.2020.108276

### Assessment
- **Quality:** MEDIUM - Focus is on proteomics
- **Relevance:** MEDIUM - Survival data available
- **Ease of Download:** MEDIUM - Multiple portals
- **Recommendation:** Supplementary dataset for multi-modal analysis

---

## Alternative Label Strategies

### Using Survival as Treatment Response Proxy

When direct treatment response labels are unavailable, survival metrics can serve as proxies:

#### Progression-Free Survival (PFS)
```
Good Response: PFS > 12 months
Poor Response: PFS <= 12 months
```

#### Overall Survival (OS)
```
Good Prognosis: OS > 36 months
Poor Prognosis: OS <= 36 months
```

#### Platinum Sensitivity (Standard Clinical Definition)
```
Platinum Sensitive: Recurrence > 6 months after completing platinum therapy
Platinum Resistant: Recurrence <= 6 months after completing platinum therapy
Platinum Refractory: Progression during platinum therapy
```

---

## Download Priority and Workflow

### Phase 1: Primary Datasets (Recommended for Hackathon)
1. **Ovarian Bevacizumab Response** - Direct treatment response labels
2. **PTRC-HGSOC** - Platinum sensitivity labels

### Phase 2: Validation Datasets
3. **TCGA-OV** - Large-scale validation with survival proxies

### Phase 3: Supplementary (If Time Permits)
4. **CMB-OV** - Additional samples
5. **CPTAC-OV** - Multi-modal analysis

---

## Tools for Data Access

### IBM Aspera Connect
Required for TCIA downloads:
- Download: https://www.ibm.com/products/aspera/downloads#Client-deployed+software
- Enables high-speed transfer of large imaging datasets

### GDC Data Transfer Tool
For TCGA/GDC data:
```bash
pip install gdc-client
gdc-client download -m manifest.txt
```

### cBioPortal API
For clinical data:
```python
import requests

# Get clinical data for TCGA-OV
url = "https://www.cbioportal.org/api/studies/ov_tcga_pub/clinical-data"
params = {"clinicalDataType": "PATIENT"}
response = requests.get(url, params=params)
clinical_data = response.json()
```

---

## Summary Recommendations

For the MedGemma Hackathon treatment response prediction task:

1. **Start with Ovarian Bevacizumab Response** - Best fit for the use case
   - 288 WSIs with direct treatment effectiveness labels
   - Well-balanced classes (56% effective, 44% invalid)
   - Published benchmarks available

2. **Add PTRC-HGSOC for generalization** - Platinum response
   - 348 WSIs with chemotherapy sensitivity
   - Different treatment modality for robustness testing

3. **Use TCGA-OV for large-scale validation**
   - 1,481 slides with survival-based proxy labels
   - Match with cBioPortal clinical data

Total available: ~2,100 WSIs with treatment response or survival labels

---

## References

1. Wang CW, et al. Sci Data. 2022;9:25. DOI: 10.1038/s41597-022-01127-6
2. Chowdhury S, et al. Cell. 2023;186(16):3476-3498.e35. DOI: 10.1016/j.cell.2023.07.004
3. Cancer Genome Atlas Research Network. Nature. 2011;474(7353):609-615.
4. Hu Y, et al. Cell Reports. 2020;33(3):108276. DOI: 10.1016/j.celrep.2020.108276
