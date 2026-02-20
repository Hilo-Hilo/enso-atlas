# Enso Atlas Frontend -- Integration Test Results

**Date:** 2026-02-20
**Build Status:** PASS
**ESLint:** PASS
**Model Scope Regression:** PASS (53/53)

---

## Build and Quality Checks

```
npm run build: PASS
npm run lint: PASS
npm run check:model-scope: PASS
npm run check:heatmap-alignment: PASS
```

Notes:
- Frontend build and lint succeeded after multi-project merges.
- Project-scoped model checks passed for ovarian and lung routing.
- Heatmap-grid alignment checks passed with model-scope safeguards in place.

---

## Regression Test Summary (53/53 PASS)

Post-merge regression suite validated project isolation and model scoping end-to-end.

Validated areas include:
- project-scoped model listing
- project-scoped analyze/heatmap/report routing
- model selection reset/rehydrate behavior on project switch
- fallback safety when project metadata refresh is required
- prevention of cross-project model leakage

---

## API Tests (17/17 PASS)

All backend API endpoints tested against running deployment on port 8003.

| Endpoint | Method | Status | Notes |
|----------|--------|--------|-------|
| /health | GET | PASS | Returns status, version, model state |
| /api/slides | GET | PASS | Returns slide metadata for active/project context |
| /api/slides/{id} | GET | PASS | Individual slide with embedding status |
| /api/dzi/{id} | GET | PASS | Deep Zoom Image tile serving |
| /api/analyze | POST | PASS | Project-scoped TransMIL analysis (6 total models across 2 projects) |
| /api/analyze (cached) | POST | PASS | Cached result retrieval |
| /api/report | POST | PASS | MedGemma 1.5 4B report generation |
| /api/semantic-search | POST | PASS | MedSigLIP text-to-patch search |
| /api/heatmap/{id} | GET | PASS | Attention heatmap with scoped model enforcement |
| /api/similar/{id} | GET | PASS | FAISS similar case retrieval |
| /api/projects | GET | PASS | Project listing with config |
| /api/projects | POST | PASS | Project creation |
| /api/projects/{id} | PUT | PASS | Project update |
| /api/projects/{id} | DELETE | PASS | Project deletion |
| /api/projects/{id}/slides | GET | PASS | Project-scoped slides (ovarian + lung) |
| /api/projects/{id}/models | GET | PASS | Project-scoped models |
| /api/analyze-batch | POST | PASS | Batch analysis with scoped model selection |

---

## Browser UI Tests (All PASS)

Tested in Chrome against frontend at port 3002.

| Feature | Status | Notes |
|---------|--------|-------|
| 3-panel resizable layout | PASS | Case Selection, WSI Viewer, Analysis Results |
| View mode toggle | PASS | Oncologist, Pathologist, Batch modes |
| Slide list with thumbnails | PASS | Ovarian and lung project slide sets load correctly |
| Project switcher | PASS | Switching projects updates model scope and dataset context |
| Project-scoped model picker | PASS | Shows only allowed models for active project |
| Run Analysis button | PASS | Triggers full project-scoped pipeline |
| TransMIL prediction display | PASS | Correct model outputs by project |
| Attention heatmap overlay | PASS | Heatmap generation blocked for out-of-scope models |
| Evidence patches panel | PASS | Top-K patches with normalized attention weights |
| Similar cases panel | PASS | Project-scoped FAISS retrieval |
| Semantic search (MedSigLIP) | PASS | Works where enabled by project features |
| AI report generation | PASS | MedGemma report with morphology descriptions |
| PDF export | PASS | Client-side jsPDF generation |
| JSON export | PASS | Blob download |
| Batch analysis | PASS | Multi-slide selection and parallel processing |
| Project Management UI | PASS | CRUD operations for projects |
| Slide Manager | PASS | Thumbnails, filtering, metadata display |
| Annotation tools | PASS | Circle, rectangle, freehand, measure, note |
| Dark mode | PASS | System-aware theming |
| Keyboard shortcuts | PASS | Full navigation and viewer controls |
| AI Assistant (agentic) | PASS | 7-step workflow execution |
| Error boundary | PASS | Graceful error handling |

---

## Test Environment

- **Backend:** Docker Compose on NVIDIA DGX Spark (ARM64, 128GB)
- **Backend port:** 8003
- **Frontend:** Next.js 14.2, built and served on port 3002
- **Browser:** Chrome (latest)
- **Database:** PostgreSQL with project/model/slide junction data
- **Models loaded:** Path Foundation (CPU), MedGemma 1.5 4B (GPU), MedSigLIP (GPU), 6x TransMIL classification models
- **Projects validated:** `ovarian-platinum`, `lung-stage`
