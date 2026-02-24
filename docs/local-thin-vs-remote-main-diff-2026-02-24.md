# Local thin copy vs remote main diff

Generated: 2026-02-24T15:03:06.418642

## Summary

- Changed files (same path, different content): **23**
- Files only in local thin copy: **40**
- Files only in remote/main repo: **22**

## Files merged into branch `ui/local-thin-merge-20260224`

- `frontend/src/app/api/heatmap/[slideId]/[modelId]/route.ts`
- `frontend/src/app/globals.css`
- `frontend/src/app/page.tsx`
- `frontend/src/app/projects/page.tsx`
- `frontend/src/components/layout/DisclaimerBanner.tsx`
- `frontend/src/components/layout/Footer.tsx`
- `frontend/src/components/layout/Header.tsx`
- `frontend/src/components/layout/UserDropdown.tsx`
- `frontend/src/components/panels/AnalysisControls.tsx`
- `frontend/src/components/panels/BatchAnalysisPanel.tsx`
- `frontend/src/components/panels/ModelPicker.tsx`
- `frontend/src/components/panels/MultiModelPredictionPanel.tsx`
- `frontend/src/components/panels/OncologistSummaryView.tsx`
- `frontend/src/components/panels/OutlierDetectorPanel.tsx`
- `frontend/src/components/panels/PredictionPanel.tsx`
- `frontend/src/components/panels/ReportPanel.tsx`
- `frontend/src/components/panels/SemanticSearchPanel.tsx`
- `frontend/src/components/panels/SlideSelector.tsx`
- `frontend/src/components/ui/Logo.tsx`
- `frontend/src/components/ui/PredictionGauge.tsx`
- `frontend/src/components/viewer/WSIViewer.tsx`
- `frontend/src/contexts/ProjectContext.tsx`
- `frontend/src/lib/api.ts`
- `frontend/src/lib/mock-data.ts`
- `frontend/src/lib/pdfExport.ts`
- `frontend/src/types/index.ts`
- `frontend/tailwind.config.ts`

## Changed files (all)

- `README.md`
- `config/projects.yaml`
- `docker/Dockerfile`
- `docker/docker-compose.yaml`
- `frontend/.env.local`
- `frontend/README.md`
- `frontend/tsconfig.tsbuildinfo`
- `pyproject.toml`
- `requirements.txt`
- `scripts/multi_model_inference.py`
- `scripts/start.sh`
- `scripts/validate_project_modularity.py`
- `src/enso_atlas/agent/workflow.py`
- `src/enso_atlas/api/main.py`
- `src/enso_atlas/api/model_scope.py`
- `src/enso_atlas/api/pdf_export.py`
- `src/enso_atlas/api/project_routes.py`
- `src/enso_atlas/api/report_tasks.py`
- `src/enso_atlas/core.py`
- `tests/test_api_comprehensive.py`
- `tests/test_backend_project_scoping_regressions.py`
- `tests/test_frontend_heatmap_proxy_scoping.py`
- `tests/test_heatmap_grid_alignment.py`

## Local-only files

- `.env.local`
- `DEBUGGING_GUIDE.md`
- `IMPLEMENTATION_GUIDE.md`
- `TECHNICAL_SPECIFICATION.md`
- `docker/.env.example`
- `docs/DATASETS.md`
- `docs/KAGGLE_SUBMISSION.md`
- `docs/SUBMISSION_CHECKLIST.md`
- `docs/SUBMISSION_WRITEUP.md`
- `docs/SUBMISSION_WRITEUP.pdf`
- `docs/VIDEO_SCRIPT.md`
- `docs/VIDEO_SCRIPT.pdf`
- `docs/reproduce.md`
- `docs/screenshots/01-main-dashboard.png`
- `docs/screenshots/02-oncologist-view.png`
- `docs/screenshots/03-pathologist-view.png`
- `docs/screenshots/04-batch-view.png`
- `docs/screenshots/05-slide-manager.png`
- `docs/screenshots/06-projects.png`
- `docs/screenshots/analysis-results.png`
- `docs/screenshots/batch-analysis-lung.png`
- `docs/screenshots/dashboard-lung-oncologist.png`
- `docs/screenshots/dashboard-lung-pathologist.png`
- `docs/screenshots/main-view.png`
- `docs/screenshots/prediction-panel.png`
- `docs/screenshots/project-management.png`
- `docs/screenshots/similar-cases.png`
- `docs/screenshots/slide-manager.png`
- `docs/screenshots/wsi-viewer.png`
- `docs/technical-writeup-draft.md`
- `docs/technical_writeup.md`
- `embedder_32-path.py`
- `models/medgemma-4b-it/.cache/huggingface/.gitignore`
- `models/medgemma-4b-it/.cache/huggingface/download/model-00001-of-00002.safetensors.metadata`
- `models/medgemma-4b-it/.cache/huggingface/download/model-00002-of-00002.safetensors.metadata`
- `models/transmil_best.pt`
- `package-lock.json`
- `scripts/create_gdc_thumbnails.py`
- `submission/WRITEUP.md`
- `submission/enso-atlas-kaggle-notebook.ipynb`

## Remote-only files

- `.github/workflows/ci.yml`
- `docs.md`
- `frontend/src/app/api/slides/[slideId]/dzi/route.ts`
- `frontend/src/app/api/slides/[slideId]/dzi_files/[level]/[tileSpec]/route.ts`
- `frontend/src/app/api/slides/[slideId]/patches/[patchId]/route.ts`
- `frontend/src/app/api/slides/[slideId]/thumbnail/route.ts`
- `frontend/src/app/slides/page.tsx`
- `frontend/src/components/slides/BulkActions.tsx`
- `frontend/src/components/slides/FilterPanel.tsx`
- `frontend/src/components/slides/SlideGrid.tsx`
- `frontend/src/components/slides/SlideModals.tsx`
- `frontend/src/components/slides/SlideTable.tsx`
- `frontend/src/components/slides/index.ts`
- `frontend/src/lib/stats-store.ts`
- `outputs/training/training_20260131_230722.log`
- `outputs/training/training_20260131_230734.log`
- `outputs/training/training_20260131_230753.log`
- `tests/test_issue80_project_scoping_and_wsi_resolution.py`
- `tests/test_issue_multimodel_threshold_yaml.py`
- `tests/test_semantic_search_frontend_patch_preview_regression.py`
- `tests/test_semantic_search_patch_coordinate_helpers.py`
- `tests/test_validate_project_modularity_guardrails.py`