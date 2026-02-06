# Enso Atlas - 3-Minute Video Demo Script

## Structure (3:00 total)

### Opening (0:00 - 0:20)
- Title card: "Enso Atlas: On-Premise Pathology Evidence Engine"
- Problem statement: "Treatment response prediction for ovarian cancer patients requires analyzing whole-slide images -- a process that takes hours per case"
- "Enso Atlas brings foundation model-powered analysis directly to hospital pathology labs"

### Architecture Overview (0:20 - 0:40)
- Quick diagram: WSI -> Path Foundation -> TransMIL -> Prediction + Evidence
- Highlight: runs entirely on-premise (DGX Spark), no PHI leaves the network
- Three HAI-DEF models: Path Foundation (embeddings), MedGemma (reports), MedSigLIP (search)

### Demo: Single Case Analysis (0:40 - 1:30)
- Open Enso Atlas in browser
- Select TCGA ovarian case from case list (208 available)
- Click "Run Analysis"
- Show progress: embedding -> prediction -> evidence -> report
- Result: RESPONDER 98% with attention heatmap overlay
- Zoom into WSI with OpenSeadragon viewer
- Show evidence patches with attention weights
- Show MedGemma-generated clinical report with morphology descriptions

### Demo: Clinical Features (1:30 - 2:20)
- Similar Cases: FAISS retrieval shows 5 most similar cases from training set
- Semantic Search: Type "tumor infiltrating lymphocytes" -> MedSigLIP finds matching regions
- AI Assistant: Multi-step agentic workflow (analyze + retrieve + compare + report)
- PDF Export: Download clinical report as PDF
- Batch Mode: Select multiple slides for parallel processing

### Fine-Tuning and Technical Depth (2:20 - 2:45)
- Show TransMIL training results: 5-fold CV, AUC curves
- Attention mechanism provides interpretable evidence
- Path Foundation embeddings: 384-dim features from 224x224 patches
- Level 0 (full resolution) patches for cellular-level detail

### Closing (2:45 - 3:00)
- Impact: "Every pathology lab with a workstation can run Enso Atlas"
- Open source, reproducible, clinically grounded
- Safety disclaimers: research tool, requires pathologist validation
- "Enso Atlas - Evidence-based pathology, on your hardware"

## Recording Notes
- Use Chrome, frontend at http://100.111.126.23:3002
- Pre-load a case to avoid cold-start wait
- Screen recording: 1920x1080, 30fps
- Voiceover or captions (captions preferred for accessibility)
