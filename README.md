# Enso Atlas

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**On-Premise Pathology Evidence Engine for Treatment-Response Prediction**

Enso Atlas is an on-premise pathology evidence engine that analyzes whole-slide images (WSIs) using TransMIL attention-based classification, Path Foundation embeddings, and MedGemma clinical reporting to predict treatment response with interpretable, auditable evidence.

---

## Highlights

- **Local-first**: Runs entirely on-premise; no PHI leaves the hospital network
- **Evidence-based**: Attention heatmaps, evidence patches, semantic search, and structured reports provide auditable clinical evidence
- **Trained models**: 5 TransMIL classifiers trained on 208 TCGA ovarian cancer slides with best AUC of 0.907 (platinum sensitivity)
- **Foundation-model powered**: Path Foundation embeddings, MedGemma report generation, MedSigLIP semantic search
- **Production-ready**: FastAPI backend + Next.js frontend + PostgreSQL with Docker Compose deployment

---

## Screenshots

### Main Application View

![Main View](docs/screenshots/main-view.png)

*Enso Atlas main interface showing the case selection panel, WSI viewer, and analysis panels.*

### Analysis Results with Prediction Panel

![Analysis Results](docs/screenshots/analysis-results.png)

*Complete analysis showing prediction results with confidence scores, response probability, and evidence patches.*

### WSI Viewer with Heatmap Overlay

![WSI Viewer](docs/screenshots/wsi-viewer.png)

*OpenSeadragon-based whole-slide image viewer with TransMIL attention heatmap overlay, zoom controls, and minimap navigation.*

### Similar Cases Panel

![Similar Cases](docs/screenshots/similar-cases.png)

*FAISS-powered similar case retrieval showing morphologically similar cases from the reference cohort with similarity scores.*

### Prediction and Evidence Panels

![Prediction Panel](docs/screenshots/prediction-panel.png)

*Detailed prediction results with confidence score, response probability threshold, and evidence patches ranked by attention weight.*

---

## Quick Start

### Docker Deployment (Recommended)

```bash
# Clone the repository
git clone https://github.com/Hilo-Hilo/med-gemma-hackathon.git
cd med-gemma-hackathon

# Build and start backend + database
docker compose -f docker/docker-compose.yaml build
docker compose -f docker/docker-compose.yaml up -d

# Backend API available at http://localhost:8003 (~3.5 min startup for MedGemma loading)

# Build and start frontend
cd frontend
npm install
npm run build
npx next start -p 3002

# Frontend available at http://localhost:3002
```

### Local Development

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e .

# Start the API server (port 8000 locally, 8003 via Docker)
python -m uvicorn enso_atlas.api.main:app --reload --host 0.0.0.0 --port 8000

# In a separate terminal
cd frontend
npm install
npm run dev
# Frontend runs at http://localhost:3000 (dev) or http://localhost:3002 (production)
```

---

## Architecture

```
                           Enso Atlas Architecture

    +------------------+     +------------------+     +------------------+
    |   WSI Input      |     |   FastAPI        |     |   Next.js 14     |
    |   (.svs, .ndpi)  |---->|   Backend :8003  |<----|   Frontend :3002 |
    +------------------+     +------------------+     +------------------+
                                    |
           +------------+-----------+-----------+------------+
           |            |           |           |            |
           v            v           v           v            v
    +----------+  +-----------+  +-------+  +----------+  +--------+
    |   Path   |  | TransMIL  |  | FAISS |  | MedGemma |  |MedSig- |
    |Foundation|  | Classifier|  | Index |  | Reporter |  |  LIP   |
    |  (CPU)   |  |  (GPU)    |  |       |  |  (GPU)   |  | (GPU)  |
    +----------+  +-----------+  +-------+  +----------+  +--------+
                                    |
                             +-------------+
                             | PostgreSQL  |
                             |    :5433    |
                             +-------------+
```

### Core Components

| Component | Description |
|-----------|-------------|
| **WSI Processing** | OpenSlide-based processing with tissue detection |
| **Path Foundation** | 384-dim patch embeddings from Google's foundation model (CPU) |
| **TransMIL** | Transformer-based MIL for slide-level classification |
| **MedSigLIP** | Text-to-patch semantic search (GPU) |
| **FAISS Retrieval** | Similar case search from reference cohort |
| **MedGemma 1.5 4B** | Structured clinical report generation (GPU, ~20s/report) |
| **PostgreSQL** | Slide metadata, analysis results, and result caching |

### Model Performance

| Model | Task | AUC |
|-------|------|-----|
| platinum_sensitivity | Platinum treatment response | 0.907 |
| tumor_grade | Tumor grade classification | 0.752 |
| survival_5y | 5-year survival prediction | 0.697 |
| survival_3y | 3-year survival prediction | 0.645 |
| survival_1y | 1-year survival prediction | 0.639 |

Best single model AUC on full dataset: 0.879 with optimal threshold 0.917 (Youden's J index).

### Tech Stack

| Layer | Technology |
|-------|------------|
| WSI I/O | OpenSlide |
| Embeddings | Path Foundation (ViT-S, 384-dim) |
| Semantic Search | MedSigLIP (text-to-patch retrieval) |
| Classification | TransMIL (Transformer-based MIL) |
| Retrieval | FAISS |
| Reporting | MedGemma 1.5 4B |
| Backend | FastAPI + Python 3.10+ + asyncpg |
| Frontend | Next.js 14.2 + TypeScript + Tailwind CSS |
| Viewer | OpenSeadragon |
| Database | PostgreSQL |
| Deployment | Docker Compose on NVIDIA DGX Spark (ARM64) |

---

## API Reference

All endpoints are served at `http://localhost:8003` (Docker) or `http://localhost:8000` (local).

### Core Endpoints

```bash
# Health check
curl http://localhost:8003/api/health

# List slides (optionally filtered by project)
curl http://localhost:8003/api/slides?project_id=1

# Analyze a slide
curl -X POST http://localhost:8003/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"slide_id": "TCGA-XX-XXXX", "model_id": "platinum_sensitivity"}'

# Batch analysis
curl -X POST http://localhost:8003/api/analyze/batch \
  -H "Content-Type: application/json" \
  -d '{"slide_ids": ["slide_1", "slide_2"], "model_id": "platinum_sensitivity"}'

# Generate clinical report
curl -X POST http://localhost:8003/api/report \
  -H "Content-Type: application/json" \
  -d '{"slide_id": "TCGA-XX-XXXX"}'

# Semantic search via MedSigLIP
curl -X POST http://localhost:8003/api/semantic-search \
  -H "Content-Type: application/json" \
  -d '{"slide_id": "TCGA-XX-XXXX", "query": "tumor infiltrating lymphocytes", "top_k": 10}'
```

### Full Endpoint List

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | /api/health | Health check |
| GET | /api/slides | List slides (with ?project_id= filtering) |
| GET | /api/slides/{id}/dzi | DZI metadata for OpenSeadragon |
| GET | /api/slides/{id}/thumbnail | Slide thumbnail |
| GET | /api/slides/{id}/cached-results | Cached analysis results |
| POST | /api/analyze | Single slide analysis |
| POST | /api/analyze/batch | Multi-slide batch analysis |
| POST | /api/analyze/multi-model | Multi-model ensemble analysis |
| GET | /api/models | List models (with ?project_id= filtering) |
| GET/POST/PUT/DELETE | /api/projects | Project CRUD |
| GET/POST/DELETE | /api/projects/{id}/slides | Assign/unassign slides |
| GET/POST/DELETE | /api/projects/{id}/models | Assign/unassign models |
| POST | /api/semantic-search | MedSigLIP text-to-patch search |
| GET | /api/heatmap/{slide_id}/{model_id} | TransMIL attention heatmap |
| POST | /api/report | Generate MedGemma clinical report |
| GET | /api/slides/{id}/report/pdf | Export report as PDF |

### Interactive Documentation

- Swagger UI: [http://localhost:8003/api/docs](http://localhost:8003/api/docs)
- ReDoc: [http://localhost:8003/api/redoc](http://localhost:8003/api/redoc)

---

## Project Structure

```
med-gemma-hackathon/
|-- src/enso_atlas/
|   |-- api/           # FastAPI endpoints
|   |-- embedding/     # Path Foundation embedder
|   |-- evidence/      # Heatmaps and FAISS retrieval
|   |-- mil/           # TransMIL attention classifier
|   |-- reporting/     # MedGemma report generation
|   |-- wsi/           # WSI processing
|-- frontend/          # Next.js 14.2 application
|-- docker/            # Docker Compose configuration
|-- config/            # projects.yaml and configuration
|-- data/
|   |-- tcga_full/slides/    # TCGA ovarian cancer WSIs
|   |-- embeddings/level0/   # Path Foundation embeddings
|-- models/            # Trained TransMIL weights
|-- tests/             # Unit tests
|-- docs/              # Documentation and screenshots
```

---

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `CUDA_VISIBLE_DEVICES` | GPU selection | All GPUs |
| `NEXT_PUBLIC_API_URL` | Frontend API URL | http://localhost:8003 |

### Project Configuration

Projects are managed via `config/projects.yaml` and the `/api/projects` CRUD endpoints. Each project scopes slides and models via PostgreSQL junction tables (project_slides, project_models), enabling multi-cancer support from a single deployment.

---

## Dataset

The system is trained and evaluated on **208 TCGA ovarian cancer whole-slide images** with level 0 (full resolution) Path Foundation embeddings. Classification targets include platinum sensitivity, tumor grade, and survival at 1, 3, and 5 year horizons.

---

## Docker Deployment

Services are defined in `docker/docker-compose.yaml`:

| Service | Description | Port |
|---------|-------------|------|
| enso-atlas | FastAPI backend + ML models | 8003 (host) -> 8000 (container) |
| atlas-db | PostgreSQL database | 5433 |

The backend takes approximately 3.5 minutes to fully start due to MedGemma model loading. The frontend runs separately outside Docker.

See [docs/reproduce.md](docs/reproduce.md) for detailed deployment instructions.

---

## Development

### Running Tests

```bash
pytest tests/
pytest --cov=src tests/
```

### Code Quality

```bash
ruff check src/
black src/ --check
mypy src/

cd frontend && npm run lint
```

---

## Acknowledgments

- **Google Health AI** for Path Foundation, MedGemma, and MedSigLIP
- **NVIDIA** for DGX Spark compute resources
- **TCGA** for the ovarian cancer whole-slide image dataset
- [TransMIL](https://github.com/szc19990412/TransMIL) for the Transformer-based MIL architecture

---

## License

MIT License - See [LICENSE](LICENSE) for details.

---

## References

1. Shao et al., "TransMIL: Transformer based Correlated Multiple Instance Learning for Whole Slide Image Classification," *NeurIPS*, 2021.
2. Google Health AI, [Path Foundation](https://developers.google.com/health-ai-developer-foundations/path-foundation)
3. Google, [MedGemma](https://developers.google.com/health-ai-developer-foundations/medgemma)
4. Google Health AI, [MedSigLIP](https://developers.google.com/health-ai-developer-foundations/medsiglip)
