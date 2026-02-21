# Reproduce on NVIDIA DGX Spark (Docker)

## Prerequisites

- NVIDIA DGX Spark (ARM64, 128GB unified memory) or equivalent GPU server
- Docker with NVIDIA Container Toolkit
- Node.js 18+ (for frontend)

```bash
# Verify GPU and Docker setup
nvidia-smi
docker --version
docker compose version

# Verify NVIDIA Container Toolkit
docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi
```

## Build

From the repository root:

```bash
docker compose -f docker/docker-compose.yaml build
```

## Download Models (Required for Offline Mode)

The default Compose configuration runs in offline mode (`TRANSFORMERS_OFFLINE=1`, `HF_HUB_OFFLINE=1`). Populate the model volume once, then run offline.

```bash
# Set Hugging Face token for gated model access
export HF_TOKEN="YOUR_HUGGINGFACE_TOKEN"

docker compose -f docker/docker-compose.yaml run --rm \
  -e TRANSFORMERS_OFFLINE=0 \
  -e HF_HUB_OFFLINE=0 \
  -e HF_TOKEN="$HF_TOKEN" \
  enso-atlas python /app/scripts/download_models.py --cache-dir /app/models --models all
```

## Run the Backend

```bash
docker compose -f docker/docker-compose.yaml up -d
```

This starts two services:

| Service | Description | Port |
|---------|-------------|------|
| enso-atlas | FastAPI backend + ML models | 8003 (host) -> 8000 (container) |
| atlas-db | PostgreSQL database | 5433 |

The backend takes approximately 3.5 minutes to fully start due to MedGemma model loading.

Check status and logs:

```bash
docker compose -f docker/docker-compose.yaml ps
docker logs -f enso-atlas
```

## Run the Frontend

The frontend runs separately outside Docker:

```bash
cd frontend
npm install

# Keep same-origin API routing for public hosting (Cloudflare/Tailscale)
# so browser requests use /api/... through the frontend origin.
cat > .env.local <<'EOF'
NEXT_PUBLIC_API_URL=
EOF

npm run build
npx next start -p 3002
```

## Open the UI

- Frontend: `http://<DGX_HOSTNAME_OR_IP>:3002`
- Backend API docs: `http://<DGX_HOSTNAME_OR_IP>:8003/api/docs`

## Project Configuration (Central Source of Truth)

All project definitions are managed in:

- `config/projects.yaml`

This file defines, per project:

- cancer type
- prediction target
- class labels
- enabled models
- dataset paths
- feature toggles
- thresholds

Current baseline config includes two projects and six TransMIL models total:

- `ovarian-platinum`
- `lung-stage`

## Data Layout (Project-Scoped)

```text
data/
  projects/
    ovarian-platinum/
      slides/
      embeddings/
      labels.csv
    lung-stage/
      slides/
      embeddings/
      labels.json
```

Do not use legacy flat layouts like `data/tcga_full/` for new runs.

### Embedding Integrity Guardrails (Prevent Missing Level-0)

After embedding refreshes, project switches, or file migrations, validate that each project's
`embeddings/level0/` directory matches top-level `embeddings/*.npy` files.

```bash
python scripts/validate_project_modularity.py --check-embedding-layout
```

If this check fails, strict level-0 endpoints (`/api/analyze-multi`, heatmap endpoints)
can report missing level-0 embeddings even when flat embeddings exist.

**Recovery pattern (safe):** backup stale `level0/`, then re-sync from top-level embeddings.

```bash
# Example for a project id (run where data is mounted writable)
PROJECT_ID=ovarian-platinum
EMB_DIR="data/projects/${PROJECT_ID}/embeddings"
mv "${EMB_DIR}/level0" "${EMB_DIR}/level0.pre-sync-$(date +%Y%m%d-%H%M%S)" 2>/dev/null || true
mkdir -p "${EMB_DIR}/level0"
find "${EMB_DIR}" -maxdepth 1 -type f -name '*.npy' -exec cp -a {} "${EMB_DIR}/level0/" \;
```

## Project Setup Examples

### A) Ovarian project (`ovarian-platinum`)

```bash
mkdir -p data/projects/ovarian-platinum/{slides,embeddings}
# Place ovarian .svs files in data/projects/ovarian-platinum/slides/
# Ensure labels file exists at data/projects/ovarian-platinum/labels.csv
```

Generate embeddings:

```bash
python scripts/embed_level0_pipelined.py \
  --slides_dir data/projects/ovarian-platinum/slides \
  --output_dir data/projects/ovarian-platinum/embeddings \
  --batch_size 512
```

### B) Lung stage project (`lung-stage`)

```bash
mkdir -p data/projects/lung-stage/{slides,embeddings}
# Place LUAD .svs files in data/projects/lung-stage/slides/
# Ensure labels file exists at data/projects/lung-stage/labels.json
```

Generate embeddings:

```bash
python scripts/embed_level0_pipelined.py \
  --slides_dir data/projects/lung-stage/slides \
  --output_dir data/projects/lung-stage/embeddings \
  --batch_size 512
```

After updating slides/labels, repopulate metadata:

```bash
curl -X POST http://localhost:8003/api/db/repopulate
```

## Verify GPU Access

```bash
docker exec -it enso-atlas python -c "import torch; print('cuda:', torch.cuda.is_available()); print('gpus:', torch.cuda.device_count())"
```

## Stop

```bash
docker compose -f docker/docker-compose.yaml down
```

## Troubleshooting

### GPU not visible in container

1. Check nvidia-container-toolkit is installed:
   ```bash
   docker info | grep -i nvidia
   ```

2. Test with a minimal CUDA container:
   ```bash
   docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi
   ```

### Model download fails

- Ensure HF_TOKEN is set and the account has accepted model terms for MedGemma and Path Foundation
- Check network connectivity
- Try downloading outside Docker first to verify the token works

### Backend startup is slow

The backend takes ~3.5 minutes to start because MedGemma 1.5 4B must be loaded into GPU memory. Monitor with `docker logs -f enso-atlas` and wait for the "Application startup complete" message.

### Path Foundation running on CPU

Path Foundation uses TensorFlow, which has a known incompatibility with Blackwell GPUs. It runs on CPU by design. This affects embedding generation speed but not inference (embeddings are precomputed and cached).

### Build fails on ARM64

The Docker images are built for ARM64 (DGX Spark). If building on x86_64, verify that base images support your architecture or use `--platform linux/arm64` with emulation.
