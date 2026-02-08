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

## Verify GPU Access

```bash
docker exec -it enso-atlas python -c "import torch; print('cuda:', torch.cuda.is_available()); print('gpus:', torch.cuda.device_count())"
```

## Run the Frontend

The frontend runs separately outside Docker:

```bash
cd frontend
npm install
npm run build
npx next start -p 3002
```

## Open the UI

- Frontend: `http://<DGX_HOSTNAME_OR_IP>:3002`
- Backend API docs: `http://<DGX_HOSTNAME_OR_IP>:8003/api/docs`

## Data Layout

```
data/
  tcga_full/slides/          # 208 TCGA ovarian cancer WSIs (.svs)
  embeddings/level0/         # Path Foundation 384-dim patch embeddings
config/
  projects.yaml              # Project definitions (writable for CRUD)
models/                      # Trained TransMIL weights (5 models)
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
