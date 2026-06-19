# Serving InstaNovo Models

Koina can serve three InstaNovo endpoints through the Triton Python backend:

| Model | Description |
|---|---|
| `InstaNovo` | Runs the InstaNovo transformer model for _de novo_ peptide sequencing. |
| `InstaNovoPlus` | Runs the InstaNovo+ diffusion model directly on spectra. |
| `InstaNovoWithRefinement` | Runs InstaNovo first, then refines the sequence with InstaNovo+. Recommended for most MGF workflows. |

These models require an NVIDIA GPU. The model configs declare `KIND_GPU`, and startup fails if Docker cannot see a CUDA-capable GPU.

## Requirements

- Linux host with an NVIDIA GPU.
- NVIDIA driver recent enough for CUDA 12.x containers.
- Docker with BuildKit enabled.
- NVIDIA Container Toolkit, verified with:

```bash
docker run --rm --gpus all ubuntu:latest nvidia-smi
```

The InstaNovo serving image is based on `nvcr.io/nvidia/tritonserver:24.10-py3`, installs
`torch==2.8.0+cu126`, and prefetches the default checkpoints into the image:

- `instanovo-v1.2.0.ckpt`
- `instanovoplus-v1.1.0.ckpt`

## Build the Image

From the Koina repository root, build against the public InstaNovo repository:

```bash
DOCKER_BUILDKIT=1 docker build \
  --target serving-develop \
  -t koina-instanovo:latest \
  .
```

When developing InstaNovo locally, build from a local checkout instead:

```bash
DOCKER_BUILDKIT=1 docker build \
  --target serving-develop-local-instanovo \
  --build-context instanovo=/path/to/InstaNovo \
  -t koina-instanovo:latest \
  .
```

For example, if the InstaNovo checkout is next to Koina:

```bash
DOCKER_BUILDKIT=1 docker build \
  --target serving-develop-local-instanovo \
  --build-context instanovo=../InstaNovo \
  -t koina-instanovo:latest \
  .
```

## Start a Local Triton Server

Run only the InstaNovo models:

```bash
docker run -d --rm \
  --name koina-instanovo \
  --gpus all \
  --shm-size 8G \
  -e MODEL_PATTERN='InstaNovo*' \
  -e INSTANOVO_DEVICE='cuda' \
  -p 8500:8500 \
  -p 8501:8501 \
  -p 8502:8502 \
  -v "$PWD/models:/models" \
  koina-instanovo:latest
```

Ports:

| Port | Service |
|---|---|
| `8500` | Triton gRPC inference API |
| `8501` | Triton HTTP inference API |
| `8502` | Triton metrics |

Wait until Triton is ready:

```bash
curl -fsS http://localhost:8501/v2/health/ready && echo "ready"
```

Inspect model metadata:

```bash
curl -sS http://localhost:8501/v2/models/InstaNovoWithRefinement | python3 -m json.tool
```

Follow startup logs:

```bash
docker logs -f koina-instanovo
```

Successful startup shows all three models as `READY`.

## Runtime Configuration

The serving wrapper reads these optional environment variables:

| Variable | Default | Description |
|---|---|---|
| `MODEL_PATTERN` | all models | Set to `InstaNovo*` to load only InstaNovo models. |
| `INSTANOVO_DEVICE` | auto | Use `cuda` for GPU serving. |
| `INSTANOVO_NUM_BEAMS` | `5` | Beam count used by the transformer and diffusion decoders. |
| `INSTANOVO_MAX_LENGTH` | `40` | Maximum decoded peptide length. |
| `INSTANOVO_FORCE_FP32` | `0` | Set to `1`, `true`, or `yes` to force selected model components to FP32. |
| `INSTANOVO_MODEL_ID` | `instanovo-v1.2.0` | Model ID or checkpoint path for the transformer model. |
| `INSTANOVOPLUS_MODEL_ID` | `instanovoplus-v1.1.0` | Model ID or checkpoint path for the diffusion model. |

Example with different decoding settings:

```bash
docker run -d --rm \
  --name koina-instanovo \
  --gpus all \
  --shm-size 8G \
  -e MODEL_PATTERN='InstaNovo*' \
  -e INSTANOVO_DEVICE='cuda' \
  -e INSTANOVO_NUM_BEAMS=3 \
  -e INSTANOVO_MAX_LENGTH=50 \
  -p 8500:8500 \
  -p 8501:8501 \
  -p 8502:8502 \
  -v "$PWD/models:/models" \
  koina-instanovo:latest
```

## Tensor Contract

All three endpoints use the same input tensors.

| Input | Type | Shape | Description |
|---|---|---|---|
| `mz_array` | `FP32` | `[batch, peaks]` | Padded m/z peak array. |
| `intensity_array` | `FP32` | `[batch, peaks]` | Padded intensity peak array. |
| `precursor_mz` | `FP32` | `[batch, 1]` | Precursor m/z. |
| `precursor_charge` | `INT32` | `[batch, 1]` | Precursor charge. |
| `num_peaks` | `INT32` | `[batch, 1]` | Number of real peaks before padding. |

Main outputs:

| Output | Type | Shape | Description |
|---|---|---|---|
| `predictions` | `BYTES` | `[batch, 1]` | Best peptide prediction. |
| `predictions_tokenised` | `BYTES` | `[batch, 1]` | Tokenised prediction, including modification tokens. |
| `log_probs` | `FP32` | `[batch, 1]` | Prediction log probability. |
| `confidence` | `FP32` | `[batch, 1]` | `exp(log_probs)`. |
| `delta_mass_ppm` | `FP32` | `[batch, 1]` | Precursor mass error in ppm. |
| `beam_*` | mixed | `[batch, beams]` | Beam predictions and scores. |

`InstaNovoWithRefinement` also returns:

| Output | Type | Shape | Description |
|---|---|---|---|
| `instanovo_predictions` | `BYTES` | `[batch, 1]` | Initial transformer prediction before refinement. |
| `instanovo_log_probs` | `FP32` | `[batch, 1]` | Initial transformer log probability. |
| `instanovo_confidence` | `FP32` | `[batch, 1]` | Initial transformer confidence. |

## Minimal HTTP Inference Example

This request asks for only the `confidence` output from `InstaNovo`:

```bash
curl -sS -X POST http://localhost:8501/v2/models/InstaNovo/infer \
  -H 'Content-Type: application/json' \
  -d '{
    "inputs": [
      {
        "name": "mz_array",
        "shape": [2, 5],
        "datatype": "FP32",
        "data": [110.0, 180.0, 250.0, 400.0, 0.0, 120.0, 190.0, 275.0, 410.0, 600.0]
      },
      {
        "name": "intensity_array",
        "shape": [2, 5],
        "datatype": "FP32",
        "data": [0.1, 0.9, 0.4, 0.2, 0.0, 0.2, 0.8, 0.5, 0.3, 0.1]
      },
      {
        "name": "precursor_mz",
        "shape": [2, 1],
        "datatype": "FP32",
        "data": [400.0, 500.0]
      },
      {
        "name": "precursor_charge",
        "shape": [2, 1],
        "datatype": "INT32",
        "data": [2, 2]
      },
      {
        "name": "num_peaks",
        "shape": [2, 1],
        "datatype": "INT32",
        "data": [4, 5]
      }
    ],
    "outputs": [
      {
        "name": "confidence",
        "parameters": {
          "binary_data": false
        }
      }
    ]
  }' | python3 -m json.tool
```

## End-to-end MGF Prediction

The Python example `clients/python/examples/instanovo_mgf.py` reads an MGF file, pads each batch of
spectra, calls Triton over gRPC, and writes a CSV file.

Run it with `uv` so the client dependencies are installed for the command:

```bash
uv run \
  --with "tritonclient[grpc]" \
  --with pyteomics \
  --with pandas \
  --with numpy \
  python clients/python/examples/instanovo_mgf.py input.mgf \
    --model InstaNovoWithRefinement \
    --server localhost:8500 \
    --output predictions.csv
```

Useful options:

| Option | Description |
|---|---|
| `--model` | One of `InstaNovo`, `InstaNovoPlus`, or `InstaNovoWithRefinement`. |
| `--server` | Triton gRPC endpoint, for example `localhost:8500`. |
| `--batch-size` | Number of spectra per request. Default is `16`. |
| `--ssl` | Use TLS for the gRPC connection. |
| `--output` | Output CSV path. |

The output CSV includes the original spectrum index, precursor metadata, the best prediction, beam
predictions, confidence scores, mass errors, and, for `InstaNovoWithRefinement`, the initial
InstaNovo prediction before refinement.

## Troubleshooting

### `KIND_GPU but no GPUs are available`

Docker cannot see the GPU. Check:

```bash
nvidia-smi
docker run --rm --gpus all ubuntu:latest nvidia-smi
```

Then restart the container with `--gpus all` or the correct GPU device selection.

### Model startup downloads checkpoints

The Dockerfile prefetches the default checkpoints at build time. If custom `INSTANOVO_MODEL_ID` or
`INSTANOVOPLUS_MODEL_ID` values point to other remote model IDs, those checkpoints may still be
downloaded when the model initializes.

### HTTP health endpoint prints no body

`/v2/health/ready` returns a successful empty response when the server is ready. Use `curl -fsS ... && echo ready` when scripting readiness checks.

### NumPy version

The image pins runtime NumPy below 2 because Triton 24.10 Python backend returned zero-byte output
buffers with NumPy 2.x in local testing. Do not remove that pin unless the Triton Python backend
compatibility has been revalidated.
