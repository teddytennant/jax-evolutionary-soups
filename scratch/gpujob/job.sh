#!/usr/bin/env bash
# Characterize test_evolution_improved_on_its_starting_population on an H200.
set -uo pipefail
D="$SLURM_SUBMIT_DIR"
cd "$D"
PY="$HOME/arxiv-jax/venv/bin/python"
export PYTHONPATH="$D:${PYTHONPATH:-}"
export JAX_ENABLE_X64=1

{
  echo "host $(hostname)"
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
  "$PY" -c "import jax; print('jax', jax.__version__, jax.devices(), jax.default_backend())"
} > "$D/env.txt" 2>&1

# 1. reproduce the reported failure
timeout 420 "$PY" -m pytest tests/test_end_to_end.py -p no:cacheprovider -q \
  > "$D/pytest_end_to_end.txt" 2>&1
echo "pytest rc=$?" >> "$D/pytest_end_to_end.txt"

# 2. same stage-1 experts as the CPU run, so the only difference is the backend
timeout 300 "$PY" scratch/paired.py --blob scratch/setup_cpu.pkl \
  --out "$D/paired_gpu.json" --seeds 10 --modes fixed \
  > "$D/paired_gpu.log" 2>&1
echo "paired rc=$?" >> "$D/paired_gpu.log"

# 3. the distribution, stage 1 built on this backend
timeout 600 "$PY" scratch/characterize.py --seeds 25 --start 1 --setup-seeds 0 \
  --modes fixed,per_layer,single --out "$D/gpu_setup0.json" \
  > "$D/gpu_setup0.log" 2>&1
echo "characterize rc=$?" >> "$D/gpu_setup0.log"
