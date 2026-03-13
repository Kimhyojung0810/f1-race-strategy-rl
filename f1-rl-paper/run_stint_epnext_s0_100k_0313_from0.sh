#!/usr/bin/env bash
# Split 0 100k (0313): Train 2016 / Val 2017 / Test 2018, ep_next ON

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Conda & CUDA env
export LD_LIBRARY_PATH="${HOME}/anaconda3/envs/vse38-gpu/lib/python3.8/site-packages/nvidia/cublas/lib:${HOME}/anaconda3/envs/vse38-gpu/lib/python3.8/site-packages/nvidia/cuda_runtime/lib:${HOME}/anaconda3/envs/vse38-gpu/lib/python3.8/site-packages/nvidia/cudnn/lib:${HOME}/anaconda3/envs/vse38-gpu/lib/python3.8/site-packages/nvidia/cusolver/lib:${HOME}/anaconda3/envs/vse38-gpu/lib/python3.8/site-packages/nvidia/cusparse/lib:${LD_LIBRARY_PATH:-}"
source "${HOME}/anaconda3/etc/profile.d/conda.sh"
conda activate vse38-gpu

export CUDA_VISIBLE_DEVICES=0

# Reward weights: Optuna best + stint (same as s1 run)
export L_LAP=0.7829297041629397
export L_POS=1.549193270465347
export L_PIT=0.7919197525086021
export L_STINT=0.5
export K_STINT=1.0

# ΔEP_next predictor (retrain)
export EP_NEXT_MODEL_PATH="$ROOT/machine_learning_rl_training/output/ep_next_predictor_retrain"
export USE_DELTA_EP=1

# Seasons: 2016 train / 2017 val / 2018 test
export TRAIN_SEASON=2016
export VAL_SEASON=2017
export TEST_SEASON=2018

# Training length: 100k steps
export NUM_ITERATIONS=100000

# Checkpoint every 15k steps
export CHECKPOINT_INTERVAL=15000

# Run tag / seed (episodes CSV & checkpoints will use this tag)
# 0313 v2 run: separate metrics/logs from earlier attempts
export RUN_TAG="stint_epnext_s0_100k_on_s20250313_v2"
export SEED=20250313

# Fresh run. To resume from latest checkpoint: set RESUME=1 and rerun.
export RESUME=0

cd "$ROOT"
exec python main_train_rl_agent_dqn.py

