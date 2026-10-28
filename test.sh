#!/bin/bash
# ==============================================================================
# FedMedAI: Sequential Multi-Run Benchmark Runner for Remote GPU VM
# ==============================================================================
# Quick Configuration Options:
# ------------------------------------------------------------------------------
# 1. Edit the variables directly below in this file (lines 19-23)
# 2. Or pass as positional CLI arguments when invoking:
#      ./test.sh <strategy> <model> <dataset> [base_name]
#      e.g.: ./test.sh fedbns cnn organamnist
#      e.g.: ./test.sh fedprox baseline pathmnist
#      e.g.: ./test.sh fedavg baseline organamnist my_custom_name
# 3. Or pass via environment variables:
#      STRATEGY=fedbn MODEL=baseline ./test.sh
# ==============================================================================

set -eo pipefail

# ─── QUICK CHANGE PARAMETERS ──────────────────────────────────────────────────
# Default values can be edited here, or overridden via CLI arguments ($1, $2, $3, $4):
STRATEGY="${1:-${STRATEGY:-fedavg}}"         # fedavg | fedprox | fednova | fedbn | scaffold | fedbns
MODEL="${2:-${MODEL:-baseline}}"            # baseline | cnn | baseline_bn
DATASET="${3:-${DATASET:-organamnist}}"      # organamnist | pathmnist | bloodmnist | dermamnist
BASE_NAME="${4:-${BASE_NAME:-${DATASET}_${STRATEGY}_${MODEL}}}" # Output folder prefix (e.g. organamnist_fedavg_baseline)

# Training Hyperparameters (can also be overridden via environment variables):
ROUNDS="${ROUNDS:-5}"
PARTITION="${PARTITION:-practical}"
BATCH_SIZE="${BATCH_SIZE:-256}"
LR="${LR:-0.01}"
MOMENTUM="${MOMENTUM:-0.9}"
OPTIMIZER="${OPTIMIZER:-sgd}"
LR_MODE="${LR_MODE:-fixed}"
TIMEOUT="${TIMEOUT:-0}"

# List of distinct random seeds for the sequential runs (default: 5 runs)
SEEDS=(42 43 44 45 46)
TOTAL_RUNS=${#SEEDS[@]}
# ──────────────────────────────────────────────────────────────────────────────

# 1. Disable XON/XOFF flow control so accidental Ctrl+S does not freeze SSH terminal
stty -ixon 2>/dev/null || true

# 2. Force unbuffered standard I/O for Python in terminal and redirected pipes
export PYTHONUNBUFFERED=1

# 3. Inter-run resource reclamation and cooldown
cleanup_between_runs() {
    echo ""
    echo "=================================================================="
    echo "  Reclaiming GPU memory, closing lingering sockets & cooling down..."
    echo "=================================================================="
    pkill -f "experiments.run_simulation" 2>/dev/null || true
    pkill -f "client.client" 2>/dev/null || true
    pkill -f "server.server" 2>/dev/null || true
    # Cooldown to allow Linux kernel gRPC sockets (TIME_WAIT) and CUDA handles to reset
    sleep 5
}

# 4. Clean exit trap on interrupt, termination, or SSH SIGHUP disconnect
trap 'echo -e "\n[Interrupted] Killing background workers..."; cleanup_between_runs; exit 1' INT TERM HUP

# Print configuration banner
echo "=================================================================="
echo "  FedMedAI Automated Benchmark Suite Starting"
echo "=================================================================="
echo "  Strategy  : $STRATEGY"
echo "  Model     : $MODEL"
echo "  Dataset   : $DATASET"
echo "  Base Name : $BASE_NAME"
echo "  Rounds    : $ROUNDS | Batch Size: $BATCH_SIZE | LR: $LR ($OPTIMIZER)"
echo "  Partition : $PARTITION | Mode: $LR_MODE"
echo "  Total Runs: $TOTAL_RUNS (Seeds: ${SEEDS[*]})"
echo "=================================================================="

# Pre-run cleanup before starting suite
cleanup_between_runs

# Sequential execution loop
for i in "${!SEEDS[@]}"; do
    RUN_NUM=$((i + 1))
    SEED=${SEEDS[$i]}
    RUN_NAME="${BASE_NAME}_${RUN_NUM}"

    echo ""
    echo ">>> [$RUN_NUM/$TOTAL_RUNS] Launching Run $RUN_NUM of $TOTAL_RUNS (Seed: $SEED, Name: $RUN_NAME) <<<"

    python -u -m experiments.run_simulation \
        --rounds "$ROUNDS" \
        --lr_mode "$LR_MODE" \
        --strategies "$STRATEGY" \
        --partitions "$PARTITION" \
        --batch_size "$BATCH_SIZE" \
        --datasets "$DATASET" \
        --local_test \
        --model "$MODEL" \
        --optimizer "$OPTIMIZER" \
        --learning_rate "$LR" \
        --momentum "$MOMENTUM" \
        --seed "$SEED" \
        --timeout_per_run "$TIMEOUT" \
        --name "$RUN_NAME"

    cleanup_between_runs
done

echo ""
echo "=================================================================="
echo "  All $TOTAL_RUNS Sequential FL Experiments Successfully Completed! "
echo "=================================================================="
