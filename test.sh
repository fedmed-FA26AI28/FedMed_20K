#!/bin/bash
# ==============================================================================
# FedMedAI: Sequential Multi-Run Benchmark Runner for Remote GPU VM
# ==============================================================================
# Quick Configuration Options:
# ------------------------------------------------------------------------------
# 1. Edit the variables directly below in this file (lines 20-38)
# 2. Or pass as positional CLI arguments when invoking:
#      ./test.sh "<strategy1> [strategy2]..." <model> "<dataset1> [dataset2]..." [custom_base_name]
#      e.g.: ./test.sh fedavg baseline organamnist
#      e.g.: ./test.sh "fedavg fedprox fedbn" baseline organamnist
#      e.g.: ./test.sh "fedavg fedprox" baseline "organamnist pathmnist"
#      e.g.: ./test.sh fedbns cnn organamnist,pathmnist
# 3. Or pass via environment variables:
#      STRATEGIES="fedavg fedprox" DATASETS="organamnist pathmnist" MODEL=baseline ./test.sh
# ==============================================================================

set -eo pipefail

# ─── QUICK CHANGE PARAMETERS ──────────────────────────────────────────────────
# Default values can be edited here, or overridden via CLI arguments ($1, $2, $3, $4):

# Strategies: edit DEFAULT_STRATEGIES below or pass via CLI $1 / env STRATEGIES / env STRATEGY.
# Supports space-separated ("fedavg fedprox") or comma-separated ("fedavg,fedprox"):
DEFAULT_STRATEGIES=(fedavg)

RAW_STRATEGIES="${1:-${STRATEGIES:-${STRATEGY:-}}}"
if [ -n "$RAW_STRATEGIES" ]; then
    CLEAN_STRATEGIES="${RAW_STRATEGIES//,/ }"
    read -r -a STRATEGIES <<< "$CLEAN_STRATEGIES"
else
    STRATEGIES=("${DEFAULT_STRATEGIES[@]}")
fi

TOTAL_STRATEGIES=${#STRATEGIES[@]}

MODEL="${2:-${MODEL:-baseline}}"            # baseline | cnn | baseline_bn

# Datasets: edit DEFAULT_DATASETS below or pass via CLI $3 / env DATASETS / env DATASET.
# Supports space-separated ("organamnist pathmnist") or comma-separated ("organamnist,pathmnist"):
DEFAULT_DATASETS=(organamnist)

RAW_DATASETS="${3:-${DATASETS:-${DATASET:-}}}"
if [ -n "$RAW_DATASETS" ]; then
    CLEAN_DATASETS="${RAW_DATASETS//,/ }"
    read -r -a DATASETS <<< "$CLEAN_DATASETS"
else
    DATASETS=("${DEFAULT_DATASETS[@]}")
fi

TOTAL_DATASETS=${#DATASETS[@]}

# Optional custom base name prefix (CLI $4 or env BASE_NAME).
# If left empty, defaults automatically to "${CURRENT_DATASET}_${CURRENT_STRATEGY}_${MODEL}" for each combination.
CUSTOM_BASE_NAME="${4:-${BASE_NAME:-}}"

# Training Hyperparameters (can also be overridden via environment variables):
ROUNDS="${ROUNDS:-80}"
PARTITION="${PARTITION:-practical}"
BATCH_SIZE="${BATCH_SIZE:-256}"
LR="${LR:-0.01}"
MOMENTUM="${MOMENTUM:-0.9}"
OPTIMIZER="${OPTIMIZER:-sgd}"
LR_MODE="${LR_MODE:-fixed}"
TIMEOUT="${TIMEOUT:-0}"
COVERAGE_KAPPA="${COVERAGE_KAPPA:-}"
COVERAGE_MU_HEAD="${COVERAGE_MU_HEAD:-}"
LA_TAU="${LA_TAU:-}"
LA_EPSILON="${LA_EPSILON:-}"

# List of random seeds for the sequential runs (default: 5 runs per config)
SEEDS=(42 43 44 45 46)
RUNS_PER_CONFIG=${#SEEDS[@]}
TOTAL_ALL_RUNS=$((TOTAL_DATASETS * TOTAL_STRATEGIES * RUNS_PER_CONFIG))
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
echo "  Strategies    : ${STRATEGIES[*]} ($TOTAL_STRATEGIES strategy/strategies)"
echo "  Model         : $MODEL"
echo "  Datasets      : ${DATASETS[*]} ($TOTAL_DATASETS dataset(s))"
echo "  Runs/Config   : $RUNS_PER_CONFIG (Seeds: ${SEEDS[*]})"
echo "  Total Runs    : $TOTAL_ALL_RUNS total sequential FL experiments"
echo "  Rounds        : $ROUNDS | Batch Size: $BATCH_SIZE | LR: $LR ($OPTIMIZER)"
echo "  Partition     : $PARTITION | Mode: $LR_MODE"
echo "=================================================================="

# Pre-run cleanup before starting suite
cleanup_between_runs

GLOBAL_RUN_COUNTER=0

# Sequential execution loop across each dataset
for d_idx in "${!DATASETS[@]}"; do
    DATASET_NUM=$((d_idx + 1))
    CURRENT_DATASET="${DATASETS[$d_idx]}"

    echo ""
    echo "##################################################################"
    echo "  [Dataset $DATASET_NUM/$TOTAL_DATASETS] Target Dataset: $CURRENT_DATASET"
    echo "##################################################################"

    # Loop across each strategy for the current dataset
    for s_idx in "${!STRATEGIES[@]}"; do
        STRAT_NUM=$((s_idx + 1))
        CURRENT_STRATEGY="${STRATEGIES[$s_idx]}"

        # Determine Base Name for this (dataset, strategy) combination
        if [ -n "$CUSTOM_BASE_NAME" ]; then
            if [ "$TOTAL_DATASETS" -eq 1 ] && [ "$TOTAL_STRATEGIES" -eq 1 ]; then
                BASE_NAME="$CUSTOM_BASE_NAME"
            else
                BASE_NAME="${CUSTOM_BASE_NAME}_${CURRENT_DATASET}_${CURRENT_STRATEGY}"
            fi
        else
            BASE_NAME="${CURRENT_DATASET}_${CURRENT_STRATEGY}_${MODEL}"
        fi

        echo ""
        echo "------------------------------------------------------------------"
        echo "  --> [Dataset: $CURRENT_DATASET ($DATASET_NUM/$TOTAL_DATASETS) | Strategy: $CURRENT_STRATEGY ($STRAT_NUM/$TOTAL_STRATEGIES)]"
        echo "      Base Name: $BASE_NAME | Runs: $RUNS_PER_CONFIG"
        echo "------------------------------------------------------------------"

        # Sequential execution loop for each seed
        for i in "${!SEEDS[@]}"; do
            RUN_NUM=$((i + 1))
            SEED=${SEEDS[$i]}
            RUN_NAME="${BASE_NAME}_${RUN_NUM}"
            GLOBAL_RUN_COUNTER=$((GLOBAL_RUN_COUNTER + 1))

            echo ""
            echo ">>> [Overall: $GLOBAL_RUN_COUNTER/$TOTAL_ALL_RUNS | $CURRENT_DATASET | $CURRENT_STRATEGY: Run $RUN_NUM/$RUNS_PER_CONFIG] (Seed: $SEED, Name: $RUN_NAME) <<<"

            EXTRA_ARGS=()
            [ -n "$COVERAGE_KAPPA" ] && EXTRA_ARGS+=(--coverage_kappa "$COVERAGE_KAPPA")
            [ -n "$COVERAGE_MU_HEAD" ] && EXTRA_ARGS+=(--coverage_mu_head "$COVERAGE_MU_HEAD")
            [ -n "$LA_TAU" ] && EXTRA_ARGS+=(--la_tau "$LA_TAU")
            [ -n "$LA_EPSILON" ] && EXTRA_ARGS+=(--la_epsilon "$LA_EPSILON")

            python -u -m experiments.run_simulation \
                --rounds "$ROUNDS" \
                --lr_mode "$LR_MODE" \
                --strategies "$CURRENT_STRATEGY" \
                --partitions "$PARTITION" \
                --batch_size "$BATCH_SIZE" \
                --datasets "$CURRENT_DATASET" \
                --local_test \
                --model "$MODEL" \
                --optimizer "$OPTIMIZER" \
                --learning_rate "$LR" \
                --momentum "$MOMENTUM" \
                --seed "$SEED" \
                --timeout_per_run "$TIMEOUT" \
                --name "$RUN_NAME" \
                "${EXTRA_ARGS[@]}"

            cleanup_between_runs
        done
    done
done

echo ""
echo "=================================================================="
echo "  All $TOTAL_ALL_RUNS Sequential FL Experiments Successfully Completed! "
echo "  Datasets  : ${DATASETS[*]}"
echo "  Strategies: ${STRATEGIES[*]}"
echo "  Runs Each : $RUNS_PER_CONFIG"
echo "=================================================================="

echo ""
echo "=================================================================="
echo "  Pushing experiment results to GitHub..."
echo "=================================================================="
git add results/
git commit -m "feat(benchmarks): results for ${DATASETS[*]} x ${STRATEGIES[*]} (${MODEL}, ${TOTAL_ALL_RUNS} runs)" || echo "No new changes to commit."
CURRENT_BRANCH=$(git branch --show-current 2>/dev/null || git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "main")
git push origin "$CURRENT_BRANCH" || echo "[Warning] Git push failed. Please push manually from the VM."
