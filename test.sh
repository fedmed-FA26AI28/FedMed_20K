#!/bin/bash
# ==============================================================================
# FedMedAI: Sequential Multi-Run Benchmark Runner for Remote GPU VM
# ==============================================================================
# Quick Configuration Options:
# ------------------------------------------------------------------------------
# 1. Edit the variables directly below in this file (lines 20-65)
# 2. Or pass as positional CLI arguments when invoking:
#      ./test.sh "<strategy1>..." <model> "<dataset1>..." [partitions] [custom_base_name]
#      e.g.: ./test.sh fedavg baseline organamnist
#      e.g.: ./test.sh fedavg baseline organamnist dirichlet
#      e.g.: ./test.sh fedavg baseline organamnist dirichlet_0.3
#      e.g.: ./test.sh fedavg baseline organamnist "practical dirichlet_0.3"
#      e.g.: ./test.sh "fedavg fedprox" baseline "organamnist pathmnist" dirichlet
#      e.g.: ./test.sh fedbns cnn organamnist,pathmnist practical
# 3. Or pass via environment variables:
#      PARTITION=dirichlet ./test.sh
#      PARTITIONS="practical dirichlet_0.3" STRATEGIES="fedavg fedprox" ./test.sh
# ==============================================================================

set -eo pipefail

# ─── QUICK CHANGE PARAMETERS ──────────────────────────────────────────────────
# Default values can be edited here, or overridden via CLI arguments:

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

# Partitions: edit DEFAULT_PARTITIONS below or pass via CLI $4 / env PARTITIONS / env PARTITION.
# Supported options:
#   - 'practical'     : Clinical institution imbalance (FedSLD ISBI 2022)
#   - 'pathological'  : Class-subset non-IID
#   - 'dirichlet'     : Dirichlet non-IID (defaults to dirichlet_0.3)
#   - 'dirichlet_0.3' : Dirichlet with alpha=0.3
#   - 'dirichlet_0.1' : Dirichlet with alpha=0.1
# Can be single or multiple: e.g. (practical) or (dirichlet_0.3) or (practical dirichlet_0.3)
DEFAULT_PARTITIONS=(practical)

# Smart detection of CLI $4 and $5 between Partitions and Custom Base Name:
ARG4="${4:-}"
ARG5="${5:-}"

RAW_PARTITIONS="${PARTITIONS:-${PARTITION:-}}"
CUSTOM_BASE_NAME="${BASE_NAME:-}"

is_partition_like() {
    local val
    val="$(echo "$1" | tr '[:upper:]' '[:lower:]')"
    [[ "$val" =~ practical|pathological|dirichlet ]]
}

if [ -n "$ARG4" ]; then
    if is_partition_like "$ARG4"; then
        RAW_PARTITIONS="$ARG4"
        [ -n "$ARG5" ] && CUSTOM_BASE_NAME="$ARG5"
    else
        CUSTOM_BASE_NAME="$ARG4"
        [ -n "$ARG5" ] && RAW_PARTITIONS="$ARG5"
    fi
fi

if [ -n "$RAW_PARTITIONS" ]; then
    CLEAN_PARTITIONS="${RAW_PARTITIONS//,/ }"
    read -r -a PARSED_PARTITIONS <<< "$CLEAN_PARTITIONS"
else
    PARSED_PARTITIONS=("${DEFAULT_PARTITIONS[@]}")
fi

# Optional Dirichlet alpha override (e.g., ALPHA=0.1 or DIRICHLET_ALPHA=0.1; defaults to 0.3)
DIRICHLET_ALPHA="${ALPHA:-${DIRICHLET_ALPHA:-0.3}}"

# Normalize "dirichlet" -> "dirichlet_${DIRICHLET_ALPHA}" (default: dirichlet_0.3)
PARTITIONS=()
for p in "${PARSED_PARTITIONS[@]}"; do
    p_lower="$(echo "$p" | tr '[:upper:]' '[:lower:]')"
    if [ "$p_lower" = "dirichlet" ]; then
        PARTITIONS+=("dirichlet_${DIRICHLET_ALPHA}")
    else
        PARTITIONS+=("$p_lower")
    fi
done

TOTAL_PARTITIONS=${#PARTITIONS[@]}

# Training Hyperparameters (can also be overridden via environment variables):
ROUNDS="${ROUNDS:-80}"
BATCH_SIZE="${BATCH_SIZE:-256}"
LR="${LR:-0.01}"
MOMENTUM="${MOMENTUM:-0.9}"
OPTIMIZER="${OPTIMIZER:-sgd}"
LR_MODE="${LR_MODE:-fixed}"
TIMEOUT="${TIMEOUT:-0}"
SCR_GAMMA="${SCR_GAMMA:-}"
SCR_HEAD_LAMBDA="${SCR_HEAD_LAMBDA:-}"
PROXIMAL_MU="${PROXIMAL_MU:-}"

# List of random seeds for the sequential runs (default: 5 runs per config)
SEEDS=(42 43 44 45 46)
RUNS_PER_CONFIG=${#SEEDS[@]}
TOTAL_ALL_RUNS=$((TOTAL_DATASETS * TOTAL_PARTITIONS * TOTAL_STRATEGIES * RUNS_PER_CONFIG))
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
echo "  Partitions    : ${PARTITIONS[*]} ($TOTAL_PARTITIONS partition(s))"
echo "  Runs/Config   : $RUNS_PER_CONFIG (Seeds: ${SEEDS[*]})"
echo "  Total Runs    : $TOTAL_ALL_RUNS total sequential FL experiments"
echo "  Rounds        : $ROUNDS | Batch Size: $BATCH_SIZE | LR: $LR ($OPTIMIZER)"
echo "  Mode          : $LR_MODE"
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

    # Sequential execution loop across each partition mode
    for p_idx in "${!PARTITIONS[@]}"; do
        PART_NUM=$((p_idx + 1))
        CURRENT_PARTITION="${PARTITIONS[$p_idx]}"

        echo ""
        echo "=================================================================="
        echo "  [Partition $PART_NUM/$TOTAL_PARTITIONS] Target Partition: $CURRENT_PARTITION"
        echo "=================================================================="

        # Loop across each strategy for the current dataset & partition
        for s_idx in "${!STRATEGIES[@]}"; do
            STRAT_NUM=$((s_idx + 1))
            CURRENT_STRATEGY="${STRATEGIES[$s_idx]}"

            # Determine Base Name for this (dataset, partition, strategy) combination
            if [ -n "$CUSTOM_BASE_NAME" ]; then
                if [ "$TOTAL_DATASETS" -eq 1 ] && [ "$TOTAL_PARTITIONS" -eq 1 ] && [ "$TOTAL_STRATEGIES" -eq 1 ]; then
                    BASE_NAME="$CUSTOM_BASE_NAME"
                else
                    BASE_NAME="${CUSTOM_BASE_NAME}_${CURRENT_DATASET}_${CURRENT_PARTITION}_${CURRENT_STRATEGY}_${MODEL}"
                fi
            else
                BASE_NAME="${CURRENT_DATASET}_${CURRENT_PARTITION}_${CURRENT_STRATEGY}_${MODEL}"
            fi

            echo ""
            echo "------------------------------------------------------------------"
            echo "  --> [Dataset: $CURRENT_DATASET ($DATASET_NUM/$TOTAL_DATASETS) | Partition: $CURRENT_PARTITION ($PART_NUM/$TOTAL_PARTITIONS) | Strategy: $CURRENT_STRATEGY ($STRAT_NUM/$TOTAL_STRATEGIES)]"
            echo "      Base Name: $BASE_NAME | Runs: $RUNS_PER_CONFIG"
            echo "------------------------------------------------------------------"

            # Sequential execution loop for each seed
            for i in "${!SEEDS[@]}"; do
                RUN_NUM=$((i + 1))
                SEED=${SEEDS[$i]}
                RUN_NAME="${BASE_NAME}_${RUN_NUM}"
                GLOBAL_RUN_COUNTER=$((GLOBAL_RUN_COUNTER + 1))

                echo ""
                echo ">>> [Overall: $GLOBAL_RUN_COUNTER/$TOTAL_ALL_RUNS | $CURRENT_DATASET | $CURRENT_PARTITION | $CURRENT_STRATEGY: Run $RUN_NUM/$RUNS_PER_CONFIG] (Seed: $SEED, Name: $RUN_NAME) <<<"

                EXTRA_ARGS=()
                [ -n "$SCR_GAMMA" ] && EXTRA_ARGS+=(--scr_gamma "$SCR_GAMMA")
                [ -n "$SCR_HEAD_LAMBDA" ] && EXTRA_ARGS+=(--scr_head_lambda "$SCR_HEAD_LAMBDA")
                [ -n "$PROXIMAL_MU" ] && EXTRA_ARGS+=(--proximal_mu "$PROXIMAL_MU")

                python -u -m experiments.run_simulation \
                    --rounds "$ROUNDS" \
                    --lr_mode "$LR_MODE" \
                    --strategies "$CURRENT_STRATEGY" \
                    --partitions "$CURRENT_PARTITION" \
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
done

echo ""
echo "=================================================================="
echo "  All $TOTAL_ALL_RUNS Sequential FL Experiments Successfully Completed! "
echo "  Datasets  : ${DATASETS[*]}"
echo "  Partitions: ${PARTITIONS[*]}"
echo "  Strategies: ${STRATEGIES[*]}"
echo "  Runs Each : $RUNS_PER_CONFIG"
echo "=================================================================="

echo ""
echo "=================================================================="
echo "  Pushing experiment results to GitHub..."
echo "=================================================================="
git add results/
git commit -m "feat(benchmarks): results for ${DATASETS[*]} x ${PARTITIONS[*]} x ${STRATEGIES[*]} (${MODEL}, ${TOTAL_ALL_RUNS} runs)" || echo "No new changes to commit."
CURRENT_BRANCH=$(git branch --show-current 2>/dev/null || git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "main")
git pull --rebase origin "$CURRENT_BRANCH" || echo "[Warning] Git pull --rebase failed. Please check for conflicts."
git push origin "$CURRENT_BRANCH" || echo "[Warning] Git push failed. Please push manually from the VM."

