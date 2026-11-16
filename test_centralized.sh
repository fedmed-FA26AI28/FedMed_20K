#!/bin/bash
# ==============================================================================
# FedMedAI: Sequential Centralized Training Benchmark Runner for Remote GPU VM
# ==============================================================================
# Quick Configuration Options:
# ------------------------------------------------------------------------------
# 1. Edit the variables directly below in this file (lines 30-150)
# 2. Or pass as positional CLI arguments when invoking:
#      ./test_centralized.sh [model] [dataset] [custom_base_name]
#      e.g.: ./test_centralized.sh baseline_bn pathmnist
#      e.g.: ./test_centralized.sh pathmnist baseline_bn
#      e.g.: ./test_centralized.sh cnn
#      e.g.: ./test_centralized.sh organamnist
#      e.g.: ./test_centralized.sh "baseline_bn cnn" "pathmnist organamnist"
# 3. Or pass as named CLI flags:
#      ./test_centralized.sh --model cnn --dataset bloodmnist
#      ./test_centralized.sh -m baseline_bn -d pathmnist
#      ./test_centralized.sh --dataset organamnist --epochs 400
# 4. Or pass via environment variables:
#      MODEL=cnn DATASET=bloodmnist ./test_centralized.sh
#      MODELS="baseline_bn cnn" DATASETS="pathmnist bloodmnist" ./test_centralized.sh
#      EPOCHS=400 BATCH_SIZE=256 OPTIMIZER=sgd ./test_centralized.sh
# ==============================================================================

set -eo pipefail

# ─── QUICK CHANGE PARAMETERS ──────────────────────────────────────────────────
# Default values match configs/experiment.yaml (except epochs=400 & seeds 42..46):

DEFAULT_MODELS=(baseline_bn)
DEFAULT_DATASETS=(pathmnist)

RAW_MODELS="${MODELS:-${MODEL:-}}"
RAW_DATASETS="${DATASETS:-${DATASET:-}}"
CUSTOM_BASE_NAME="${BASE_NAME:-}"

is_dataset_like() {
    local val
    val="$(echo "$1" | tr '[:upper:]' '[:lower:]')"
    [[ "$val" =~ mnist ]]
}

is_model_like() {
    local val
    val="$(echo "$1" | tr '[:upper:]' '[:lower:]')"
    [[ "$val" =~ baseline|cnn|paper|classic ]]
}

# Parse named CLI flags and collect positional arguments
POSITIONAL=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        -m|--model|--models)
            RAW_MODELS="$2"
            shift 2
            ;;
        -d|--dataset|--datasets)
            RAW_DATASETS="$2"
            shift 2
            ;;
        -n|--name|--base_name|--base-name)
            CUSTOM_BASE_NAME="$2"
            shift 2
            ;;
        -e|--epochs)
            EPOCHS="$2"
            shift 2
            ;;
        -b|--batch_size|--batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --opt|--optimizer)
            OPTIMIZER="$2"
            shift 2
            ;;
        --lr|--learning_rate|--learning-rate)
            LR="$2"
            shift 2
            ;;
        -c|--config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        -h|--help)
            echo "=================================================================="
            echo "  FedMedAI Centralized Training Benchmark Suite Runner"
            echo "=================================================================="
            echo "Usage: ./test_centralized.sh [options] [model] [dataset] [custom_base_name]"
            echo ""
            echo "Options:"
            echo "  -m, --model <name>       Model architecture (baseline_bn, baseline, cnn)"
            echo "  -d, --dataset <name>     Dataset name (pathmnist, bloodmnist, organamnist, dermamnist)"
            echo "  -e, --epochs <int>       Number of training epochs (default: 400)"
            echo "  -b, --batch_size <int>   Batch size (default: 256)"
            echo "      --optimizer <name>   Optimizer: sgd, adam, adamw (default: sgd)"
            echo "      --lr <float>         Initial learning rate (default: 0.01)"
            echo "  -n, --name <str>         Custom base name for output run directory"
            echo "  -c, --config <path>      Path to YAML experiment config (default: configs/experiment.yaml)"
            echo "  -h, --help               Show this help message and exit"
            echo ""
            echo "Positional Usage Examples:"
            echo "  ./test_centralized.sh baseline_bn pathmnist"
            echo "  ./test_centralized.sh pathmnist baseline_bn"
            echo "  ./test_centralized.sh cnn"
            echo "  ./test_centralized.sh organamnist"
            echo "  ./test_centralized.sh \"baseline_bn cnn\" \"pathmnist bloodmnist\""
            echo ""
            echo "Named Flags Examples:"
            echo "  ./test_centralized.sh --model cnn --dataset bloodmnist"
            echo "  ./test_centralized.sh -m baseline_bn -d organamnist -e 400"
            echo ""
            echo "Environment Variable Examples:"
            echo "  MODEL=cnn DATASET=organamnist ./test_centralized.sh"
            echo "  EPOCHS=400 BATCH_SIZE=256 ./test_centralized.sh"
            echo "=================================================================="
            exit 0
            ;;
        *)
            POSITIONAL+=("$1")
            shift
            ;;
    esac
done

# Smart detection of positional arguments (handles either [model] [dataset] or [dataset] [model])
if [ ${#POSITIONAL[@]} -gt 0 ]; then
    ARG1="${POSITIONAL[0]:-}"
    ARG2="${POSITIONAL[1]:-}"
    ARG3="${POSITIONAL[2]:-}"

    if is_dataset_like "$ARG1" && ! is_dataset_like "$ARG2"; then
        [ -z "$RAW_DATASETS" ] && RAW_DATASETS="$ARG1"
        [ -n "$ARG2" ] && [ -z "$RAW_MODELS" ] && RAW_MODELS="$ARG2"
        [ -n "$ARG3" ] && [ -z "$CUSTOM_BASE_NAME" ] && CUSTOM_BASE_NAME="$ARG3"
    elif is_model_like "$ARG1" || [ -n "$ARG2" ]; then
        [ -z "$RAW_MODELS" ] && RAW_MODELS="$ARG1"
        [ -n "$ARG2" ] && [ -z "$RAW_DATASETS" ] && RAW_DATASETS="$ARG2"
        [ -n "$ARG3" ] && [ -z "$CUSTOM_BASE_NAME" ] && CUSTOM_BASE_NAME="$ARG3"
    else
        if is_dataset_like "$ARG1"; then
            [ -z "$RAW_DATASETS" ] && RAW_DATASETS="$ARG1"
        else
            [ -z "$RAW_MODELS" ] && RAW_MODELS="$ARG1"
        fi
    fi
fi

# Parse models list (supports space-separated or comma-separated)
if [ -n "$RAW_MODELS" ]; then
    CLEAN_MODELS="${RAW_MODELS//,/ }"
    read -r -a MODELS <<< "$CLEAN_MODELS"
else
    MODELS=("${DEFAULT_MODELS[@]}")
fi
TOTAL_MODELS=${#MODELS[@]}

# Parse datasets list (supports space-separated or comma-separated)
if [ -n "$RAW_DATASETS" ]; then
    CLEAN_DATASETS="${RAW_DATASETS//,/ }"
    read -r -a DATASETS <<< "$CLEAN_DATASETS"
else
    DATASETS=("${DEFAULT_DATASETS[@]}")
fi
TOTAL_DATASETS=${#DATASETS[@]}

# Centralized Training Hyperparameters (Identical to configs/experiment.yaml, except epochs=400)
CONFIG_FILE="${CONFIG_FILE:-${CONFIG:-configs/experiment.yaml}}"
EPOCHS="${EPOCHS:-400}"
BATCH_SIZE="${BATCH_SIZE:-256}"
OPTIMIZER="${OPTIMIZER:-sgd}"
LR="${LR:-0.01}"
MOMENTUM="${MOMENTUM:-0.9}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
CLIP_GRAD_NORM="${CLIP_GRAD_NORM:-}"
LR_PATIENCE="${LR_PATIENCE:-0}"
LR_FACTOR="${LR_FACTOR:-0.5}"
MIN_LR="${MIN_LR:-1e-6}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-0}"     # 0 = disabled (trains full 400 epochs)
EARLY_STOP_METRIC="${EARLY_STOP_METRIC:-accuracy}" # 'accuracy' or 'loss'
AUGMENT="${AUGMENT:-false}"
RANDOM_CROP="${RANDOM_CROP:-false}"
RANDOM_FLIP="${RANDOM_FLIP:-false}"

# 5 sequential runs per configuration across random seeds (42 43 44 45 46)
SEEDS=(42 43 44 45 46)
RUNS_PER_CONFIG=${#SEEDS[@]}
TOTAL_ALL_RUNS=$((TOTAL_MODELS * TOTAL_DATASETS * RUNS_PER_CONFIG))

# Python executable resolution
PYTHON="${PYTHON:-python}"
# ──────────────────────────────────────────────────────────────────────────────

# 1. Disable XON/XOFF flow control so accidental Ctrl+S does not freeze SSH terminal
stty -ixon 2>/dev/null || true

# 2. Force unbuffered standard I/O for Python in terminal and redirected pipes
export PYTHONUNBUFFERED=1

# 3. Inter-run resource reclamation and cooldown
cleanup_between_runs() {
    echo ""
    echo "=================================================================="
    echo "  Reclaiming GPU memory, closing lingering processes & cooling down..."
    echo "=================================================================="
    pkill -f "experiments.train_centralized" 2>/dev/null || true
    # Cooldown to allow GPU memory and cached tensors to completely clear
    sleep 5
}

# 4. Clean exit trap on interrupt, termination, or SSH SIGHUP disconnect
trap 'echo -e "\n[Interrupted] Killing centralized workers..."; cleanup_between_runs; exit 1' INT TERM HUP

# Print configuration banner
echo "=================================================================="
echo "  FedMedAI Centralized Training Benchmark Suite Starting"
echo "=================================================================="
echo "  Config Source : $CONFIG_FILE"
echo "  Models        : ${MODELS[*]} ($TOTAL_MODELS model(s))"
echo "  Datasets      : ${DATASETS[*]} ($TOTAL_DATASETS dataset(s))"
echo "  Runs/Config   : $RUNS_PER_CONFIG (Seeds: ${SEEDS[*]})"
echo "  Total Runs    : $TOTAL_ALL_RUNS total sequential centralized experiments"
echo "  Epochs        : $EPOCHS | Batch Size: $BATCH_SIZE"
echo "  Optimizer     : $OPTIMIZER | Base LR: $LR | Momentum: $MOMENTUM | Weight Decay: $WEIGHT_DECAY"
if [ "$EARLY_STOP_PATIENCE" -gt 0 ] 2>/dev/null; then
    echo "  Early Stopping: Enabled ($EARLY_STOP_PATIENCE epochs patience, metric: $EARLY_STOP_METRIC)"
else
    echo "  Early Stopping: Disabled (0 patience - running full $EPOCHS epochs)"
fi
echo "  Augmentation  : augment=$AUGMENT, random_crop=$RANDOM_CROP, random_flip=$RANDOM_FLIP"
echo "=================================================================="

# Pre-run cleanup before starting suite
cleanup_between_runs

GLOBAL_RUN_COUNTER=0

# Sequential execution loop across each model
for m_idx in "${!MODELS[@]}"; do
    MODEL_NUM=$((m_idx + 1))
    CURRENT_MODEL="${MODELS[$m_idx]}"

    echo ""
    echo "##################################################################"
    echo "  [Model $MODEL_NUM/$TOTAL_MODELS] Target Architecture: $CURRENT_MODEL"
    echo "##################################################################"

    # Sequential execution loop across each dataset
    for d_idx in "${!DATASETS[@]}"; do
        DATASET_NUM=$((d_idx + 1))
        CURRENT_DATASET="${DATASETS[$d_idx]}"

        # Determine Base Name for this (dataset, model) combination
        if [ -n "$CUSTOM_BASE_NAME" ]; then
            if [ "$TOTAL_DATASETS" -eq 1 ] && [ "$TOTAL_MODELS" -eq 1 ]; then
                BASE_NAME="$CUSTOM_BASE_NAME"
            else
                BASE_NAME="${CUSTOM_BASE_NAME}_${CURRENT_DATASET}_${CURRENT_MODEL}"
            fi
        else
            BASE_NAME="${CURRENT_DATASET}_${CURRENT_MODEL}"
        fi

        echo ""
        echo "=================================================================="
        echo "  [Dataset $DATASET_NUM/$TOTAL_DATASETS] $CURRENT_DATASET | Model: $CURRENT_MODEL ($MODEL_NUM/$TOTAL_MODELS)"
        echo "  Base Name: $BASE_NAME | Runs: $RUNS_PER_CONFIG"
        echo "=================================================================="

        # Sequential execution loop for each seed
        for i in "${!SEEDS[@]}"; do
            RUN_NUM=$((i + 1))
            SEED=${SEEDS[$i]}
            RUN_NAME="${BASE_NAME}_run_${RUN_NUM}"
            GLOBAL_RUN_COUNTER=$((GLOBAL_RUN_COUNTER + 1))

            echo ""
            echo ">>> [Overall: $GLOBAL_RUN_COUNTER/$TOTAL_ALL_RUNS | $CURRENT_DATASET | $CURRENT_MODEL: Run $RUN_NUM/$RUNS_PER_CONFIG] (Seed: $SEED, Name: $RUN_NAME) <<<"

            EXTRA_ARGS=()
            [ -n "$CONFIG_FILE" ] && EXTRA_ARGS+=(--config "$CONFIG_FILE")
            [ -n "$CLIP_GRAD_NORM" ] && EXTRA_ARGS+=(--clip_grad_norm "$CLIP_GRAD_NORM")
            [ -n "$LR_PATIENCE" ] && EXTRA_ARGS+=(--lr_patience "$LR_PATIENCE")
            [ -n "$LR_FACTOR" ] && EXTRA_ARGS+=(--lr_factor "$LR_FACTOR")
            [ -n "$MIN_LR" ] && EXTRA_ARGS+=(--min_lr "$MIN_LR")

            if [ "$AUGMENT" = "true" ] || [ "$AUGMENT" = "1" ]; then
                EXTRA_ARGS+=(--augment)
            else
                EXTRA_ARGS+=(--no-augment)
            fi

            if [ "$RANDOM_CROP" = "true" ] || [ "$RANDOM_CROP" = "1" ]; then
                EXTRA_ARGS+=(--random_crop)
            else
                EXTRA_ARGS+=(--no-random_crop)
            fi

            if [ "$RANDOM_FLIP" = "true" ] || [ "$RANDOM_FLIP" = "1" ]; then
                EXTRA_ARGS+=(--random_flip)
            else
                EXTRA_ARGS+=(--no-random_flip)
            fi

            "$PYTHON" -u -m experiments.train_centralized \
                --dataset "$CURRENT_DATASET" \
                --model "$CURRENT_MODEL" \
                --epochs "$EPOCHS" \
                --batch_size "$BATCH_SIZE" \
                --optimizer "$OPTIMIZER" \
                --learning_rate "$LR" \
                --momentum "$MOMENTUM" \
                --weight_decay "$WEIGHT_DECAY" \
                --early_stop_patience "$EARLY_STOP_PATIENCE" \
                --early_stop_metric "$EARLY_STOP_METRIC" \
                --seed "$SEED" \
                --name "$RUN_NAME" \
                "${EXTRA_ARGS[@]}"

            cleanup_between_runs
        done
    done
done

echo ""
echo "=================================================================="
echo "  All $TOTAL_ALL_RUNS Sequential Centralized Experiments Successfully Completed!"
echo "  Models    : ${MODELS[*]}"
echo "  Datasets  : ${DATASETS[*]}"
echo "  Runs Each : $RUNS_PER_CONFIG"
echo "  Epochs    : $EPOCHS | Batch Size: $BATCH_SIZE | Optimizer: $OPTIMIZER"
echo "=================================================================="

echo ""
echo "=================================================================="
echo "  Pushing centralized experiment results to GitHub..."
echo "=================================================================="
git add results/centralized/ 2>/dev/null || true
git commit -m "feat(benchmarks): centralized results for ${DATASETS[*]} (${MODELS[*]}, ${TOTAL_ALL_RUNS} runs, epochs=${EPOCHS}, bs=${BATCH_SIZE}, opt=${OPTIMIZER})" || echo "No new changes to commit."
CURRENT_BRANCH=$(git branch --show-current 2>/dev/null || git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "main")
git pull --rebase origin "$CURRENT_BRANCH" 2>/dev/null || echo "[Warning] Git pull --rebase failed. Please check for conflicts."
git push origin "$CURRENT_BRANCH" 2>/dev/null || echo "[Warning] Git push failed. Please push manually from the VM."
