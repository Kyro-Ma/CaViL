#!/usr/bin/env bash
set -euo pipefail

# Simple Prompt Tuning Runner with Logging
# Usage: ./run_prompt.sh <category1> [category2] ...
# Example: ./run_prompt.sh Baby_Products
# Or with custom params: LR=1e-4 NUM_EPOCHS=3 ./run_prompt_tuning_simple.sh Baby_Products

# Configuration
MAX_LENGTH="${MAX_LENGTH:-2048}"
LR="${LR:-1e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-5e-5}"
NUM_EPOCHS="${NUM_EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-1}"

# Get script directory
SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SRC_DIR}"

# Create logs directory
LOGS_DIR="./logs"
mkdir -p "${LOGS_DIR}"

# If no categories provided, show usage
if [ "$#" -eq 0 ]; then
    echo "[ERROR] Please provide at least one category as argument."
    echo "Usage: ./run_prompt_tuning_simple.sh <category1> [category2] ..."
    echo "Example: ./run_prompt_tuning_simple.sh Baby_Products"
    exit 1
fi

# Process each category
for category in "$@"; do
    echo ""
    echo "============================================"
    echo "Starting prompt tuning for: ${category}"
    echo "============================================"
    
    # Setup output directory
    OUTPUT_DIR="./new_out_finetuned/${category}/prompt_tuning"
    
    # Create output directory
    mkdir -p "${OUTPUT_DIR}"
    
    # Setup log file with timestamp
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    LOG_FILE="${LOGS_DIR}/${category}_tuning_${TIMESTAMP}.log"
    GPU_LOG_FILE="${LOGS_DIR}/${category}_tuning_${TIMESTAMP}_gpu.log"
    
    echo "Log file: ${LOG_FILE}"
    echo "GPU log file: ${GPU_LOG_FILE}"
    
    # Log header
    {
        echo "======================================"
        echo "Prompt Tuning Run: ${category}"
        echo "Start Time: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "======================================"
        echo ""
        echo "Configuration:"
        echo "  Max Length: ${MAX_LENGTH}"
        echo "  Learning Rate: ${LR}"
        echo "  Weight Decay: ${WEIGHT_DECAY}"
        echo "  Epochs: ${NUM_EPOCHS}"
        echo "  Batch Size: ${BATCH_SIZE}"
        echo "  Save LoRA per epoch: YES (default)"
        echo ""
        echo "Paths:"
        echo "  Output: ${OUTPUT_DIR}"
        echo "  Category: ${category}"
        echo ""
    } | tee "${LOG_FILE}"
    
    # Function to monitor GPU in background
    monitor_gpu() {
        local gpu_log_file=$1
        while true; do
            {
                echo "=== $(date '+%Y-%m-%d %H:%M:%S') ==="
                nvidia-smi --query-gpu=utilization.gpu,utilization.memory,temperature.gpu,power.draw --format=csv,noheader,nounits 2>/dev/null || echo "GPU monitoring unavailable"
            } >> "${gpu_log_file}"
            sleep 10
        done
    }
    
    # Start GPU monitoring in background (if nvidia-smi available)
    GPU_MONITOR_PID=""
    if command -v nvidia-smi &> /dev/null; then
        monitor_gpu "${GPU_LOG_FILE}" &
        GPU_MONITOR_PID=$!
        echo "GPU monitoring started (PID: ${GPU_MONITOR_PID})" | tee -a "${LOG_FILE}"
    fi
    
    # Build command
    CMD="python3 prompt_tuning.py"
    CMD="${CMD} --category '${category}'"
    CMD="${CMD} --finetune_output_dir '${OUTPUT_DIR}'"
    CMD="${CMD} --max_length ${MAX_LENGTH}"
    CMD="${CMD} --batch_size ${BATCH_SIZE}"
    CMD="${CMD} --lr ${LR}"
    CMD="${CMD} --weight_decay ${WEIGHT_DECAY}"
    CMD="${CMD} --num_epochs ${NUM_EPOCHS}"
    CMD="${CMD} --save_every_epoch"
    
    # Record start time
    START_TIME=$(date +%s)
    echo "" | tee -a "${LOG_FILE}"
    echo "Command:" | tee -a "${LOG_FILE}"
    echo "${CMD}" | tee -a "${LOG_FILE}"
    echo "" | tee -a "${LOG_FILE}"
    
    # Run training
    if eval "${CMD}" 2>&1 | tee -a "${LOG_FILE}"; then
        STATUS="SUCCESS"
    else
        STATUS="FAILED"
    fi
    
    # Stop GPU monitoring
    if [ -n "${GPU_MONITOR_PID}" ]; then
        kill ${GPU_MONITOR_PID} 2>/dev/null || true
    fi
    
    # Record end time and duration
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    DURATION_MIN=$((DURATION / 60))
    DURATION_SEC=$((DURATION % 60))
    
    # Log summary
    {
        echo ""
        echo "======================================"
        echo "Run Summary"
        echo "======================================"
        echo "Category: ${category}"
        echo "Status: ${STATUS}"
        echo "Start Time: $(date -d @${START_TIME} '+%Y-%m-%d %H:%M:%S')"
        echo "End Time: $(date -d @${END_TIME} '+%Y-%m-%d %H:%M:%S')"
        echo "Duration: ${DURATION_MIN}m ${DURATION_SEC}s (${DURATION}s)"
        echo "Output Directory: ${OUTPUT_DIR}"
        echo "======================================"
    } | tee -a "${LOG_FILE}"
    
    echo "✓ Prompt tuning completed for ${category}"
done

echo ""
echo "============================================"
echo "All categories processed!"
echo "Logs saved in: ${LOGS_DIR}/"
echo "============================================"
