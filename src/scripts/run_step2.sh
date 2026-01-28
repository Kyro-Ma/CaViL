#!/bin/bash
# ============================================================
# Simple runner for step2_add_llava_descriptions.py
# Usage: bash run_step2.sh Automotive
# ============================================================

# ---- Check input ----
if [ $# -ne 1 ]; then
  echo "Usage: bash $0 <CATEGORY>"
  exit 1
fi

CATEGORY=$1
DATE=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="../logs"
mkdir -p $LOG_DIR
LOG_FILE="$LOG_DIR/run_step2_${CATEGORY}_${DATE}.log"

# ---- Environment Info ----
echo "===============================" | tee $LOG_FILE
echo "Run started at: $(date)" | tee -a $LOG_FILE
echo "Category: $CATEGORY" | tee -a $LOG_FILE
echo "Hostname: $(hostname)" | tee -a $LOG_FILE
echo "Python: $(python3 --version 2>&1)" | tee -a $LOG_FILE
echo "Torch: $(python3 -c 'import torch; print(torch.__version__)')" | tee -a $LOG_FILE
echo "CUDA: $(python3 -c 'import torch; print(torch.version.cuda)')" | tee -a $LOG_FILE
echo "NVIDIA-SMI:" | tee -a $LOG_FILE
nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv,noheader,nounits | tee -a $LOG_FILE
echo "===============================" | tee -a $LOG_FILE

# ---- Paths ----
TRAIN_IN="../../data/item2meta_train_${CATEGORY}_4.json"
TRAIN_OUT="../../data/item2meta_train_${CATEGORY}.with_desc_4.json"
VALID_IN="../../data/item2meta_valid_${CATEGORY}_4.jsonl"
VALID_OUT="../../data/item2meta_valid_${CATEGORY}.with_desc_4.jsonl"
TRAIN_IMG="../../data/train_images"
VALID_IMG="../../data/valid_images"

# ---- Start timer ----
START_TIME=$(date +%s)

# ---- Run ----
echo "[INFO] Starting LLaVA caption generation..." | tee -a $LOG_FILE
python3 step2_add_llava_descriptions.py \
  --model_name "llava-hf/llava-v1.6-mistral-7b-hf" \
  --device auto \
  --max_new_tokens 128 \
  --train_in $TRAIN_IN \
  --train_out $TRAIN_OUT \
  --train_images_dir $TRAIN_IMG \
  --valid_in $VALID_IN \
  --valid_out $VALID_OUT \
  --valid_images_dir $VALID_IMG | tee -a $LOG_FILE

# ---- End timer ----
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

# ---- Summary ----
echo "===============================" | tee -a $LOG_FILE
echo "Run finished at: $(date)" | tee -a $LOG_FILE
echo "Total runtime: ${ELAPSED} seconds" | tee -a $LOG_FILE
echo "Output files:" | tee -a $LOG_FILE
echo "  $TRAIN_OUT" | tee -a $LOG_FILE
echo "  $VALID_OUT" | tee -a $LOG_FILE
echo "===============================" | tee -a $LOG_FILE
