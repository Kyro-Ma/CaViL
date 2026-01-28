#!/bin/bash
# ============================================
#  LaViC Image Crawl Runner
# ============================================

if [ -z "$1" ]; then
  echo "Use: bash run_crawl.sh <category>"
  echo "EXP: bash run_crawl.sh office_products"
  echo "BAC: nohup bash run_crawl.sh <category> > nohup_crawl.log 2>&1 &"
  exit 1
fi

CATEGORY=$1
LOG_DIR="logs"
LOG_FILE="${LOG_DIR}/crawl_${CATEGORY}_$(date +%Y%m%d_%H%M%S).log"

# make directory for logs
mkdir -p "$LOG_DIR"

echo "============================================" | tee -a "$LOG_FILE"
echo "Run crawl image: [$CATEGORY]" | tee -a "$LOG_FILE"
echo "Start time: $(date)" | tee -a "$LOG_FILE"
echo "--------------------------------------------" | tee -a "$LOG_FILE"

# start time
START_TIME=$(date +%s)

# logs
python3 crawl_images_per_cate.py --category "$CATEGORY" 2>&1 | tee -a "$LOG_FILE"

# end time
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo "--------------------------------------------" | tee -a "$LOG_FILE"
echo "Crwal finish [$CATEGORY]" | tee -a "$LOG_FILE"
echo "End time: $(date)" | tee -a "$LOG_FILE"
echo "Overall time: ${ELAPSED} s" | tee -a "$LOG_FILE"

# count images #
TRAIN_DIR="../data/train_images/${CATEGORY}"
VALID_DIR="../data/valid_images/${CATEGORY}"

TRAIN_COUNT=$(find "$TRAIN_DIR" -type f -name "*.jpg" 2>/dev/null | wc -l)
VALID_COUNT=$(find "$VALID_DIR" -type f -name "*.jpg" 2>/dev/null | wc -l)

TOTAL_COUNT=$((TRAIN_COUNT + VALID_COUNT))

echo "--------------------------------------------" | tee -a "$LOG_FILE"
echo "Image:" | tee -a "$LOG_FILE"
echo "Train image #: $TRAIN_COUNT" | tee -a "$LOG_FILE"
echo "Valid image #: $VALID_COUNT" | tee -a "$LOG_FILE"
echo "Total: $TOTAL_COUNT" | tee -a "$LOG_FILE"
echo "Logs: $LOG_FILE" | tee -a "$LOG_FILE"
echo "============================================" | tee -a "$LOG_FILE"
