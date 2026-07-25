#!/bin/bash
# Run the bottom-up feature extractor across every GPU on the machine.
#
#   bash scripts/extract_sharded.sh <image_dir> <output_dir> [num_gpus]
#
# The extractor is single-GPU and processes images in index order, so the work
# splits cleanly: each GPU takes a contiguous slice via --start_index/--end_index.
# The 123k COCO images behind the VisDial train split take ~13 hours on one L40S
# and well under two across eight.
#
# Safe to re-run: the extractor skips images whose .npy already exists, so an
# interrupted shard resumes rather than starting over.

set -euo pipefail

IMAGE_DIR="${1:?usage: extract_sharded.sh <image_dir> <output_dir> [num_gpus]}"
OUTPUT_DIR="${2:?usage: extract_sharded.sh <image_dir> <output_dir> [num_gpus]}"
NUM_GPUS="${3:-$(nvidia-smi --list-gpus | wc -l)}"

WORK="${WORK:-/data/schwari9}"
PYTHON="${PYTHON:-$WORK/fga/.venv/bin/python}"
EXTRACTOR="${EXTRACTOR:-$WORK/extract_features_vmb.py}"
DETECTOR_DIR="${DETECTOR_DIR:-$WORK/fga/data/detector}"
LOG_DIR="${LOG_DIR:-$WORK/logs}"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

TOTAL=$(find "$IMAGE_DIR" -maxdepth 1 -type f \( -name '*.jpg' -o -name '*.jpeg' -o -name '*.png' \) | wc -l)
if [ "$TOTAL" -eq 0 ]; then
    echo "no images found in $IMAGE_DIR" >&2
    exit 1
fi

PER_GPU=$(( (TOTAL + NUM_GPUS - 1) / NUM_GPUS ))
echo "$TOTAL images, $NUM_GPUS GPUs, $PER_GPU images per shard"

for gpu in $(seq 0 $((NUM_GPUS - 1))); do
    start=$((gpu * PER_GPU))
    end=$((start + PER_GPU))
    [ "$end" -gt "$TOTAL" ] && end=$TOTAL
    [ "$start" -ge "$TOTAL" ] && break

    log="$LOG_DIR/extract_gpu${gpu}.log"
    echo "  gpu $gpu: images [$start, $end)  -> $log"
    CUDA_VISIBLE_DEVICES="$gpu" setsid nohup "$PYTHON" "$EXTRACTOR" \
        --model_file  "$DETECTOR_DIR/detectron_model.pth" \
        --config_file "$DETECTOR_DIR/detectron_model.yaml" \
        --image_dir   "$IMAGE_DIR" \
        --output_folder "$OUTPUT_DIR" \
        --num_features 36 \
        --batch_size 8 \
        --start_index "$start" \
        --end_index "$end" \
        > "$log" 2>&1 < /dev/null &
    disown
done

echo
echo "all shards launched. progress:"
echo "  ls $OUTPUT_DIR | grep -c _info.npy   # of $TOTAL"
echo "  nvidia-smi --query-gpu=index,utilization.gpu --format=csv"
