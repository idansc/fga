#!/bin/bash
# Regenerate the bottom-up image features on a CUDA node, then assemble the h5.
#
#   SBATCH_PARTITION=<p> SBATCH_ACCOUNT=<a> SPLIT=val sbatch slurm/extract_features.sbatch
#
# Do this only if the original frcnn_features_new.h5 cannot be found. It uses the
# same detector the paper did -- Faster R-CNN, ResNeXt-101, fine-tuned on Visual
# Genome -- so the result is equivalent to the original features, not a substitute.
#
# The extractor depends on vqa-maskrcnn-benchmark, which needs CUDA to compile its
# custom ops. That is why this runs on the cluster and not on a laptop.

#SBATCH --job-name=fga-features
#SBATCH --partition=gpu
#SBATCH --account=default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/features-%j.out
#SBATCH --error=logs/features-%j.err

set -euo pipefail

mkdir -p logs

REPO="${REPO:-$HOME/fga}"
SPLIT="${SPLIT:?set SPLIT=train|val|test}"
DATA_DIR="${DATA_DIR:-$REPO/data}"
IMAGE_DIR="${IMAGE_DIR:-$DATA_DIR/images}"
DETECTOR_DIR="${DETECTOR_DIR:-$DATA_DIR/detector}"
FEATURE_DIR="${FEATURE_DIR:-$DATA_DIR/extracted/$SPLIT}"
BENCHMARK_DIR="${BENCHMARK_DIR:-$HOME/vqa-maskrcnn-benchmark}"

# Which image directories back each split.
case "$SPLIT" in
    val)   IMAGE_SUBDIRS="VisualDialog_val2018" ;;
    test)  IMAGE_SUBDIRS="VisualDialog_test2018" ;;
    train) IMAGE_SUBDIRS="train2014 val2014" ;;
    *) echo "SPLIT must be train, val or test" >&2; exit 1 ;;
esac

cd "$REPO"
source .venv/bin/activate

nvidia-smi

bash scripts/download_extraction_assets.sh detector
bash scripts/download_extraction_assets.sh "$SPLIT"

# The extractor itself. Build once; the custom ops need a GPU node to compile.
if [ ! -d "$BENCHMARK_DIR" ]; then
    echo "Building vqa-maskrcnn-benchmark in $BENCHMARK_DIR"
    git clone https://gitlab.com/vedanuj/vqa-maskrcnn-benchmark.git "$BENCHMARK_DIR"
    (cd "$BENCHMARK_DIR" && python setup.py build develop)
fi

mkdir -p "$FEATURE_DIR"
for subdir in $IMAGE_SUBDIRS; do
    echo "extracting from $IMAGE_DIR/$subdir"
    python "$BENCHMARK_DIR/script/extract_features.py" \
        --model_file  "$DETECTOR_DIR/detectron_model.pth" \
        --config_file "$DETECTOR_DIR/detectron_model.yaml" \
        --image_dir   "$IMAGE_DIR/$subdir" \
        --output_folder "$FEATURE_DIR" \
        --num_features 36 \
        --batch_size 8
done

# Assemble into the layout the model reads, ordered by visdial_params.json.
python scripts/build_features_h5.py \
    --source npy \
    --features_dir "$FEATURE_DIR" \
    --split "$SPLIT" \
    --visdial_params_path "$DATA_DIR/visdial_params.json" \
    --output "$DATA_DIR/frcnn_features_new.h5" \
    --num_regions 37 \
    --overwrite

echo "Wrote ${SPLIT}_features. Validate with:"
echo "  python scripts/check_data.py --splits $SPLIT --image_features_path $DATA_DIR/frcnn_features_new.h5"
