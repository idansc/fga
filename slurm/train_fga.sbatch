#!/bin/bash
# Train FGA on a SLURM cluster.
#
#   SBATCH_PARTITION=<your-partition> SBATCH_ACCOUNT=<your-account> \
#       sbatch slurm/train_fga.sbatch
#   squeue -u $USER
#   tail -f logs/fga-<jobid>.out
#
# The #SBATCH lines below are placeholders. sbatch reads SBATCH_PARTITION and
# SBATCH_ACCOUNT from the submitting environment and they take precedence, so
# set those rather than committing your own queue names. Adjust --gres, --mem
# and --time to whatever your allocation allows.

#SBATCH --job-name=fga
#SBATCH --partition=gpu
#SBATCH --account=default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/fga-%j.out
#SBATCH --error=logs/fga-%j.err

set -euo pipefail

mkdir -p logs

REPO="${REPO:-$HOME/fga}"
DATA_DIR="${DATA_DIR:-$REPO/data}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO/models/fga-frcnn}"

cd "$REPO"

# Environment. Swap for `module load` / conda if that is how your allocation is set up.
if [ -d "$REPO/.venv" ]; then
    source "$REPO/.venv/bin/activate"
else
    python -m venv "$REPO/.venv"
    source "$REPO/.venv/bin/activate"
    pip install --upgrade pip
    pip install -e ".[train]"
fi

nvidia-smi
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

# One GPU is enough for the paper configuration. For several, replace `python`
# with `torchrun --nproc_per_node=$SLURM_GPUS_ON_NODE` -- Trainer handles DDP.
python scripts/run_visual_dialog.py \
    --visdial_data_path      "$DATA_DIR/visdial_data.h5" \
    --visdial_params_path    "$DATA_DIR/visdial_params.json" \
    --image_features_path    "$DATA_DIR/frcnn_features_new.h5" \
    --dense_annotations_path "$DATA_DIR/visdial_1.0_val_dense_annotations.json" \
    --output_dir             "$OUTPUT_DIR" \
    --do_train --do_eval --write_submission \
    --per_device_train_batch_size 128 \
    --per_device_eval_batch_size 64 \
    --learning_rate 1e-3 \
    --num_train_epochs 10 \
    --word_embed_dim 200 \
    --hidden_ques_dim 512 \
    --hidden_ans_dim 512 \
    --hidden_hist_dim 128 \
    --hidden_cap_dim 128 \
    --dataloader_num_workers 4 \
    --eval_strategy epoch \
    --save_strategy epoch \
    --save_total_limit 2 \
    --load_best_model_at_end \
    --metric_for_best_model mrr \
    --greater_is_better true \
    --logging_steps 100 \
    --bf16 \
    --seed 0 \
    --report_to tensorboard

echo "Done. Model in $OUTPUT_DIR"
