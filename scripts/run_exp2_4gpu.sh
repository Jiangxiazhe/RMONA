#!/bin/bash
# Exp2 pMNIST: 4-GPU parallel, 5 methods × lr sweep × 3 seeds (39 runs)
#
#   GPU0: rmona(9)        GPU1: manifold_muon(9)
#   GPU2: cayley(9)       GPU3: expRNN(6) + cayleyRNN(6)
#
# Notes:
#   - Results are written to results/exp2_summary_gpu{N}.csv (with an lr column);
#     merge them afterwards.
#   - muon/mona are not rerun (Euclidean baselines; 2000 steps already showed
#     constraint collapse).
#   - --resume skips already-completed (seed, method, lr) runs.
set -e
cd "$(dirname "$0")/.."

STEPS=${STEPS:-12000}
SEEDS=${SEEDS:-"0 1 2"}
LR_GRID=${LR_GRID:-"0.005,0.01,0.02"}
PARAM_LR=${PARAM_LR:-"0.001"}
LOG_EVERY=200

mkdir -p results
echo "steps=$STEPS seeds=$SEEDS lr_grid=$LR_GRID param_lr=$PARAM_LR"

start_one() {  # $1=gpu $2=tag $3=methods...
    local gpu=$1 tag=$2; shift 2
    CUDA_VISIBLE_DEVICES=$gpu nohup python examples/exp2_pmnist.py \
        --steps "$STEPS" --seeds $SEEDS \
        --lr_grid "$LR_GRID" --param_lr_grid "$PARAM_LR" --log_every $LOG_EVERY \
        --out_tag "$tag" --resume --methods "$@" \
        > "results/exp2_sweep_$tag.log" 2>&1 &
    echo "  GPU$gpu -> results/exp2_sweep_$tag.log (pid $!)"
}

start_one 0 gpu0 rmona
start_one 1 gpu1 manifold_muon
start_one 2 gpu2 cayley
start_one 3 gpu3 expRNN cayleyRNN

echo "started. monitor: tail -f results/exp2_sweep_gpu*.log"
