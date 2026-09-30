#!/bin/bash
# =============================================================================
# Step 2 -- conformal calibration.
#
# 18 tasks = 3 ISOs x 6 (source, method) cells, i.e. every row of Table 1.
#
#   iso_idx  = TASK_ID / 6      (0=MISO, 1=SPP, 2=ERCOT)
#   cell_idx = TASK_ID % 6      (see CELLS below)
#
#   sbatch slurm/step2_calibrate.slurm                 # everything
#   sbatch --array=0-5 slurm/step2_calibrate.slurm     # MISO only
#
# Requires step 1 to have produced the aggregated forecasts.
# =============================================================================
#SBATCH -A gts-phentenryck3-ai4opt
#SBATCH -J cop_cal
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --qos embers
#SBATCH --array=0-17
#SBATCH --output=/storage/scratch1/1/amoradi30/copula_cacp_results/logs/cal_%A_%a.out
#SBATCH --error=/storage/scratch1/1/amoradi30/copula_cacp_results/logs/cal_%A_%a.err

set -euo pipefail

REPO="/storage/home/hcoda1/1/amoradi30/copula-cacp"
RESULTS="/storage/scratch1/1/amoradi30/copula_cacp_results"
mkdir -p "$RESULTS/logs"

ISOS=(MISO SPP ERCOT)
# "source:method" -- the six rows of Table 1
CELLS=("nrel:none" "nrel:cqr" "nrel:kernel" "copula:none" "copula:cqr" "copula:kernel")

N_CELLS=${#CELLS[@]}
ISO="${ISOS[$((SLURM_ARRAY_TASK_ID / N_CELLS))]}"
CELL="${CELLS[$((SLURM_ARRAY_TASK_ID % N_CELLS))]}"
SOURCE="${CELL%%:*}"
METHOD="${CELL##*:}"

echo "[$(date)] step 2: ${ISO} ${SOURCE}+${METHOD}"
cd "$REPO"
python scripts/step2_calibrate.py \
    --iso "$ISO" \
    --source "$SOURCE" \
    --method "$METHOD" \
    --log_dir "$RESULTS/logs/tuning/${ISO}" \
    --verbose 2
echo "[$(date)] done: ${ISO} ${SOURCE}+${METHOD}"
