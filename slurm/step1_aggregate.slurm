#!/bin/bash
# =============================================================================
# Step 1 -- Gaussian-copula aggregation, one task per ISO.
#
#   sbatch slurm/step1_aggregate.slurm
#
# MISO is the long pole (751 sites x 8760 hours); ERCOT and SPP finish sooner.
# =============================================================================
#SBATCH -A gts-phentenryck3-ai4opt
#SBATCH -J cop_agg
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --qos embers
#SBATCH --array=0-2
#SBATCH --output=/storage/scratch1/1/amoradi30/copula_cacp_results/logs/agg_%A_%a.out
#SBATCH --error=/storage/scratch1/1/amoradi30/copula_cacp_results/logs/agg_%A_%a.err

set -euo pipefail

REPO="/storage/home/hcoda1/1/amoradi30/copula-cacp"
mkdir -p /storage/scratch1/1/amoradi30/copula_cacp_results/logs

ISOS=(MISO SPP ERCOT)
ISO="${ISOS[$SLURM_ARRAY_TASK_ID]}"

echo "[$(date)] step 1: ${ISO}"
cd "$REPO"
python scripts/step1_aggregate.py --iso "$ISO" --quiet
echo "[$(date)] done: ${ISO}"
