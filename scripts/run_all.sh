#!/bin/bash
# Submit the whole pipeline, with step 2 held until step 1 finishes.
#
#   bash scripts/run_all.sh
#
# Steps 3 and 4 are cheap; run them by hand once the array jobs are done.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

echo "== step 1: copula aggregation =="
AGG_ID=$(sbatch --parsable slurm/step1_aggregate.slurm)
echo "   job ${AGG_ID}"

echo "== step 2: conformal calibration (after step 1) =="
CAL_ID=$(sbatch --parsable --dependency=afterok:"${AGG_ID}" slurm/step2_calibrate.slurm)
echo "   job ${CAL_ID}"

cat <<EOF

Submitted.  Watch with:  squeue -u \$USER

When both finish:
  python scripts/step3_table.py
  python scripts/step4_figures.py --iso MISO
EOF
