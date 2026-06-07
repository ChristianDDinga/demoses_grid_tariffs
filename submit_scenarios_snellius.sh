#!/bin/bash
#SBATCH --job-name="DEMOSES-grid-tariffs"
#SBATCH --partition=rome
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=2G
#SBATCH --array=0-5%6   # scenarios 1-5 (core) + 6 (exploratory); lower %N if limited by the Gurobi license
#SBATCH --requeue

# Submit from the repository root (where run_simulation.py and .venv live):
#   sbatch submit_scenarios_snellius.sh
#
# Each array task: (1) generates this scenario's tariff parameter files into
# results/<config-stem>/inputs/tariff_params/, then (2) runs the full simulation.
# run_simulation.py resolves the bare tariff filenames in each config against that folder.

# The withdrawal weighting schedule and all generation settings (base rate, growth, base/target
# year, holidays) are read from each config's `tariffs.generation` block; edit the config to change
# them (e.g. swap to data/weighting_factors_transmission.csv for a 150 kV sensitivity).

# Scenario config files, ordered 1..6 (index 0..5). Scenario 6 is exploratory.
CONFIGS=(
  "configs/scenarios/scenario_1_flat_volumetric.yaml"
  "configs/scenarios/scenario_2_capacity_withdrawal.yaml"
  "configs/scenarios/scenario_3_tou_capacity_withdrawal.yaml"
  "configs/scenarios/scenario_4_tou_capacity_volumetric_withdrawal.yaml"
  "configs/scenarios/scenario_5_tou_capacity_both.yaml"
  "configs/scenarios/scenario_6_tou_capacity_volumetric_both.yaml"
)

CONFIG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
SCENARIO_NAME="$(basename "$CONFIG" .yaml)"

LOG_DIR="logs/${SCENARIO_NAME}"
mkdir -p "$LOG_DIR"
exec > >(tee "${LOG_DIR}/${SCENARIO_NAME}.out")
exec 2>&1

echo "Array task ${SLURM_ARRAY_TASK_ID}: scenario '${SCENARIO_NAME}' from ${CONFIG}"
echo "Started at: $(date)"

# Load required modules.
module load 2024
module load Gurobi/12.0.1-GCCcore-13.3.0
export GRB_LICENSE_FILE=/home/cdohdinga/gurobi.lic

# Activate the virtual environment.
source .venv/bin/activate

# Switch to the main git branch
git switch main

# 1. Generate this scenario's tariff parameter files into results/<stem>/inputs/tariff_params/.
python src/demoses_grid_tariffs/generate_tariffs.py --config "$CONFIG"

# 2. Run the full workflow (DHN optimization -> prosumers -> AC power flow) for this scenario.
srun python run_simulation.py --config "$CONFIG"

echo "Finished at: $(date)"
