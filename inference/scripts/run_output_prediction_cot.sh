#!/bin/bash

set -euo pipefail

dirs=(
  "Qwen3-14B-Base-deepscaler-MASA"
  "Qwen3-14B-Base-deepscaler-GRPO"
)

models=(
  "yjyjyj98/Qwen3-14B-Base-deepscaler-MASA"
  "yjyjyj98/Qwen3-14B-Base-deepscaler-GRPO"
)

temperatures=(0.6)

mkdir -p slurm_logs

# How many array tasks? Keep this in sync with "#SBATCH --array=0-1"
ARRAY_BEGIN=0
ARRAY_END=1

have_sbatch=false
if command -v sbatch >/dev/null 2>&1; then
  have_sbatch=true
fi

for ((i=0; i<${#models[@]}; i++)); do
  model=${models[$i]}
  base_dir=${dirs[$i]}
  echo "Model: $model"

  for temperature in "${temperatures[@]}"; do
    dir="${base_dir}+cot_temp${temperature}_output"

    # Create the task script (bash will treat #SBATCH lines as comments when run locally)
    cat > temp_sbatch_script.sh <<'EOS'
#!/bin/bash
#SBATCH --output=slurm_logs/slurm-%A-%a.out
#SBATCH --error=slurm_logs/slurm-%A-%a.err
#SBATCH --partition=YOUR_PARTITION_HERE
#SBATCH --array=0-1
#SBATCH --cpus-per-task=10
#SBATCH --gpus=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=0GB
#SBATCH --time=03:00:00

# ----- Runtime parameters injected below when we generate this file -----
EOS

    # Append runtime vars (expanded now)
    cat >> temp_sbatch_script.sh <<EOS
dir="${dir}"
MODEL="${model}"
TEMPERATURE="${temperature}"
SIZE=800
GPUS=1

i=\${SLURM_ARRAY_TASK_ID:-0}
ip=\$((i+1))

echo "Output dir: \$dir"
mkdir -p "model_generations_raw/\$dir"

echo "Starting iteration \$i with start and end  \$((i*SIZE/GPUS)) \$((ip*SIZE/GPUS))"

python main.py \
  --model "\$MODEL" \
  --use_auth_token \
  --trust_remote_code \
  --tasks output_prediction \
  --batch_size 10 \
  --n_samples 32 \
  --max_length_generation 16384 \
  --precision bf16 \
  --limit "\$SIZE" \
  --temperature "\$TEMPERATURE" \
  --save_generations \
  --save_generations_path "model_generations_raw/\${dir}/shard_\$i.json" \
  --start "\$((i*SIZE/GPUS))" \
  --end "\$((ip*SIZE/GPUS))" \
  --tensor_parallel_size 8 \
  --cot \
  --shuffle
EOS

    chmod +x temp_sbatch_script.sh

    if $have_sbatch; then
      # --- SLURM path ---
      # ⚠️ Set a real partition name:
      #    sed -i 's/YOUR_PARTITION_HERE/gpuq/' temp_sbatch_script.sh
      echo "[SLURM] Submitting with sbatch"
      sbatch temp_sbatch_script.sh
      rm -f temp_sbatch_script.sh
    else
      # --- Local fallback: emulate array 0..1 by looping ---
      echo "[LOCAL] sbatch not found — running tasks locally as a loop"
      for ((task_id=$ARRAY_BEGIN; task_id<=$ARRAY_END; task_id++)); do
        echo "[LOCAL] Running array task $task_id"
        SLURM_ARRAY_TASK_ID=$task_id bash ./temp_sbatch_script.sh
      done
      rm -f temp_sbatch_script.sh
    fi

  done
done
