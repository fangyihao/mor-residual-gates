#!/bin/bash
set -e

launcher_type="accelerate"

if [[ "$1" == "deepspeed" ]] || [[ "$1" == "accelerate" ]]; then
  launcher_type="$1"
  shift
fi

if [[ "$1" == "online" ]] || [[ "$1" == "offline" ]]; then
  user_specified_run_mode="$1"
  shift
fi

if [ -n "$user_specified_run_mode" ]; then
  export WANDB_MODE="$user_specified_run_mode"
else
  export WANDB_MODE="online" # Default value if not specified
fi
echo "INFO: WANDB_MODE is set to '$WANDB_MODE'"

# Optional Hydra overrides after -- apply to every selected preset.
if [[ $# -lt 2 || -z "$1" || "$2" == -- ]]; then
  echo "Usage: $0 [accelerate|deepspeed] [online|offline] GPU_IDS {CONFIG [CONFIG ...]|all} [-- OPTIONS ...]" >&2
  exit 1
fi
gpu_numbers="$1"
shift
num_processes=$(echo "$gpu_numbers" | tr ',' '\n' | grep -c .)
config_names=()
while [[ $# -gt 0 && "$1" != -- ]]; do
  config_names+=("$1")
  shift
done
if [[ $# -gt 0 ]]; then
  shift
fi

for config_name in "${config_names[@]}"; do
  if [[ "$config_name" == all ]]; then
    if [[ ${#config_names[@]} -ne 1 ]]; then
      echo "Use all as the sole preset and pass suite options after --." >&2
      exit 2
    fi
    exec python "$(dirname "$0")/run_residual_suite.py" "$launcher_type" "$gpu_numbers" "$@"
  fi
done

for config_name in "${config_names[@]}"; do
  # Generate a 5-digit random port.
  random_port=$((10000 + RANDOM % 90000))

  # Execute the command.
  echo "Running with config: $config_name, GPUs: $gpu_numbers, Port: $random_port"

  if [ "$launcher_type" == "deepspeed" ]; then
    echo "Launch with DeepSpeed..."
    HYDRA_FULL_ERROR=1 deepspeed --include "localhost:$gpu_numbers" --no_local_rank --master_port "$random_port" pretrain.py --config-name "$config_name" "$@"
  elif [ "$launcher_type" == "accelerate" ]; then
    echo "Launch with accelerate..."
    if [ "$num_processes" -eq 1 ]; then
      # Use single GPU config for single process.
      HYDRA_FULL_ERROR=1 accelerate launch --config_file acc_configs/single_gpu_config.yaml --gpu_ids $gpu_numbers --num_processes $num_processes --main_process_port "$random_port" pretrain.py --config-name "$config_name" "$@"
    else
      # Use default config for multiple processes.
      HYDRA_FULL_ERROR=1 accelerate launch --config_file acc_configs/default_config.yaml --gpu_ids $gpu_numbers --num_processes $num_processes --main_process_port "$random_port" pretrain.py --config-name "$config_name" "$@"
    fi
  fi
  # Optional: Add a delay after each config execution.
  # sleep 5 # Wait for 5 seconds.
done

echo "All configurations processed."