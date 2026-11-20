#!/usr/bin/env bash
# Launch one torchrun worker per listed Ascend device on this host.
set -eu
usage() {
    echo "Usage: bash scripts/run_npu.sh {train|all|eval|check} DEVICE_IDS [experiment arguments...]" >&2
    exit 2
}
[[ $# -ge 2 ]] || usage
command=$1
device_ids=$2
shift 2
[[ $device_ids =~ ^(0|[1-9][0-9]*)(,(0|[1-9][0-9]*))*$ ]] || usage
IFS=, read -ra devices <<< "$device_ids"
previous=-1
for id in "${devices[@]}"; do
    if (( id <= previous )); then
        echo "Ascend device IDs must be distinct and in ascending order." >&2
        exit 2
    fi
    previous=$id
done

nnodes=${NNODES:-1}
[[ $nnodes =~ ^[1-9][0-9]*$ ]] || usage
if [[ $nnodes == 1 ]]; then
    rendezvous=(--standalone --local-addr=127.0.0.1)
else
    : "${MASTER_ADDR:?Set MASTER_ADDR to the rank-zero host for a multi-node run}"
    : "${NODE_RANK:?Set NODE_RANK to the zero-based index of this host}"
    [[ $NODE_RANK =~ ^(0|[1-9][0-9]*)$ ]] && (( NODE_RANK < nnodes )) || usage
    rendezvous=(--nnodes="$nnodes" --node-rank="$NODE_RANK" --master-addr="$MASTER_ADDR" --master-port="${MASTER_PORT:-29500}")
fi

cd "$(dirname "$0")/.."
case "$command" in
    all) exec python scripts/run_residual_suite.py npu "$device_ids" "$@" ;;
    train)
        program=(pretrain.py --config-name "${CONFIG_NAME:-recursive_residual}" attn_implementation=sdpa
                 'hydra.run.dir=outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}/rank_${oc.env:RANK}')
        ;;
    eval) program=(evaluate_fineweb_test.py) ;;
    check) [[ $# == 0 ]] || usage; program=(--module util.device) ;;
    *) usage ;;
esac
export ASCEND_RT_VISIBLE_DEVICES="$device_ids" MOR_DEVICE=npu
export ACCELERATE_USE_CPU=false ACCELERATE_USE_DEEPSPEED=false
exec python -m torch.distributed.run "${rendezvous[@]}" --nproc-per-node="${#devices[@]}" "${program[@]}" "$@"
