"""Forward-work accounting and LaTeX rows for the Table 3 K=3 comparison."""
from statistics import mean

LABELS = {"rezero": "Layer-specific ReZero", "rezero_stage": "Stage-shared ReZero"}
METRICS = ("fineweb_nll", "LD", "HS", "PQ", "WG", "ARC", "MMLU", "Avg")
FLOP_CONVENTION = (
    "Local forward-only estimate. Two FLOPs per multiply-accumulate for Q/K/V/O, the three MLP "
    "projections, causal QK and AV, linear routers and the vocabulary projection. Excludes "
    "elementwise operations (including gates), backward passes and checkpoint recomputation."
)


def forward_work(model, length, tokens, observed_token_layers, observed_causal_pairs=None):
    """Estimate Middle-Cycle forward work, including the padding in compacted batches."""
    routing = model.get("mor_routing", {})
    mor = routing.get("mor", {})
    rec = routing.get("recursive", {})
    expert = mor.get("expert", {})
    if (model.get("model_type") != "llama" or rec.get("sharing") != "middle_cycle"
            or mor.get("router_type") != "linear" or mor.get("rand_router")
            or routing.get("kv_sharing", {}).get("enable") is not False
            or not tokens or tokens % length or not observed_token_layers):
        return None
    k = rec["num_recursion"]
    depth, remainder = divmod(model["num_hidden_layers"] - 2, k)
    if remainder or depth < 1:
        return None
    details = {}
    if mor.get("type") == "token":
        if not observed_causal_pairs:
            return None
        causal_pairs = observed_causal_pairs
        router_outputs = tokens * k  # one K-way linear decision before the entire cycle
    elif mor.get("type") == "expert":
        if expert.get("cap_warmup_step") != 0 or expert.get("sampling") != "aux_loss" or not expert.get("include_first"):
            return None
        capacities = [float(x) for x in mor["capacity"].split(",")]
        if len(capacities) != k:
            return None
        sizes = [max(1, int(length * c)) for c in capacities]
        block_lengths = [length, length] + sizes * depth
        sequences = tokens // length
        if sequences * sum(block_lengths) != observed_token_layers:
            return None
        causal_pairs = sequences * sum(s * (s + 1) // 2 for s in block_lengths)
        router_outputs = sequences * (length + sum(sizes[:-1]))
        details["tokens_per_stage"] = sizes
    else:
        return None

    d = model["hidden_size"]
    heads = model["num_attention_heads"]
    head_dim = model.get("head_dim") or d // heads
    q = heads * head_dim
    kv = model["num_key_value_heads"] * head_dim
    matrices = 2 * d * q + 2 * d * kv + 3 * d * model["intermediate_size"]
    counts = {
        "decoder_projections": 2 * matrices * observed_token_layers,
        # QK and AV each cost 2*q FLOPs per visible (query, key) pair.
        "causal_attention": 4 * q * causal_pairs,
        "router_projections": 2 * d * router_outputs,
        "vocabulary_projection": 2 * d * model["vocab_size"] * tokens,
    }
    return {"total": sum(counts.values()), "components": counts, **details}


def table3_rows(report):
    """Build a two-row fragment only from paired, complete measurements at the released budget."""
    rows = {method: [r for r in report["local_runs"] if r["method"] == method] for method in LABELS}
    first, second = rows.values()
    seeds = sorted(r["seed"] for r in first)
    if not seeds or seeds != sorted(r["seed"] for r in second):
        return [], "Both ReZero variants need the same set of training seeds."
    all_rows = first + second
    if any(r["recipe_changes"] or not r["released_token_budget_reached"] for r in all_rows):
        return [], "These runs use a modified training recipe or token budget."
    if any(any(k not in r["metrics"] for k in METRICS) for r in all_rows):
        return [], "Downstream metrics are incomplete. Run evaluation with all Table 3 tasks."
    if any(not r["parameters"] or not r["forward_work"] for r in all_rows):
        return [], "Parameter counts or verified training-work records are unavailable."
    router = report["router_type"]
    if len({(r["step"], r["input_tokens"]) for r in all_rows}) != 1:
        return [], "Training steps or input tokens differ across the two variants."
    if router == "expert" and len({r["executed_token_layers"] for r in all_rows}) != 1:
        return [], "Fixed-capacity executed token-layers differ across the two variants."

    lines = []
    for method, runs in rows.items():
        counts = {r["parameters"]["non_embedding"] for r in runs}
        if len(counts) != 1:
            return [], "The parameter count varies between seeds of one method."
        params = counts.pop()
        flops = mean(r["forward_work"]["total"] for r in runs)
        means = report["local_summary"][method]
        cells = [LABELS[method], router.title(), "Cache", "M-Cyc", "3", f"{params / 1e6:.0f}M",
                 f"{flops / 1e18:.1f}" + r"$^{*}$", f"{runs[0]['input_tokens'] / 1e9:.2f}B"]
        cells += [format(means[k]["mean"], ".4f" if k == "fineweb_nll" else ".1f") for k in METRICS]
        lines.append(" & ".join(cells) + r" \\")
    return lines, "Paired training seeds " + ", ".join(map(str, seeds)) + "."


def write_table3(path, report):
    lines, note = table3_rows(report)
    content = ["% Bae et al., NeurIPS 2025, Table 3. Append inside the original 16-column tabular.",
               "% " + note]
    if lines:
        sample = next(iter(report["evaluation_data"].values()))
        shots = sorted({n for task in report["benchmark_protocol"].values() for n in task["n_shot"].values()
                        if isinstance(n, int)})
        content += [f"% Local FineWeb evaluation: {sample['packed_sequences']} packed sequences. "
                    f"Local harness shot counts: {shots}. See the paper JSON for per-task settings."]
        content += ["% Caption note: new rows are local seed means. * " + FLOP_CONVENTION,
                    "% Cite the published rows separately and report the local scoring protocol.", *lines]
    path.write_text("\n".join(content) + "\n")
    return note
