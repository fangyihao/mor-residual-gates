"""Fixed, layer-specific, and stage-shared scaling of repeated decoder blocks."""
import math

from torch import nn


LEARNED_GATE_MODES = {"rezero", "rezero_stage"}


def stage_scales(mode, num_recursion, lambda_r=1.0, zeta=1.0):
    """Coefficients for j=2,...,K. None denotes a trainable zero-initialized gate."""
    if num_recursion < 1:
        raise ValueError("num_recursion must be positive")
    if mode not in LEARNED_GATE_MODES and mode not in {"full", "uniform", "power"}:
        raise ValueError(f"Unknown recursive residual mode {mode!r}")
    if not all(math.isfinite(v) and v >= 0 for v in (lambda_r, zeta)):
        raise ValueError("lambda_r and zeta must be finite and nonnegative")
    if mode in LEARNED_GATE_MODES:
        return [None] * (num_recursion - 1)
    if mode == "full":
        return [1.0] * (num_recursion - 1)
    # Divide each power weight by the j=2 weight before normalization. The
    # common factor cancels and the denominator stays >= 1 even for large zeta.
    weights = [(2 / j) ** (zeta if mode == "power" else 0.0) for j in range(2, num_recursion + 1)]
    total = sum(weights)
    return [lambda_r * (w / total) for w in weights]


def configure_residual_layers(model, recursive):
    settings = recursive.get("residual")
    if settings is None:
        return
    layout = dict(settings, sharing=recursive.sharing, num_recursion=recursive.num_recursion)
    model.config.recursive_residual = layout
    install_residual_layers(model.model.layers, layout)


def install_residual_layers(layers, layout):
    """Map virtual decoder positions to reuse stages after weight sharing/trimming."""
    mode = layout["mode"]
    k = layout["num_recursion"]
    scales = stage_scales(mode, k, layout.get("lambda_r", 1.0), layout.get("zeta", 1.0))
    sharing = layout["sharing"]
    if sharing not in {"cycle", "sequence", "middle_cycle", "middle_sequence"}:
        raise ValueError(f"Unknown sharing layout {sharing!r}")
    offset = int(sharing.startswith("middle_"))
    length = len(layers) - 2 * offset
    if length <= 0 or length % k:
        raise ValueError("Shared decoder depth must be a positive multiple of num_recursion")
    base_depth = length // k
    if mode == "rezero_stage":
        # A shared module keeps the scalar tied even when device placement
        # replaces Parameter objects in individual decoder modules.
        weight = layers[offset].input_layernorm.weight
        stage_gates = [nn.ParameterDict({"recursive_gate": weight.new_zeros(())}) for _ in scales]
    for i, layer in enumerate(layers):
        # Trimming can change the virtual cache index of a retained layer.
        layer.self_attn.layer_idx = i
        pos = i - offset
        if not 0 <= pos < length:
            continue
        stage = pos // base_depth if sharing.endswith("cycle") else pos % k
        if stage == 0 or mode == "full":
            continue
        scale = scales[stage - 1]
        if mode == "rezero_stage":
            layer.stage_gate = stage_gates[stage - 1]
        elif scale is None:
            layer.recursive_gate = nn.Parameter(layer.input_layernorm.weight.new_zeros(()))
        else:
            layer.recursive_gate = scale
